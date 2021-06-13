import argparse
import os
import sys
import torch
import yaml
import pandas as pd
from thop import profile, clever_format
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as transforms

from utils.general import check_file
from utils.datasets import create_dataloader, create_dataloader_with_dataset, LoadImagesAndLabels
from model import Model, set_bn_eval
from utils.cgd_utils import recall, LabelSmoothingCrossEntropyLoss, BatchHardTripletLoss, ImageReader, MPerClassSampler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(net, optim):
    net.train()
    # fix bn on backbone network
    net.apply(set_bn_eval)
    total_loss, total_correct, total_num, data_bar = 0, 0, 0, tqdm(train_data_loader, file=sys.stdout)
    for inputs, labels in data_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        features, classes = net(inputs)
        class_loss = class_criterion(classes, labels)
        feature_loss = feature_criterion(features, labels)
        loss = class_loss + feature_loss
        optim.zero_grad()
        loss.backward()
        optim.step()
        pred = torch.argmax(classes, dim=-1)
        total_loss += loss.item() * inputs.size(0)
        total_correct += torch.sum(pred == labels).item()
        total_num += inputs.size(0)
        data_bar.set_description('Train Epoch {}/{} - Loss:{:.4f} - Acc:{:.2f}%'
                                 .format(epoch, num_epochs, total_loss / total_num, total_correct / total_num * 100))

    return total_loss / total_num, total_correct / total_num * 100


def test(net, recall_ids):
    net.eval()
    with torch.no_grad():
        # obtain feature vectors for all data
        for key in eval_dict.keys():
            eval_dict[key]['features'] = []
            for inputs, labels in tqdm(eval_dict[key]['data_loader'], desc='processing {} data'.format(key)):
                inputs, labels = inputs.to(device), labels.to(device)
                features, classes = net(inputs)
                eval_dict[key]['features'].append(features)
            eval_dict[key]['features'] = torch.cat(eval_dict[key]['features'], dim=0)

        # compute recall metric
        # if data_name == 'isc':
        acc_list = recall(eval_dict['test']['features'], query_data_set.labels, recall_ids,
                          eval_dict['gallery']['features'], gallery_data_set.labels)
        # else:
        #     acc_list = recall(eval_dict['test']['features'], query_data_set.labels, recall_ids)
    desc = 'Test Epoch {}/{} '.format(epoch, num_epochs)
    for index, rank_id in enumerate(recall_ids):
        desc += 'R@{}:{:.2f}% '.format(rank_id, acc_list[index] * 100)
        results['test_recall@{}'.format(rank_id)].append(acc_list[index] * 100)
    print(desc)
    return acc_list[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CGD')
    parser.add_argument('--data_path', default='data/image_objects_cgd.yaml', type=str, help='datasets path')
    parser.add_argument('--data_name', default='acm', type=str, choices=['acm'],
                        help='dataset name')
    parser.add_argument('--crop_type', default='uncropped', type=str, choices=['uncropped', 'cropped'],
                        help='crop data or not, it only works for car or cub dataset')
    parser.add_argument('--backbone_type', default='resnet50', type=str, choices=['resnet50', 'resnext50'],
                        help='backbone network type')
    parser.add_argument('--gd_config', default='SG', type=str,
                        choices=['S', 'M', 'G', 'SM', 'MS', 'SG', 'GS', 'MG', 'GM', 'SMG', 'MSG', 'GSM'],
                        help='global descriptors config')
    parser.add_argument('--feature_dim', default=1536, type=int, help='feature dim')
    parser.add_argument('--smoothing', default=0.1, type=float, help='smoothing value for label smoothing')
    parser.add_argument('--temperature', default=0.5, type=float,
                        help='temperature scaling used in softmax cross-entropy loss')
    parser.add_argument('--margin', default=0.1, type=float, help='margin of m for triplet loss')
    parser.add_argument('--recalls', default='1,2,4,8', type=str, help='selected recall')
    parser.add_argument('--batch_size', default=128, type=int, help='train batch size')
    parser.add_argument('--num_epochs', default=20, type=int, help='train epoch number')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to existing model')
    parser.add_argument('--save-dir', dest='save_dir', help='The directory used to save the trained models',
                        default='save_temp', type=str)

    opt = parser.parse_args()
    # args parse
    data_path, data_name, crop_type, backbone_type = opt.data_path, opt.data_name, opt.crop_type, opt.backbone_type
    gd_config, feature_dim, smoothing, temperature = opt.gd_config, opt.feature_dim, opt.smoothing, opt.temperature
    margin, recalls, batch_size = opt.margin, [int(k) for k in opt.recalls.split(',')], opt.batch_size
    num_epochs = opt.num_epochs

    # Check the save_dir exists or not
    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)

    save_name_pre = '{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(data_name, crop_type, backbone_type, gd_config, feature_dim,
                                                        smoothing, temperature, margin, batch_size)

    results = {'train_loss': [], 'train_accuracy': []}
    for recall_id in recalls:
        results['test_recall@{}'.format(recall_id)] = []

    # Load data
    opt.data_path = check_file(opt.data_path)
    with open(opt.data_path) as f:
        data_dict = yaml.load(f)  # model dict

    train_path = data_dict['train']
    query_path = data_dict['query']
    gallery_path = data_dict['gallery']

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.Resize((252, 252)), transforms.RandomCrop(224),
                                    transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
    # Trainloader
    train_data_set = LoadImagesAndLabels(train_path, transform=transform, cache_images=False, use_instance_id=False)
    train_sample = MPerClassSampler(train_data_set.labels, batch_size)
    train_data_loader = create_dataloader_with_dataset(train_data_set, opt.batch_size, sampler=None)

    test_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
    query_data_loader, query_data_set = create_dataloader(query_path, opt.batch_size, cache=False,
                                                          transform=test_transform, use_instance_id=False)
    eval_dict = {'test': {'data_loader': query_data_loader}}
    gallery_data_loader, gallery_data_set = create_dataloader(gallery_path, opt.batch_size, cache=False,
                                                              transform=test_transform, use_instance_id=False)
    eval_dict['gallery'] = {'data_loader': gallery_data_loader}

    # model setup, model profile, optimizer config and loss definition
    model = Model(backbone_type, gd_config, feature_dim, num_classes=23).to(device)
    # model = Model(backbone_type, gd_config, feature_dim, num_classes=len(train_data_set.class_to_idx)).to(device)
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> resume, loading existing model '{}'".format(opt.resume))
            state_dict = torch.load(opt.resume)
            model.load_state_dict(state_dict, strict=False)
        else:
            print("=> no model found at '{}'".format(opt.resume))

    flops, params = profile(model, inputs=(torch.randn(1, 3, 224, 224).to(device),))
    flops, params = clever_format([flops, params])
    print('# Model Params: {} FLOPs: {}'.format(params, flops))
    optimizer = Adam(model.parameters(), lr=1e-4)
    lr_scheduler = MultiStepLR(optimizer, milestones=[int(0.6 * num_epochs), int(0.8 * num_epochs)], gamma=0.1)
    class_criterion = LabelSmoothingCrossEntropyLoss(smoothing=smoothing, temperature=temperature)
    feature_criterion = BatchHardTripletLoss(margin=margin)

    best_recall = 0.0
    for epoch in range(1, num_epochs + 1):
        train_loss, train_accuracy = train(model, optimizer)
        results['train_loss'].append(train_loss)
        results['train_accuracy'].append(train_accuracy)
        rank = test(model, recalls)
        lr_scheduler.step()

        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv(os.path.join(opt.save_dir, 'results/{}_statistics.csv'.format(save_name_pre)),
                         index_label='epoch')
        # save database and model
        data_base = {}
        if rank > best_recall:
            best_recall = rank
            data_base['test_images'] = query_data_set.images
            data_base['test_labels'] = query_data_set.labels
            data_base['test_features'] = eval_dict['test']['features']

            data_base['gallery_images'] = gallery_data_set.images
            data_base['gallery_labels'] = gallery_data_set.labels
            data_base['gallery_features'] = eval_dict['gallery']['features']
            torch.save(data_base, os.path.join(opt.save_dir, 'results/{}_data_base.pth'.format(save_name_pre)))
            torch.save(model.state_dict(), os.path.join(opt.save_dir, 'results/{}_model.pth'.format(save_name_pre)))
