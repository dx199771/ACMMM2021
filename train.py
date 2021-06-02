import os, json, cv2
import random


def lable_lookup(lookup):
    labels = ["short sleeve top", "long sleeve top", "short sleeve shirt", "long sleeve shirt", "vest top", "sling top",
              "sleeveless top", "short outwear", "short vest", "long sleeve dress", "short sleeve dress",
              "sleeveless dress", "long vest", "long outwear", "bodysuit", "classical", "short skirt", "medium skirt",
              "long skirt", "shorts", "medium shorts", "trousers", "overalls"]
    return labels.index(lookup)

def generate_annotation(data,name):

    for i in range(len(data)):
        label = data[i]["label"]
        box = abbox_to_relbox(data[i]["box"],800,800)
        print(lable_lookup(label),box[0],box[1],box[2],box[3])
        f = open("./data/train/labels/{}.txt".format(name), "a")
        f.write("{} {} {} {} {}\n".format(lable_lookup(label),box[0],box[1],box[2],box[3]))
        f.close()
    return 0

def generate_frame_image(path,frame,name):
    print(name)
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture(path)
    cap.set(1, frame)
    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    ret, frame = cap.read()
    if ret == True:
        # Save the resulting frame
        cv2.imwrite("./data/train/images/"+name+".jpg", frame)

    cap.release()
    # Closes all the frames
    cv2.destroyAllWindows()
    return 0

def abbox_to_relbox(box,width,height):
    x1,y1,x2,y2 = box
    new_x1 = 1 if (x1+x2)/2/width>1 else (x1+x2)/2/width
    new_y1 = 1 if (y1+y2)/2/height>1 else (y1+y2)/2/height
    new_x2 = 1 if (x2-x1)/width >1 else (x2-x1)/width
    new_y2 = 1 if (y2-y1)/height>1 else (y2-y1)/height

    return [new_x1,new_y1,new_x2,new_y2]


def get_yolo_data(data_path, video_path):
    data_list = os.listdir(data_path)
    for i in data_list[:5000]:
        data_anno_path = os.path.join(data_path,i)
        with open(data_anno_path) as f:
            data = json.load(f)
            for j in range(10):
                name = "{}{}".format(i,j)
                data_ = data["frames"][j]["annotations"]
                frame_num = data["frames"][j]["frame_index"]
                generate_annotation(data_, name)
                video_path_ = os.path.join(video_path, i[:-5]+".mp4")
                generate_frame_image(video_path_, frame_num, name)
"""
data_path = "../train_dataset_part1/video_annotation"
video_path = "../train_dataset_part1/video"

get_yolo_data(data_path,video_path)
data_path = "../train_dataset_part2/video_annotation"
video_path = "../train_dataset_part2/video"

get_yolo_data(data_path,video_path)
data_path = "../train_dataset_part3/video_annotation"
video_path = "../train_dataset_part3/video"

get_yolo_data(data_path,video_path)
data_path = "../train_dataset_part4/video_annotation"
video_path = "../train_dataset_part4/video"

get_yolo_data(data_path,video_path)
data_path = "../train_dataset_part5/video_annotation"
video_path = "../train_dataset_part5/video"

get_yolo_data(data_path,video_path)
data_path = "../train_dataset_part6/video_annotation"
video_path = "../train_dataset_part6/video"

get_yolo_data(data_path,video_path)
data_path = "../train_dataset_part7/video_annotation"
video_path = "../train_dataset_part7/video"

get_yolo_data(data_path,video_path)
"""

"""
    Yolo data annotation for image data
"""

def generate_image_annotation(data,name):
    with open(data) as f:
        data = json.load(f)
        img_name = data["img_name"]
        item_id = data["item_id"]
        item_anno = data["annotations"]
        h,w = get_img_data(img_path, img_name, item_id)
        print(h,w,img_name,item_id)
        get_img_label(img_name,item_id,item_anno,h,w)
def get_img_label(name, id, anno,h ,w):
    for i in anno:
        label = i["label"]
        anno_ = abbox_to_relbox(i["box"], w,h)
        f = open("./all_img_data/{}/train/labels/{}_{}.txt".format(dataset,id,name[:-4]), "a")
        f.write("{} {} {} {} {}\n".format(lable_lookup(label),anno_[0],anno_[1],anno_[2],anno_[3]))
        f.close()

def get_img_data(path, name, id):
    img_files = os.path.join(path, id, name)
    img = cv2.imread(img_files)
    cv2.imwrite("./all_img_data/{}/train/images/{}_{}".format(dataset,id,name),img)
    (h,w,_) = img.shape
    return h,w
#def create_img_annotation():
data_path = "../train_dataset_part5/image_annotation"
img_path = "../train_dataset_part5/image"
dataset = "dataset5"
data_list = os.listdir(data_path)
length = len(data_list)
for i, v in enumerate(data_list):
    print(i,"/",length)
    image_path = os.path.join(data_path,v)
    data_list = os.listdir(image_path)
    for j in data_list:
        generate_image_annotation(os.path.join(image_path,j), v)

print(data_list)
"""
data_path = "../train_dataset_part5/image_annotation"
img_path = "../train_dataset_part5/image"
dataset = "dataset5"
data_list = os.listdir(data_path)
length = len(data_list)
for i, v in enumerate(data_list):
    print(i,"/",length)
    image_path = os.path.join(data_path,v)
    data_list = os.listdir(image_path)
    for j in data_list:
        generate_image_annotation(os.path.join(image_path,j), v)

print(data_list)


all_test = os.listdir("../test_dataset_A/test_dataset_A/image")
for i in all_test:
    path = os.path.join("../test_dataset_A/test_dataset_A/image",i)
    path_ = os.listdir(path)
    for j in path_:
        print(os.path.join(path,j))
        img = cv2.imread(os.path.join(path,j))
        name = "{}{}.jpg".format(i,j)
        print(name)
        cv2.imwrite(name,img)

"""
