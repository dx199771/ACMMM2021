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
        box = abbox_to_relbox(data[i]["box"],540,960)
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
    new_x1 = (x1+x2)/2/width
    new_y1 = (y1+y2)/2/height
    new_x2 = (x2-x1)/width
    new_y2 = (y2-y1)/height

    return [new_x1,new_y1,new_x2,new_y2]

data_path = "../data/train_dataset_part2/video_annotation"
video_path = "../data/train_dataset_part2/video"
video_extract = "../data/video_extract"

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


