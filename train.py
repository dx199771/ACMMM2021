import os, json, cv2
import numpy as np




data_path = "../data/train_dataset_part1/video_annotation"
video_path = "../data/train_dataset_part1/video/000041.mp4"
video_extract = "../data/video_extract"

data_list = os.listdir(data_path)
for i in data_list:
    data_anno_path = os.path.join(data_path,i)
    with open(data_anno_path) as f:
        data = json.load(f)
        data = data["frames"]
    #print(data)
        #print(data["frames"][0]["frame_index"])

def get_annotation():
    return 0

def abbox_to_relbox(box,width,height):
    x1,y1,x2,y2 = box
    new_x1 = (x1+x2)/2/width
    new_y1 = (y1+y2)/2/height
    new_x2 = (x2-x1)/width
    new_y2 = (y2-y1)/height

    return [new_x1,new_y1,new_x2,new_y2]

print(abbox_to_relbox([88,602,450,959],540,960))

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture(video_path)

# Check if camera opened successfully
if (cap.isOpened()== False):
    print("Error opening video stream or file")

# Read until video is completed
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:

        # Display the resulting frame
        cv2.imshow('Frame',frame)
        cv2.imwrite("", frame)
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

cap.release()
# Closes all the frames
cv2.destroyAllWindows()