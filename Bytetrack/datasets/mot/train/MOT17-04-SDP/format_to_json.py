import json 
tracking ={}


import cv2
import pandas as pd

# Load the ground truth file
gt_path = '/home/sophie/uncertain-identity-aware-tracking/Bytetrack/tools/datasets/mot/train/MOT17-04-SDP/gt/gt.txt'
ground_truth = pd.read_csv(gt_path, header=None)

# Assign column names based on MOT17 format
ground_truth.columns = ['frame', 'id', 'bbox_left', 'bbox_top', 'bbox_width', 'bbox_height', 'confidence', 'class', 'visibility']

# Filter ground truth data for a specific frame
frame_number = 1  # Change this to the desired frame number
frame_data = ground_truth[ground_truth['frame'] == frame_number]

# Load the corresponding image
img_path = f'/home/sophie/uncertain-identity-aware-tracking/Bytetrack/tools/datasets/mot/train/MOT17-04-SDP/img1/{frame_number:06d}.jpg'  # MOT17 images are typically named with 6 digits
image = cv2.imread(img_path)

# Draw the bounding boxes on the image
for index, row in frame_data.iterrows():
    x, y, w, h = int(row['bbox_left']), int(row['bbox_top']), int(row['bbox_width']), int(row['bbox_height'])
    # Draw a rectangle (bounding box)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box with thickness 2

    # Optionally, add the object ID
    object_id = row['id']
    cv2.putText(image, f'ID: {object_id}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Save or display the image with bounding boxes
cv2.imwrite('/home/sophie/uncertain-identity-aware-tracking/Bytetrack/tools/datasets/mot/train/MOT17-04-SDP/visualize/output_image_with_bboxes.png', image)
# cv2.imshow('Image', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()




with open('/home/sophie/uncertain-identity-aware-tracking/Bytetrack/tools/datasets/mot/train/MOT17-04-SDP/gt/gt.txt', 'r') as file:
    # Read each line in the file
    for line in file:
        # Remove any trailing whitespace/newline characters and split by comma
        #print(line.strip())
        # ['frame', 'id', 'bbox_left', 'bbox_top', 'bbox_width', 'bbox_height', 'confidence', 'class', 'visibility']

        
        
        frame_id, id, x,y,w,h,conf,_,_ = line.strip().split(',')
        print(line.strip().split(','))
        print(frame_id,id,x,y,w,h)
        # Print the extracted elements for each line
        if frame_id not in tracking.keys():
            tracking[frame_id]={}
        tracking[frame_id][id]=[x,y,w,h]
        #break

with open("tracking_formated.json", "w") as file:
    json.dump(tracking,file)