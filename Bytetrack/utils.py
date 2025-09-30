import cv2 
import os
import json
def put_results_on_video(track, save_path, video_path=None, image_path=None):
    """not wel done yet for image_paths, you can use fairmot notebook for that

    Args:
        track (_type_): _description_
        save_path (_type_): _description_
        video_path (_type_, optional): _description_. Defaults to None.
        image_path (_type_, optional): _description_. Defaults to None.
    """
    
    if video_path is not None:
        cap = cv2.VideoCapture(video_path)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        vid_writer = cv2.VideoWriter(save_path,  cv2.VideoWriter_fourcc(*'mp4v') , fps, (int(width), int(height)))     
        # Center coordinates
        center_coordinates = (625, 70)
        # Radius of circle
        radius = 2
        # Blue color in BGR
        color = (255, 0, 0)
        # Line thickness of 2 px
        thickness = 2
        ret_val, frame = cap.read()
        frame_id=1
        while ret_val : 
            ret_val, frame = cap.read()
            #add feeder and drinker center 
            frame = cv2.circle(frame, center_coordinates, radius, color, thickness)
            frame = cv2.circle(frame, (90,102), radius, color, thickness)
            #addd
            if str(frame_id) in track.keys():
                if (frame_id!="0") :
                    cv2.putText(frame, str(frame_id),(90+580, 20),0, 5e-3 * 200, (0,255,0),2)
                    cv2.rectangle(frame, (580, 20), (90+580, 115+20) ,(255,255,255), 2)

                    for track_id,tlwh in track[str(frame_id)].items():
                        tid= str(track_id)   
                        cv2.rectangle(frame, (int(tlwh[0]), int(tlwh[1])), (int(tlwh[0])+int(tlwh[2]), int(tlwh[1])+int(tlwh[3])) ,(255,255,255), 2)
                        cv2.putText(frame, str(tid),(int(tlwh[0]), int(tlwh[1])),0, 5e-3 * 200, (0,255,0),2)
                        
                    vid_writer.write(frame)
                    #print(frame_id)
            else:
                #vid_writer.write(frame)
                a=0
            #print("\n", "\n")
            frame_id=frame_id+1
        
        #print("a frame done")
        vid_writer.release()
        print("video done at", save_path)
    else: 
        # Directory containing images
        image_folder = image_path
        video_name = save_path

        # Collect images from the folder
        images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
        images.sort()  # Ensure they are in the correct order

        # Get the dimensions of the images
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape

        # Define the codec and create VideoWriter object
        #fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        print(video_name, cv2.VideoWriter_fourcc(*"mp4v"), 1, (width, height))
        vid_writer = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*"mp4v"), 1, (width, height))

        # Write each image to the video
        for image in images:
            frame_id = int(image.split(".")[0])
            if frame_id >525:
                frame_id -=525
                frame = cv2.imread(os.path.join(image_folder, image))
                if str(frame_id) in track.keys():
                    if (frame_id!="0") :
                        #print(frame_id)
                        cv2.putText(frame, str(frame_id),(90+580, 20),0, 5e-3 * 200, (0,255,0),2)
                        for track_id,tlwh in track[str(frame_id)].items():
                            tid= str(track_id)   
                            cv2.rectangle(frame, (int(tlwh[0]), int(tlwh[1])), (int(tlwh[0])+int(tlwh[2]), int(tlwh[1])+int(tlwh[3])) ,(255,255,255), 2)
                            cv2.rectangle(frame, (580, 20), (90+580, 115+20) ,(255,255,255), 2)
                            cv2.putText(frame, str(tid),(int(tlwh[0]), int(tlwh[1])),0, 5e-3 * 200, (0,255,0),2)
                        vid_writer.write(frame)


            
        vid_writer.release()
        print("video done at", save_path)
    #plutôt la surface d'intersection des rectangles plutôt que la distance eucledienne 



"""video_name="MOT17-04-FRCNN"
gt_file = "/home/sophie/uncertain-identity-aware-tracking/Bytetrack/YOLOX_outputs/yolox_x_ablation/track_results/"+video_name+"_tracking_result.json"
#gt_file ="/home/sophie/uncertain-identity-aware-tracking/Bytetrack/YOLOX_outputs/yolox_x_ablation/videos/MOT17-04-FRCNN/gt_video_tracking_result.json"
with open(gt_file, 'r') as json_file:
        label_track = json.load(json_file)
#put_results_on_video ( label_track , save_path="/home/sophie/uncertain-identity-aware-tracking/Bytetrack/YOLOX_outputs/yolox_x_ablation/videos/"+video_name+"_bytetrack.mp4" , image_path="/home/sophie/uncertain-identity-aware-tracking/Bytetrack/datasets/mot/train/"+video_name+"/img1")
"""
'''save_path ="/home/sophie/uncertain-identity-aware-tracking/Bytetrack/YOLOX_outputs/yolox_x_ablation/videos/MOT17-04-FRCNN/gt_video_tracking.mp4"
video_path="/home/sophie/uncertain-identity-aware-tracking/Bytetrack/YOLOX_outputs/yolox_x_ablation/videos/MOT17-04-FRCNN/gt_video.mp4"
put_results_on_video ( label_track , save_path=save_path, video_path=video_path )
'''