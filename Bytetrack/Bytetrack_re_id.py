import pandas as pd 
import numpy as np
import json
import cv2 
import copy 
track_with_observation_file= "Bytetrack/videos/GR77_20200512_111314DBN_result_with_observations_feeder.json"
re_id_track_result_file = "Bytetrack/videos/GR77_20200512_111314DBN_re_id.json"

re_id_video_file = "Bytetrack/videos/GR77_20200512_111314DBN_re_id.mp4"
video_path = "Bytetrack/videos/GR77_20200512_111314.mp4"


save_video=False
def iou (boxA,boxB):
    boxA=[boxA[0],boxA[1],boxA[0]+boxA[2],boxA[1]+boxA[3]]
    boxB=[boxB[0],boxA[1],boxB[0]+boxB[2],boxB[1]+boxB[3]]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1))
    boxBArea = abs((boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1))
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou  #np.linalg.norm(np.array([float(track[0]), float(track[1])+float(track[3])/2])-np.array([600,17.5]))
min_iou_matching_gt_detections=0.7
def put_results_on_video(track, video_path, save_path):
    #print(video_path, save_path)
    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height)))     
    # Center coordinates
    center_coordinates = (625, 70)
    # Radius of circle
    radius = 2
    # Blue color in BGR
    color = (255, 0, 0)
    # Line thickness of 2 px
    thickness = 2
    
    
    """with open(track_with_observation_file, 'r') as json_file:
        data = json.load(json_file)"""
        
    ret_val, frame = cap.read()
    frame_id=1
    
    while ret_val : 
        ret_val, frame = cap.read()
        #add feeder and drinker center 
        frame = cv2.circle(frame, center_coordinates, radius, color, thickness)
        frame = cv2.circle(frame, (90,102), radius, color, thickness)
        #addd
        if str(frame_id) in track.keys() :
            if (frame_id!="0") :
                cv2.putText(frame, str(frame_id),(90+580, 20),0, 5e-3 * 200, (0,255,0),2)
                for track_id,tlwh in track[str(frame_id)].items():
                    tid= str(track_id)   
                    cv2.rectangle(frame, (int(tlwh[0]), int(tlwh[1])), (int(tlwh[0])+int(tlwh[2]), int(tlwh[1])+int(tlwh[3])) ,(255,255,255), 2)
                    #cv2.rectangle(frame, (580, 20), (90+580, 115+20) ,(255,255,255), 2)
                    cv2.putText(frame, str(tid),(int(tlwh[0]), int(tlwh[1])),0, 5e-3 * 200, (0,255,0),2)

                vid_writer.write(frame)
        #print("\n", "\n")
        frame_id=frame_id+1
        #print("a frame done")
    vid_writer.release()
    print("video done")
    #plutôt la surface d'intersection des rectangles plutôt que la distance eucledienne 


def produce_re_id_results(track_with_observation_file, tracking_file, re_id_track_result_file, save_video=False, video_path="", ):
    with open(track_with_observation_file, 'r') as json_file:
        data = json.load(json_file)
    with open(tracking_file, 'r') as json_file:
        data_tracking = json.load(json_file)
        
    
        
        
    tracking_result={}
    observation_infos =pd.DataFrame(columns=["frame_id", "observed"])
    matching={}
    matching_bboxes={}
    
    #pour chaque ligne avec une observation garder le temps, atq, le track_id avec le max de chance d'être l'atq
    for frame_id, frame_infos in data.items():
        if frame_id!="0": #we remove 0 because it doesn't have track_id
            dct={}
            if int(frame_id)==98:
                print("check")
            if "observation" in frame_infos.keys():
                #print("***",frame_infos["observation"].keys())
                check_frame_info= copy.deepcopy(frame_infos)
                for atq in frame_infos["observation"].keys():
                    #print(atq , frame_infos["observation"].keys())
                    corrected = {}
                    check_1 = np.array(frame_infos["observation"][atq])
                    #print(frame_id, atq)
                    #frame_infos["observation"][atq] = np.array( frame_infos["observation"][atq].strip('[]').split() , dtype=float)
                    # Convert the list of strings to a numpy array of floats
                    #array = np.array(string_array, dtype=float)
                    max_track_id =np.argmax(np.array(frame_infos["observation"][atq]))
                    
                    #we can't certify current and tracking real current will match so we check
                    #print(check_1, max(frame_infos["observation"][atq]))
                    crs_max_track_id=frame_infos["current"][max_track_id]['track_id'] #the track_id corresponding to the object with the highest atq match probability
                    max_matching=0
                    '''for tracker_track_id, tracker_track in data_tracking[frame_id].items():
                        if iou(tracker_track , data[str(int(frame_id+1))]["previous"][max_track_id]['location'])>max_matching:
                            crs_max_track_id = tracker_track_id
                            max_matching = iou(tracker_track , data[str(int(frame_id+1))]["previous"][max_track_id]['location'])
                    if max_matching<min_iou_matching_gt_detections: # list(tracker_track)== list(frame_infos["current"][max_track_id]['location']):
                        crs_max_track_id = None'''
                        
                    
                    #We have to make sure that his hasn't been already taken by another atq of the same frame
                    ####much more than that it should be an hungarian
                    track_id =  crs_max_track_id#frame_infos["current"][max_track_id]['track_id']
                    
                    """if atq=="13" :
                        print("check 6,24 for   atq 13 before frame 7")"""
                    """if len(frame_infos["observation"][atq])!=len ((data_tracking[frame_id].keys()) ) :
                        print(atq, frame_id,"size error in re_id",len(frame_infos["observation"][atq]),len ((frame_infos["current"]) ))
                    """
                    observation_infos = pd.concat([observation_infos, pd.DataFrame([{"frame_id":frame_id, "atq":atq, "track_id":track_id }])], ignore_index=True)
                    if track_id is not None: 
                        if atq in matching.keys() : 
                            if matching[atq]!=track_id:
                                #print("there were an id switch at frame ", frame_id," between", matching[atq] , "and", track_id, "with atq", atq)
                                corrected[track_id] = matching[atq]
                                corrected[matching[atq]] = track_id
                                
                                tmp_frame_info=copy.deepcopy(data_tracking[frame_id])
                            
                                for idx_track, track in tmp_frame_info.items():#enumerate(frame_infos["current"]):
                                    tlwh =track#["location"]
                                    track_id = int(idx_track)#track["track_id"]
                                    
                                    if track_id in  corrected.keys(): 
                                        if str(track_id) in data_tracking[frame_id].keys() :
                                            if str(corrected[track_id]) in data_tracking[frame_id].keys():
                                                tmp_track_location=copy.deepcopy(tmp_frame_info[str(corrected[track_id])])
                                                data_tracking[frame_id][str(corrected[track_id])] = copy.deepcopy( data_tracking[frame_id][str(track_id)])
                                                data_tracking[frame_id][str(track_id)] = tmp_track_location
                                                break
                                                
                                            
                                            else:
                                                corrected_track_id= corrected[track_id]
                                                print("in frame_id ",frame_id, "the atq",atq, "help corrected",track_id, "and", corrected_track_id)
                                                #frame_infos["current"][idx_track]["track_id"]=   corrected_track_id
                                                #new_frame_info[corrected_track_id] =track
                                                data_tracking[frame_id][str(corrected[track_id])] = data_tracking[frame_id].pop(str(track_id)) 
                                                print("check")
                                    
                                for frame_id_2 in data_tracking.keys():
                                    #... les frames suivantes ###  repeter 
                                    if set([str(tr_id) for tr_id in corrected.keys()]) & (set(data_tracking[frame_id_2].keys())) :

                                        tmp_frame_info= copy.deepcopy(data_tracking[frame_id_2])
                                        if int(frame_id_2)> int(frame_id):
                                            for idx_track, track in tmp_frame_info.items():
                                                track_id = int(idx_track)
                                                if track_id in  corrected.keys(): 
                                                    #print(tmp_frame_info)
                                                    
                                                    if str(corrected[track_id]) in data_tracking[frame_id_2].keys():
                                                        tmp_track_location=copy.deepcopy(tmp_frame_info[str(corrected[track_id])])
                                                        data_tracking[frame_id_2][str(corrected[track_id])] = copy.deepcopy( data_tracking[frame_id_2][str(track_id)])
                                                        data_tracking[frame_id_2][str(track_id)] = tmp_track_location
                                                    else:    
                                                        data_tracking[frame_id_2][str(corrected[track_id])] = data_tracking[frame_id_2].pop(str(track_id))
                                                    #print(frame_infos_2["current"][idx_track]["track_id"], corrected[track["track_id"]])
                                            
                
                        #if track_id is not None:
                        else:
                            matching[atq] = int(track_id)    
                        #matching_bboxes[atq]= frame_infos["current"][max_track_id]['location']
                    #we correct track_id by the observation
                    else:
                        print("*******track_id is None for max_track_id", frame_infos["current"][max_track_id]['track_id'])
                """if check_frame_info != frame_infos:
                    print("frame_info",check_frame_info == frame_infos)"""
            for track_id, track in data_tracking[frame_id].items():
                tlwh =track#["location"]
                #track_id = track["track_id"]
                if str(track_id)!="null" and track_id is not None:
                    dct[track_id]=(int(tlwh[0]), int(tlwh[1]), int(tlwh[2]), int(tlwh[3]) )       
            tracking_result[frame_id]=dct

    with open(re_id_track_result_file, 'w') as outfile:
            json.dump(tracking_result, outfile)
    print("re-id is done")
    observation_infos.to_csv(re_id_track_result_file.split(".json")[0]+"_observations.csv")
    
    if save_video:
        put_results_on_video(tracking_result, video_path, save_path=re_id_track_result_file.split(".json")[0]+".mp4")
        
    return tracking_result
    

import pandas as pd 
import numpy as np 
import json 
import plotly.express as px
import plotly.graph_objects as go

label_file= "Bytetrack/videos/labels_with_atq.json"
track_result = "Bytetrack/videos/GR77_20200512_111314tracking_result.json"
def read_data(file):
    #here we will go through detections of deepsort 
    import json
    track={}
    with open(file) as f:
        json_file = json.load(f) 

    for frame, detections in json_file.items():
        frame=int(frame)
        track[frame]={}
        for id, detection in detections.items():
            track[frame][id]={}
            track[frame][id]= tuple(detection)

    return track

def iou (boxA,boxB):
    boxA=[boxA[0],boxA[1],boxA[0]+boxA[2],boxA[1]+boxA[3]]
    boxB=[boxB[0],boxA[1],boxB[0]+boxB[2],boxB[1]+boxB[3]]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1))
    boxBArea = abs((boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1))
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou  #np.linalg.norm(np.array([float(track[0]), float(track[1])+float(track[3])/2])-np.array([600,17.5]))

"""label_file= "Bytetrack/videos/labels_without_atq.json"
label = read_data(label_file)
#put_results_on_video ( label , save_path="label_video.mp4" , video_path=video_path, track_with_observation_file=track_with_observation_file)
"""

#tracking_result =produce_re_id_results(track_with_observation_file, re_id_track_result_file)
"""put_results_on_video ( tracking_result , save_path="Bytetrack/videos/visualize/re_id_feeder_video.mp4" , video_path=video_path)"""