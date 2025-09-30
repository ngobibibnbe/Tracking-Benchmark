import pandas as pd 
import numpy as np

import numpy as np  
import pandas as pd

import numpy as np
from scipy.optimize import linear_sum_assignment
#! pip install mpmath
from mpmath import *
import pandas as pd
import numpy as np 
import datetime as dt
import json 
import copy
import motmetrics as mm
import re
import pandas as pd 
from ATQ import adding_atq
from forwardBackward import process_forwad_backward
import os 
import time 
from Bytetrack_re_id import produce_re_id_results 
from Bytetrack_re_id import put_results_on_video

import datetime
import os
import shutil

####get all dataset names 
def get_all_folders(path):
    folders = [f.name for f in os.scandir(path) if f.is_dir() and 'MOT' in f.name]
    return folders

def convert_mot_format_to_json_format(gt_path=None,destination_path=None):
    """
        read the MOT17 format and store in our dictionnary tracking format at the destination provided
    """
    with open(gt_path, 'r') as file:
        # Read each line in the file
        tracking={}
        for line in file:
            # Remove any trailing whitespace/newline characters and split by comma
            #print(line.strip())
            # ['frame', 'id', 'bbox_left', 'bbox_top', 'bbox_width', 'bbox_height', 'confidence', 'class', 'visibility']
            frame_id, id, x,y,w,h,conf,class_,_ = line.strip().split(',')
            """print(line.strip().split(','))
            print(frame_id,id,x,y,w,h)"""
            # Print the extracted elements for each line
            if int(class_)==1:
                cr_frame_id = str(int(frame_id)-1) #because halfs start at one
                if cr_frame_id not in tracking.keys():
                    tracking[cr_frame_id]={}
                tracking[cr_frame_id][id]=[int(x),int(y),int(w),int(h)]
    
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    with open(destination_path, "w") as file:
        json.dump(tracking,file)
    print('tracking saved at', destination_path)
    return tracking

base_path ="/data/home/sophie/sophie_2024-10-08/uncertain-identity-aware-tracking/Bytetrack/YOLOX_outputs/yolox_x_mix_mot20_ch/track_results"
annotation_path="/data/home/sophie/sophie_2024-10-08/uncertain-identity-aware-tracking/Bytetrack/tools/datasets/MOT20/train"
path = "/data/home/sophie/sophie_2024-10-08/uncertain-identity-aware-tracking/Bytetrack/tools/datasets/MOT20/train"
folders = get_all_folders(path)
print(folders)







def evaluate (tracks,track_result, limit=float("inf"),column_indice= "model") :
    acc = mm.MOTAccumulator(auto_id=False)
    for key, track in tracks.items():
        if key in  track_result.keys()  :
            frame_detection=track_result[key]
            real_animals=np.array([individual["rectangle"] for individual  in list(track.values())])
            hypothesis_animal= np.array([individual["rectangle"] for individual  in list(frame_detection.values())])
            #print(real_animals, "*****", hypothesis_animal)
            #print(help(mm.distances.iou_matrix))
            distances = mm.distances.iou_matrix(real_animals, hypothesis_animal, max_iou=0.2)
            #print("deepsort",hypothesis_animal)
            #print("real",real_animals)
            #print(distances)
            #print(real_animals)
            """try:
                acc.update([float(key1) for key1 in list(track.keys())],[ float(re.search(r'^\d+', key1)[0]) for key1 in frame_detection.keys()],distances, frameid=int(key))
            except:
                print(frame_detection.keys())
                for key1 in list(frame_detection.keys()):
                    print(key1, re.search(r'^\d+', key1))
                return 0,0,0"""
            acc.update([float(key1) for key1 in list(track.keys())],[ float(re.search(r'^\d+', key1)[0]) for key1 in frame_detection.keys()],distances, frameid=int(key))
            print("ok")
        if int(key)>= limit:
              break
    print(acc)
    mh = mm.metrics.create()
    summary = mh.compute_many(
        [acc],
        #[acc_deepsort, acc_deeplabcut],
        #metrics=mm.metrics.motchallenge_metrics,
        #metrics=['idf1','num_frames', 'mota', 'motp'],
        names=["model_name"], #['deepsort','deeplabcut'],
        generate_overall=True
        )

    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    #print("cumulative summary o")
    #print(strsummary)
    #print(summary.columns)

    idf1=summary["idf1"].loc["model_name"]
    """for col in summary.columns:
        summary.columns = summary.columns.str.replace(col, col+'_'+column_indice)"""

    return idf1




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
            track[frame][id]["rectangle"]= tuple(detection)

    return track


def read_data_str_frame(file):
    #here we will go through detections of deepsort 
    import json
    track={}
    with open(file) as f:
        json_file = json.load(f) 

    for frame, detections in json_file.items():
        frame=int(frame)
        track[str(frame)]={}
        for id, detection in detections.items():
            track[str(frame)][id]={}
            track[str(frame)][id]= tuple(detection)

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


def precise_accuracy_track(label_track, model_track, basic_tracker=False,gt_base_number =-2):#4800):
    """cette fonction calcule le f1-score recall, accuracy des model par rapport au background
    dedans, les score des trackers et des hMM based tracker sont calculés différemment car quand le hmm based tracker est seuillé, 
    il y'a des id de track qu'il ne retourne pas dans son fichier de resultat.

    Args:
        label_track (_type_): _description_
        model_track (_type_): _description_
        basic_tracker (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    idf1 = evaluate(label_track,model_track)

    try:
        idf1 = evaluate(label_track,model_track)
    except Exception as e:
        print("exception", e)
        idf1=0
        
    print("******IDF1",idf1)
    start=True 
    nbr_frame=0
    nbr_frame_acc=0
    acc =0
    rec=0
    """if basic_tracker==True:
        matching={}
        for frame_id in label_track.keys():
            for label_atq, label_box in label_track[frame_id].items() : 
                if label_atq not in matching.keys() and label_atq!="observed":
                    matching[label_atq]=None 
                    break"""
            
    
    def match_track_and_atq(label_track, model_track):
            
            matching={}
            label_track_only_concerned = {}
            for frame_id in label_track.keys():
                label_track_only_concerned[frame_id]= {}
                if frame_id in model_track.keys() :
                    for label_atq, label_box in label_track[frame_id].items() : 
                        if float(label_atq)>gt_base_number:
                            label_track_only_concerned[frame_id][label_atq] = label_track[frame_id][label_atq]
                            
                        if label_atq not in matching.keys():
                            max_iou=float('-inf')
                            for model_atq, model_box in model_track[frame_id].items(): 
                                #print(model_atq)
                                if  label_atq!="observed" and model_atq!="observed":#fix the problem with the obseved on the label 
                                    
                                    """if str(model_atq)==str(label_atq):
                                        print(model_atq, model_box["rectangle"], label_box["rectangle"])
                                    """
                                    
                                    tmp = iou(model_box["rectangle"], label_box["rectangle"])
                                    if tmp>=max_iou:
                                        matching [label_atq]=model_atq
                                        max_iou =tmp
                
                if len(matching.keys())== len(label_track[frame_id].keys()):
                    break
            return label_track_only_concerned , {value:key for key, value in matching.items() if float(key)>gt_base_number}
    
    
    if basic_tracker==True:
        label_tracki, matching = match_track_and_atq(label_track, model_track)
    else:
        min_frame =min([int(i) for i in list(label_track.keys())])
        matching= {k:k for k in label_track[0].keys() if float(k)>gt_base_number }# matching= {k:k for k in label_track[min_frame].keys() if  'identities' not in k and float(k)>gt_base_number}
    for frame_id in label_track.keys() :
        if frame_id in model_track.keys() :
            nbr_frame+=1
            matching_frame={}
            taken_atq=[]
            remaining_atq=[label_atq for label_atq in label_track[frame_id].keys()]
            for model_atq, model_box in model_track[frame_id].items() :
                    max_iou=0
                    atq_matching_model =None
                    for  label_atq, label_box in label_track[frame_id].items():
                        if  label_atq!="observed" and model_atq!="observed" :#fix the problem with the obseved on the label 
                            tmp = iou(model_box["rectangle"], label_box["rectangle"])
                            if tmp>max_iou and label_atq not in taken_atq:
                                max_iou =tmp
                                atq_matching_model = label_atq
                    if atq_matching_model is not None:
                        taken_atq.append(atq_matching_model)
                                
                    #if basic_tracker==True:
                    if model_atq in matching.keys(): #ca c'est pour les modèles qui crèent trop de nouvelles identités
                            matching_frame[matching[model_atq] ]=atq_matching_model 
                    else:
                        if basic_tracker:
                            if atq_matching_model not in matching.values():
                                #print(atq_matching_model,"new atq at frameid:", frame_id,"for track_id",model_atq, matching)
                                #the atq could be assign to a new model identity
                                matching[model_atq ]=atq_matching_model 
                                matching_frame[matching[model_atq] ]=atq_matching_model 
                            else:
                                #remettreprint(atq_matching_model, 'atq could not be assigne to the model id :',model_atq)
                                matching_frame[model_atq ]=None
                        else:
                            #matching[atq_matching_model ]=atq_matching_model
                            matching[model_atq ]=model_atq 
                            matching_frame[matching[model_atq] ]=atq_matching_model
            remaining_atq =list( set( remaining_atq) -  set(taken_atq))
            
                                    
            filtered ={key:value for key,value in matching_frame.items() if value==key }
            
            if len(matching_frame.keys())!=0:
                nbr_frame_acc+=1
                #print(frame_id, "(",len(label_track[frame_id]),")",len(matching_frame), (len([label_atq  for label_atq in matching_frame.values() if label_atq is not None]) + len(remaining_atq)), matching_frame)
                acc = acc + len(filtered.keys())/ len(matching_frame.keys())
                #print(len([label_atq  for label_atq in matching_frame.values() if label_atq is not None]), len(remaining_atq))
                if len([label_atq  for label_atq in matching_frame.values() if label_atq is not None])+ len(remaining_atq) ==0:
                    rec=rec+0
                else:
                    rec = rec+ len(filtered.keys())/ (len([label_atq  for label_atq in matching_frame.values() if label_atq is not None]) + len(remaining_atq))
                if len(filtered.keys())/ len(label_track[frame_id].keys())>1:
                    print("stop")
        else:
            print("weird thing", frame_id, "not in model_track")#, model_track.keys())

    
    acc = acc/nbr_frame_acc
    rec=rec/nbr_frame
    print(acc, rec) 

    if acc+rec==0:
        f1=0
    else:
        f1=2*acc*rec/(acc+rec)
    return acc  , rec , f1  , idf1 
               
    
    
    
    

import importlib
import forwardBackward
importlib.reload(forwardBackward)
from forwardBackward import process_forwad_backward

home_folder = "/data/home/sophie/sophie_2024-10-08/uncertain-identity-aware-tracking/Bytetrack/"
base_path_datasets =home_folder+"/YOLOX_outputs/yolox_x_mix_mot20_ch/track_results/"
####get all dataset names 
def get_all_folders(path):
    folders = [f.name for f in os.scandir(path) if f.is_dir()]
    return folders
path = base_path_datasets
folders = get_all_folders(path)
print(folders)

###########################
######Treat a dataset #####
###########################
#dataset_name = "MOT17-13-FRCNN"#folders[0]



from Bytetrack_re_id import produce_re_id_results 

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
            track[frame][id]["rectangle"]= tuple(detection)

    return track

def read_data_for_video_generation(file):
    #here we will go through detections of deepsort 
    import json
    track={}
    with open(file) as f:
        json_file = json.load(f) 

    for frame, detections in json_file.items():
        frame=str(frame)
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



def evaluate (tracks,track_result, limit=float("inf"),column_indice= "model") :
    acc = mm.MOTAccumulator(auto_id=False)
    for key in sorted(tracks.keys()):
        track=tracks[key]
        if key in  track_result.keys()  :
            frame_detection=track_result[key]
            real_animals=np.array([list(individual["rectangle"]) for individual  in list(track.values())])
            hypothesis_animal= np.array([list(individual["rectangle"]) for individual  in list(frame_detection.values())])
            #print(real_animals, "*****", hypothesis_animal)
            distances = mm.distances.iou_matrix(real_animals, hypothesis_animal, max_iou=0.75)
            #print("deepsort",hypothesis_animal)
            #print("real",real_animals)
            #print(distances)
            #print(real_animals)
            """try:
                print("****",acc,)
                acc.update([float(key1) for key1 in list(track.keys())],[ 100000+float(re.search(r'^\d+', key1)[0]) for key1 in frame_detection.keys()],distances, frameid=int(key))
            except Exception as e :
                print("***",e)
                for key1 in list(frame_detection.keys()):
                    print(key1)# re.search(r'^\d+', key1))
                return 0"""
            acc.update([float(key1) for key1 in list(track.keys())],[ float(re.search(r'^\d+', key1)[0]) for key1 in frame_detection.keys()],distances, frameid=int(key))
        if int(key)>= limit:
              break
    

    mh = mm.metrics.create()
    summary = mh.compute_many(
        [acc],
        #[acc_deepsort, acc_deeplabcut],
        #metrics=mm.metrics.motchallenge_metrics,
        #metrics=['idf1','num_frames', 'mota', 'motp'],
        names=["model_name"], #['deepsort','deeplabcut'],
        generate_overall=True
        )

    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    #print("cumulative summary o")
    #print(strsummary)
    #print(summary.columns)

    idf1=summary["idf1"].loc["model_name"]
    """for col in summary.columns:
        summary.columns = summary.columns.str.replace(col, col+'_'+column_indice)"""

    return idf1


def precise_accuracy_track(label_track, model_track, basic_tracker=False,gt_base_number =-2):#4800):
    """cette fonction calcule le f1-score recall, accuracy des model par rapport au background
    dedans, les score des trackers et des hMM based tracker sont calculés différemment car quand le hmm based tracker est seuillé, 
    il y'a des id de track qu'il ne retourne pas dans son fichier de resultat.

    Args:
        label_track (_type_): _description_
        model_track (_type_): _description_
        basic_tracker (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    
    idf1 = evaluate(label_track,model_track)
    
    start=True 
    nbr_frame=0
    nbr_frame_acc=0
    acc =0
    rec=0
    """if basic_tracker==True:
        matching={}
        for frame_id in label_track.keys():
            for label_atq, label_box in label_track[frame_id].items() : 
                if label_atq not in matching.keys() and label_atq!="observed":
                    matching[label_atq]=None 
                    break"""
            
    
    def match_track_and_atq(label_track, model_track):
            
            matching={}
            label_track_only_concerned = {}
            for frame_id in label_track.keys():
                label_track_only_concerned[frame_id]= {}
                if frame_id in model_track.keys() :
                    for label_atq, label_box in label_track[frame_id].items() : 
                        if float(label_atq)>gt_base_number:
                            label_track_only_concerned[frame_id][label_atq] = label_track[frame_id][label_atq]
                            
                        if label_atq not in matching.keys():
                            max_iou=float('-inf')
                            for model_atq, model_box in model_track[frame_id].items(): 
                                #print(model_atq)
                                if  label_atq!="observed" and model_atq!="observed":#fix the problem with the obseved on the label 
                                    
                                    """if str(model_atq)==str(label_atq):
                                        print(model_atq, model_box["rectangle"], label_box["rectangle"])
                                    """
                                    
                                    tmp = iou(model_box["rectangle"], label_box["rectangle"])
                                    if tmp>=max_iou:
                                        matching [label_atq]=model_atq
                                        max_iou =tmp
                
                if len(matching.keys())== len(label_track[frame_id].keys()):
                    break
            return label_track_only_concerned , {value:key for key, value in matching.items() if float(key)>gt_base_number}
    
    
    if basic_tracker==True:
        label_tracki, matching = match_track_and_atq(label_track, model_track)
    else:
        min_frame =min([int(i) for i in list(label_track.keys())])
        matching= {k:k for k in label_track[min_frame].keys() if  'identities' not in k and float(k)>gt_base_number}
    for frame_id in label_track.keys() :
        if frame_id in model_track.keys() :
            nbr_frame+=1
            matching_frame={}
            taken_atq=[]
            remaining_atq=[label_atq for label_atq in label_track[frame_id].keys()]
            for model_atq, model_box in model_track[frame_id].items() :
                    max_iou=0
                    atq_matching_model =None
                    for  label_atq, label_box in label_track[frame_id].items():
                        if  label_atq!="observed" and model_atq!="observed" :#fix the problem with the obseved on the label 
                            tmp = iou(model_box["rectangle"], label_box["rectangle"])
                            if tmp>max_iou and label_atq not in taken_atq:
                                max_iou =tmp
                                atq_matching_model = label_atq
                    if atq_matching_model is not None:
                        taken_atq.append(atq_matching_model)
                                
                    #if basic_tracker==True:
                    if model_atq in matching.keys(): #ca c'est pour les modèles qui crèent trop de nouvelles identités
                            matching_frame[matching[model_atq] ]=atq_matching_model 
                    else:
                        if basic_tracker:
                            if atq_matching_model not in matching.values():
                                #print(atq_matching_model,"new atq at frameid:", frame_id,"for track_id",model_atq, matching)
                                #the atq could be assign to a new model identity
                                matching[model_atq ]=atq_matching_model 
                                matching_frame[matching[model_atq] ]=atq_matching_model 
                            else:
                                #remettreprint(atq_matching_model, 'atq could not be assigne to the model id :',model_atq)
                                matching_frame[model_atq ]=None
                        else:
                            #matching[atq_matching_model ]=atq_matching_model
                            matching[model_atq ]=model_atq 
                            matching_frame[matching[model_atq] ]=atq_matching_model
            remaining_atq =list( set( remaining_atq) -  set(taken_atq))
            
                                    
            filtered ={key:value for key,value in matching_frame.items() if value==key }
            
            if len(matching_frame.keys())!=0:
                nbr_frame_acc+=1
                #print(frame_id, "(",len(label_track[frame_id]),")",len(matching_frame), (len([label_atq  for label_atq in matching_frame.values() if label_atq is not None]) + len(remaining_atq)), matching_frame)
                acc = acc + len(filtered.keys())/ len(matching_frame.keys())
                rec = rec+ len(filtered.keys())/ (len([label_atq  for label_atq in matching_frame.values() if label_atq is not None]) + len(remaining_atq))
                if len(filtered.keys())/ len(label_track[frame_id].keys())>1:
                    print(len(filtered.keys()), len(label_track[frame_id].keys()))
                    print("stop")
        else:
            print("weird thing", frame_id, "not in model_track")#, model_track.keys())

    
    acc = acc/nbr_frame_acc
    rec=rec/nbr_frame
    print(acc, rec) 

    f1=2*acc*rec/(acc+rec)
    return acc  , rec , f1  , idf1 
               

def put_val_half_tracking(file_path, dataset_name=""): 
    base_path ="/data/home/sophie/sophie_2024-10-08/uncertain-identity-aware-tracking/Bytetrack/tools/datasets/MOT20/train/"
    dataset_path = os.path.join(base_path,dataset_name)
    
    gt_half_path=os.path.join(dataset_path,"gt/gt_val_half.txt")

    tracking_half =convert_mot_format_to_json_format(gt_half_path,'trash/trash.json')
    tracking_complete =convert_mot_format_to_json_format(os.path.join(dataset_path,"gt/gt.txt"),'trash/trash.json')

    start_half = len(tracking_complete.keys()) -  len(tracking_half.keys()) -1
    #######################################
    ######Store the empty half gt video####
    #######################################
    #length_half =len (tracking.keys()) #length of the half
    new_track={}
    with open(file_path, "r") as file:
        track = json.load(file)


    for new_key in range (len(tracking_half.keys()) ):
        new_track[str(new_key)] = track[str(new_key+start_half)]
    
    
    file_path=file_path.split(dataset_name)[0] + '/'+dataset_name+ '/'+dataset_name+".json"
        
    with open(file_path.split(".json")[0]+"_DBN_result_half_val.json", "w") as file:
        json.dump(new_track, file)
    
    
    '''file_path = '/home/sophie/uncertain-identity-aware-tracking/Bytetrack/YOLOX_outputs/fairmot_json_files_hmm_format/'+dataset_name.split("FRCNN")[0]+"SDP.json"
    new_track={}
    with open(file_path, "r") as file:
        track = json.load(file)
        
    for new_key in range (len(tracking_half.keys()) ):
        new_track[str(new_key)] = track[str(new_key+start_half)]
        #print(len(new_track[str(new_key)] ["previous"]), len(new_track[str(new_key)] ["current"]), np.array(new_track[str(new_key)] ["matrice"]).shape)
    with open(file_path.split(".json")[0]+"_half_val.json", "w") as file:
        json.dump(new_track, file)'''
    
    
           
        
def generate_tracking_result_from_observation(dbn_file, tracking_result_file):
    with open(dbn_file,"r") as file:
        data=json.load(file)
    result_track={}

    for frame_id,content in  data.items():
        result_track[str(frame_id)]={}
        for track in content["current"]:
            track_id= track["track_id"]
            if track_id is not None:
                result_track[str(frame_id)][track_id]=track["location"]
                
    with open(tracking_result_file, "w") as file:
        json.dump(result_track, file)
    print("tracking saved at", tracking_result_file)
home_folder = "/data/home/sophie/sophie_2024-10-08/uncertain-identity-aware-tracking/Bytetrack/"


min_observation_max_value=0
is_it_random =False
exp="_fairmot_high_no_random"


experiences=([ [0, True,"_fairmot_low_random" ]])  #[0, False,"_fairmot_low_no_random" ]])#, [0, True,"_fairmot_low_random" ], [0.5, False,"_fairmot_high_no_random" ] )
folders=["MOT20-03"]


for experience in experiences :
    print(experience)
    
    min_observation_max_value= experience[0]#0
    is_it_random = experience[1]#True
    exp=experience[2] #"_bytetrack_low_random"
    #dataset_interest="MOT20-03" #starting at 5 high, low_no_, lowrnandom
        
    folders=["MOT20-03"]
    #dataset_interest="MOT20-03"  #at 0
    """
    conda activate evaluation4
    cd Bytetrack
    python mot20-evaluation-generation_Fairmot.py
    """

    """exps[0]="_fairmot_low_no_random"
    exps[1]="_bytetrack_low_no_random"

    exps[2]="_fairmot_high_no_random"
    exps[3]="_bytetrack_high_no_random"


    exps[4]="_fairmot_low_random"
    exps[5]="_bytetrack_low_random"""





    dct_tmp_match={}


    def copy_file(src_path, dest_path):
        """
        Copies a file from src_path to dest_path, creating all necessary directories.
        
        :param src_path: Path to the source file
        :param dest_path: Path to the destination file
        """
        # Ensure destination directory exists
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        # Copy file
        shutil.copy2(src_path, dest_path)
        print(f"File copied to {dest_path}")
        
    tmp_folder = "/data/home/sophie/tmp_sophie/"+exp+"/"+str(datetime.datetime.now())
    for dataset_name in folders:
        
        if 'video' not in dataset_name:
                base_path= home_folder+"YOLOX_outputs/yolox_x_mix_mot20_ch/Fairmot/"+dataset_name+"/"+dataset_name
                dataset_path = os.path.join(base_path_datasets,dataset_name)
                
                
                
                gt_video=home_folder+"YOLOX_outputs/yolox_x_mix_mot20_ch/track_results/videos/"+dataset_name+"/gt_video.mp4"  
                #dbn_file= "/home/sophie/uncertain-identity-aware-tracking/Bytetrack/YOLOX_outputs/yolox_x_ablation/track_results/"+dataset_name+"/fairmot/"+dataset_name.split("-")[0]+"-"+dataset_name.split("-")[1]+"-SDP.json"
                dbn_file=home_folder+"YOLOX_outputs/yolox_x_mix_mot20_ch/Fairmot/"+dataset_name+"/"+dataset_name+".json"
                #"/home/sophie/uncertain-identity-aware-tracking/Bytetrack/YOLOX_outputs/yolox_x_ablation/track_results/MOT17-04-FRCNN/fairmot/fairmot_MOT17_04.json"
                ###use once ################### put_val_half_tracking(dbn_file, dataset_name)
                dbn_file= dbn_file.split(".json")[0]+"_DBN_result_half_val.json"
                track_file=base_path+"_tracking_result.json"
                gt_path=os.path.join("/data/home/sophie/sophie_2024-10-08/uncertain-identity-aware-tracking/Bytetrack/YOLOX_outputs/yolox_x_mix_mot20_ch/track_results/",dataset_name+"/"+dataset_name+"_gt.json")

                
                gt_path_dest=os.path.join(tmp_folder,gt_path[1:])
                print(gt_path,tmp_folder)
                copy_file(gt_path,gt_path_dest)
                copy_file(gt_video,os.path.join(tmp_folder,gt_video[1:]))
                copy_file(track_file,os.path.join(tmp_folder,track_file[1:]))
                copy_file(dbn_file,os.path.join(tmp_folder,dbn_file[1:]))

                dct_tmp_match[gt_path]=gt_path_dest
                dct_tmp_match[gt_video]=os.path.join(tmp_folder,gt_video[1:])
                dct_tmp_match[track_file] = os.path.join(tmp_folder,track_file[1:])
                dct_tmp_match[dbn_file]=os.path.join(tmp_folder,dbn_file[1:])


    """
    for dataset_name in folders:
        if dataset_name not in  dataset_interest:
            continue
        print(dataset_name)
        
        if 'videos' in dataset_name:
            continue
        print("in")
        base_path= home_folder+"YOLOX_outputs/yolox_x_mix_mot20_ch/Fairmot/"+dataset_name+"/"+dataset_name
        os.makedirs(base_path, exist_ok=True)
        

        gt_video=home_folder+"YOLOX_outputs/yolox_x_mix_mot20_ch/track_results/videos/"+dataset_name+"/gt_video.mp4"  
        #dbn_file= "/home/sophie/uncertain-identity-aware-tracking/Bytetrack/YOLOX_outputs/yolox_x_ablation/track_results/"+dataset_name+"/fairmot/"+dataset_name.split("-")[0]+"-"+dataset_name.split("-")[1]+"-SDP.json"
        dbn_file=home_folder+"YOLOX_outputs/yolox_x_mix_mot20_ch/Fairmot/"+dataset_name+"/"+dataset_name+".json"
        #"/home/sophie/uncertain-identity-aware-tracking/Bytetrack/YOLOX_outputs/yolox_x_ablation/track_results/MOT17-04-FRCNN/fairmot/fairmot_MOT17_04.json"
        ###use once ################### put_val_half_tracking(dbn_file, dataset_name)
        dbn_file= dbn_file.split(".json")[0]+"_DBN_result_half_val.json"
        track_file=base_path+"_tracking_result.json"
        gt_path=os.path.join("/data/home/sophie/sophie_2024-10-08/uncertain-identity-aware-tracking/Bytetrack/YOLOX_outputs/yolox_x_mix_mot20_ch/track_results/",dataset_name+"/"+dataset_name+"_gt.json")

        #generate_tracking_result_from_observation(dbn_file, tracking_result_file)
        #put_results_on_video (read_data_for_video_generation(tracking_result_file), save_path=gt_video.split("gt_video")[0]+"fairmot_tracking.mp4",video_path=gt_video )
        #length_half = len(read_data_for_video_generation(tracking_result_file).keys())
        
        gt_video=dct_tmp_match[gt_video]
        
        gt_path=dct_tmp_match[gt_path]
        track_file=dct_tmp_match[ track_file]
        dbn_file= dct_tmp_match[dbn_file]
        
        tracking_result_file=track_file
        video_path=gt_video # '/home/sophie/uncertain-identity-aware-tracking/Bytetrack/YOLOX_outputs/yolox_x_ablation/videos/MOT17-04-FRCNN.mp4'  

        
        def score_for_various_artificial_observations_mot():

            hmm_result_with_visits=pd.DataFrame(columns=["nbr of visits", "accuracy", "recall", "f1"])
            re_id_result_with_visits=pd.DataFrame(columns=["nbr of visits", "accuracy", "recall", "f1"])
            tracker_result_score= pd.DataFrame(columns=["nbr of visits", "accuracy", "recall", "f1"])
            base_path_2 = base_path_datasets+dataset_name+'/'+dataset_name#gt_video'

            
            if os.path.exists(base_path_2+'accuracy_over_nbr_of_visits_with_track_helping_nr'+exp+'.csv'):
                hmm_result_with_visits=pd.read_csv(base_path_2+'accuracy_over_nbr_of_visits_with_track_helping_nr'+exp+'.csv')
            if os.path.exists(base_path_2+'accuracy_Re_id_over_nbr_of_visits_with_track_helping_nr'+exp+'.csv'):
                re_id_result_with_visits=pd.read_csv(base_path_2+'accuracy_Re_id_over_nbr_of_visits_with_track_helping_nr'+exp+'.csv')
            
            

            #add error bar here 
            for j in range(0,4,1):

                for i in range (2, 4000 , 200):# range (2, 200 , 10): #[10, 100]:#  [18]: # len(label_track.keys())            
                    #home_folder=home_folder#''#/home/sophie/uncertain-identity-aware-tracking/Bytetrack/'
                    observation_file=os.path.join(tmp_folder,exp+str(i)+"_"+str(j)+"_DBN_result_with_observations_visits_per_id.json")
                    Hmm_result_file=os.path.join(tmp_folder,exp+str(i)+"_"+str(j)+"_with_atq_tracking_with_HMM_result_per_id.json")
                    re_id_track_result_file = os.path.join(tmp_folder,exp+str(i)+"_"+str(j)+"_per_id_re_id.json")

                    video_path=gt_video
                    # '/home/sophie/uncertain-identity-aware-tracking/Bytetrack/YOLOX_outputs/yolox_x_ablation/videos/MOT17-04-FRCNN.mp4'  

                    #convert_mot_format_to_json_format(gt_path='/home/sophie/uncertain-identity-aware-tracking/Bytetrack/datasets/mot/train/MOT17-04-FRCNN/gt/gt_val_half.txt',
                    #                          destination_path=gt_path )
                    adding_atq(nbr_visit=i, output_file=observation_file, feeder=False, 
                                track_file=tracking_result_file,#****replace with tracking of fairmot
                                dbn_file= dbn_file,
                                labels_file=gt_path,
                            is_it_random = is_it_random, model=False,curated_artificial_visit=None, fairmot=True)
                    
                    #Hmm_result_file = "/home/sophie/uncertain-identity-aware-tracking/Bytetrack/YOLOX_outputs/yolox_x_ablation/track_results/MOT17-02-FRCNN_tracking_result.json"
                    label_track = read_data(gt_path)

                    try:
                            print("start forward backward")
                            process_forwad_backward(observation_file,nbr_visit="per_id_2_", pigs_HMM=False, json_save_path=Hmm_result_file, video_path=video_path)
                            print("end forward backward")
                            hmm_track = read_data(Hmm_result_file)
                            #print("end forward backward2", hmm_track.keys())
                            acc, rec, f1, idf1= precise_accuracy_track(label_track, hmm_track, gt_base_number=0, basic_tracker=True)
                    except Exception as e:
                            print("######",e)
                            acc, rec, f1, idf1=0,0,0,0
                    
                
                    new_row= {'nbr of visits':i, 'accuracy':acc, 'recall':rec, "f1":f1, "idf1":idf1}
                    print("****HMM")
                    print(new_row)
                    
                    
                    hmm_result_with_visits = pd.concat([hmm_result_with_visits, pd.DataFrame([new_row])], ignore_index=True)
                    #hmm_result_with_visits.to_csv(base_path+'accuracy_over_nbr_of_visits_with_track_helping.csv')

                            
                    

                    ####Re_id_part of the work
                    #re_id_tracking_result =produce_re_id_results(track_with_observation_file =observation_file , re_id_track_result_file = re_id_track_result_file )
                    re_id_tracking_result =produce_re_id_results(track_with_observation_file =observation_file , tracking_file=base_path+"_tracking_result.json", re_id_track_result_file = re_id_track_result_file , save_video=True, video_path=gt_video)

                    re_id_track = read_data(re_id_track_result_file)
                    print("***Oki before re_id_accuracy",re_id_tracking_result.keys())
                    acc, rec, f1,idf1= precise_accuracy_track(label_track, re_id_track,  gt_base_number=-2,basic_tracker=True)
                    new_row= {'nbr of visits':i, 'accuracy':acc, 'recall':rec, "f1":f1, "idf1":idf1}
                    print(new_row)
                    
                    re_id_result_with_visits = pd.concat([re_id_result_with_visits, pd.DataFrame([new_row])], ignore_index=True)
                    hmm_result_with_visits.to_csv(base_path_2+'accuracy_over_nbr_of_visits_with_track_helping_nr'+exp+'.csv')
                    re_id_result_with_visits.to_csv(base_path_2+'accuracy_Re_id_over_nbr_of_visits_with_track_helping_nr'+exp+'.csv')
                    #re_id_result_with_visits.to_csv(base_path+'accuracy_Re_id_over_nbr_of_visits_with_track_helping.csv')
                    put_results_on_video ( re_id_tracking_result , save_path= gt_video.split("gt_video")[0]+"re_id_video.mp4" , video_path=gt_video)

                    ########Tracker result
                    tracking_result = read_data(tracking_result_file)
                    print("***tracker***",tracking_result.keys())
                    acc, rec, f1,idf1= precise_accuracy_track(label_track, tracking_result,  gt_base_number=-2,basic_tracker=True)
                    new_row= {'nbr of visits':i, 'accuracy':acc, 'recall':rec, "f1":f1, "idf1":idf1}
                    print(new_row)
                    tracker_result_score = pd.concat([tracker_result_score, pd.DataFrame([new_row])], ignore_index=True)
                    
                    tracker_result_score.to_csv(base_path+'tracking_fairmot.csv')
                
            print("hmm",hmm_result_with_visits)        
            print("re_id",re_id_result_with_visits)
            print("tracker",tracker_result_score)        
            

        score_for_various_artificial_observations_mot()
        
    """


    import os
    import pandas as pd
    import multiprocessing

    def process_dataset(dataset_name):
        """ Function to process a single dataset in parallel """
        '''if dataset_name  in dataset_interest or 'videos' in dataset_name:
            return'''
        
        print(f"Processing: {dataset_name}")

        base_path = os.path.join(home_folder, "YOLOX_outputs/yolox_x_mix_mot20_ch/Fairmot", dataset_name, dataset_name)
        os.makedirs(base_path, exist_ok=True)

        gt_video = os.path.join(home_folder, "YOLOX_outputs/yolox_x_mix_mot20_ch/track_results/videos", dataset_name, "gt_video.mp4")
        dbn_file = os.path.join(home_folder, "YOLOX_outputs/yolox_x_mix_mot20_ch/Fairmot", dataset_name, f"{dataset_name}.json")
        dbn_file = dbn_file.replace(".json", "_DBN_result_half_val.json")

        track_file = base_path + "_tracking_result.json"
        gt_path = os.path.join("/data/home/sophie/sophie_2024-10-08/uncertain-identity-aware-tracking/Bytetrack/YOLOX_outputs/yolox_x_mix_mot20_ch/track_results", dataset_name, f"{dataset_name}_gt.json")

        # Resolve mapped paths
        gt_video = dct_tmp_match[gt_video]
        gt_path = dct_tmp_match[gt_path]
        track_file = dct_tmp_match[track_file]
        dbn_file = dct_tmp_match[dbn_file]
        
        tracking_result_file = track_file

        # Score function
        score_for_various_artificial_observations_mot(dataset_name, base_path, tracking_result_file, gt_video, dbn_file, gt_path)


    def score_for_various_artificial_observations_mot(dataset_name, base_path, tracking_result_file, gt_video, dbn_file, gt_path):
        """ Function to score artificial observations for a given dataset """

        base_path_2 = os.path.join(base_path_datasets, dataset_name, dataset_name)

        hmm_result_with_visits = pd.DataFrame(columns=["nbr of visits","nbr of kept visits", "accuracy", "recall", "f1", "idf1"])
        re_id_result_with_visits = pd.DataFrame(columns=["nbr of visits" ,"nbr of kept visits",  "accuracy", "recall", "f1", "idf1"])
        tracker_result_score = pd.DataFrame(columns=["nbr of visits", "accuracy", "recall", "f1", "idf1"])

        """if os.path.exists(base_path_2 + 'accuracy_over_nbr_of_visits_with_track_helping_nr' + exp + '.csv'):
            hmm_result_with_visits = pd.read_csv(base_path_2 + 'accuracy_over_nbr_of_visits_with_track_helping_nr' + exp + '.csv')
        if os.path.exists(base_path_2 + 'accuracy_Re_id_over_nbr_of_visits_with_track_helping_nr' + exp + '.csv'):
            re_id_result_with_visits = pd.read_csv(base_path_2 + 'accuracy_Re_id_over_nbr_of_visits_with_track_helping_nr' + exp + '.csv')"""
        
        for j in range(0,4 ,1):
            for i in range(1, 4002, 200):
                observation_file = os.path.join(tmp_folder, f"{exp}{i}_{j}_DBN_result_with_observations_visits_per_id.json")
                Hmm_result_file = os.path.join(tmp_folder, f"{exp}{i}_{j}_with_atq_tracking_with_HMM_result_per_id.json")
                re_id_track_result_file = os.path.join(tmp_folder, f"{exp}{i}_{j}_per_id_re_id.json")

                nbr_kept_visit = adding_atq(
                    nbr_visit=i, output_file=observation_file, feeder=False, 
                    track_file=tracking_result_file, dbn_file=dbn_file, labels_file=gt_path,
                    is_it_random=is_it_random, model=False, curated_artificial_visit=None, fairmot=True, min_observation_max_value=min_observation_max_value
                )

                print("#############", nbr_kept_visit ,i)
                label_track = read_data(gt_path)

                try:
                    print("Start forward backward")
                    process_forwad_backward(observation_file, nbr_visit="per_id_2_", pigs_HMM=False, json_save_path=Hmm_result_file, video_path=gt_video)
                    print("End forward backward")
                    hmm_track = read_data(Hmm_result_file)
                    acc, rec, f1, idf1 = precise_accuracy_track(label_track, hmm_track, gt_base_number=0, basic_tracker=True)
                except Exception as e:
                    print("######", e)
                    acc, rec, f1, idf1 = 0, 0, 0, 0

                new_row = {'nbr of visits': i, "nbr of kept visits":nbr_kept_visit, 'accuracy': acc, 'recall': rec, "f1": f1, "idf1": idf1}
                print("****HMM", new_row)
                hmm_result_with_visits = pd.concat([hmm_result_with_visits, pd.DataFrame([new_row])], ignore_index=True)

                # Re-ID processing
                re_id_tracking_result = produce_re_id_results(
                    track_with_observation_file=observation_file, tracking_file=tracking_result_file,
                    re_id_track_result_file=re_id_track_result_file, save_video=True, video_path=gt_video
                )

                re_id_track = read_data(re_id_track_result_file)
                acc, rec, f1, idf1 = precise_accuracy_track(label_track, re_id_track, gt_base_number=-2, basic_tracker=True)
                new_row = {'nbr of visits': i, "nbr of kept visits":nbr_kept_visit, 'accuracy': acc, 'recall': rec, "f1": f1, "idf1": idf1}
                print("Re-ID", new_row)
                re_id_result_with_visits = pd.concat([re_id_result_with_visits, pd.DataFrame([new_row])], ignore_index=True)

                # Tracker result
                tracking_result = read_data(tracking_result_file)
                acc, rec, f1, idf1 = precise_accuracy_track(label_track, tracking_result, gt_base_number=-2, basic_tracker=True)
                new_row = {'nbr of visits': i, "nbr of kept visits":nbr_kept_visit,  'accuracy': acc, 'recall': rec, "f1": f1, "idf1": idf1}
                print("Tracker", new_row)
                tracker_result_score = pd.concat([tracker_result_score, pd.DataFrame([new_row])], ignore_index=True)

                # Save results
                hmm_result_with_visits.to_csv(base_path_2 + f'accuracy_over_nbr_of_visits_with_track_helping_nr{exp}.csv')
                re_id_result_with_visits.to_csv(base_path_2 + f'accuracy_Re_id_over_nbr_of_visits_with_track_helping_nr{exp}.csv')
                tracker_result_score.to_csv(base_path + 'tracking_fairmot.csv')

                put_results_on_video(re_id_tracking_result, save_path=gt_video.replace("gt_video", "re_id_video"), video_path=gt_video)

        print("HMM", hmm_result_with_visits)
        print("Re-ID", re_id_result_with_visits)
        print("Tracker", tracker_result_score)


    if __name__ == "__main__":
        num_processes = min(multiprocessing.cpu_count()-1, len(folders))
        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.map(process_dataset, folders)
