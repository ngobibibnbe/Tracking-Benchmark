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
import importlib
import forwardBackward
importlib.reload(forwardBackward)
from forwardBackward import process_forwad_backward
import os
import pandas as pd
import multiprocessing

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

################tmux orders##########################
#low no random, low random, high no random de 0 à 10
#####################################################
experiences=([0, False,"_bytetrack_low_no_random" ], [0, True,"_bytetrack_low_random" ], [0.5, False,"_bytetrack_high_no_random" ] )
folders=["MOT20-03"]


for experience in experiences :
    print(experience)
    
    min_observation_max_value= experience[0]#0
    is_it_random = experience[1]#True
    exp=experience[2] #"_bytetrack_low_random"
    #dataset_interest="MOT20-03" #starting at 5 high, low_no_, lowrnandom

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
                base_path_2 = base_path_datasets+dataset_name+'/'+dataset_name #gt_video' #'
                dataset_path = os.path.join(base_path_datasets,dataset_name)
                
                
                observation_file=base_path_2+"_DBN_result_with_observations_visits_per_id.json"
                gt_path=os.path.join(home_folder+"YOLOX_outputs/yolox_x_mix_mot20_ch/track_results",dataset_name+"/"+dataset_name+"_gt.json")#"YOLOX_outputs/yolox_x_ablation/track_results/MOT17-04-FRCNN/MOT17-04-FRCNN_val_gt.json" 
                gt_video_path=home_folder+"YOLOX_outputs/yolox_x_mix_mot20_ch/track_results/videos/"+dataset_name+"/gt_video.mp4"
                track_file= base_path_2+"_tracking_result.json"
                dbn_file= base_path_2+"_DBN_result.json"
                
                gt_path_dest=os.path.join(tmp_folder,gt_path[1:])
                print(gt_path,tmp_folder)
                copy_file(gt_path,gt_path_dest)
                copy_file(gt_video_path,os.path.join(tmp_folder,gt_video_path[1:]))
                copy_file(track_file,os.path.join(tmp_folder,track_file[1:]))
                copy_file(dbn_file,os.path.join(tmp_folder,dbn_file[1:]))

                dct_tmp_match[gt_path]=gt_path_dest
                dct_tmp_match[gt_video_path]=os.path.join(tmp_folder,gt_video_path[1:])
                dct_tmp_match[track_file] = os.path.join(tmp_folder,track_file[1:])
                dct_tmp_match[dbn_file]=os.path.join(tmp_folder,dbn_file[1:])

    '''
    for dataset_name in folders:
        print(dataset_name)
        """if dataset_name=="MOT20-01":
            continue"""
        if 'video' not in dataset_name :#and dataset_name==  dataset_interest:
            print("##############we are in dataset", dataset_name)
            base_path_2 = base_path_datasets+dataset_name+'/'+dataset_name #gt_video' #'
            dataset_path = os.path.join(base_path_datasets,dataset_name)
            #base_path= home_folder+"YOLOX_outputs/yolox_x_ablation/track_results/"+dataset_name+"/Bytetrack/"+dataset_name
            #base_path= home_folder+"YOLOX_outputs/yolox_x_ablation/track_results/"+dataset_name+"/"+dataset_name
            
            gt_video_path=dct_tmp_match[home_folder+"YOLOX_outputs/yolox_x_mix_mot20_ch/track_results/videos/"+dataset_name+"/gt_video.mp4"]
            #base_path= home_folder+"YOLOX_outputs/yolox_x_mix_mot20_ch/videos/"+dataset_name+"/gt_video"
            gt_path=dct_tmp_match[os.path.join(home_folder+"YOLOX_outputs/yolox_x_mix_mot20_ch/track_results",dataset_name+"/"+dataset_name+"_gt.json")]#"YOLOX_outputs/yolox_x_ablation/track_results/MOT17-04-FRCNN/MOT17-04-FRCNN_val_gt.json" 
            video_path=gt_video_path # '/home/sophie/uncertain-identity-aware-tracking/Bytetrack/YOLOX_outputs/yolox_x_ablation/videos/MOT17-04-FRCNN.mp4'  
            track_file=dct_tmp_match[ base_path_2+"_tracking_result.json"]
            dbn_file= dct_tmp_match[base_path_2+"_DBN_result.json"]
            def score_for_various_artificial_observations_mot():

                hmm_result_with_visits=pd.DataFrame(columns=["nbr of visits", "accuracy", "recall", "f1", "idf1"])
                re_id_result_with_visits=pd.DataFrame(columns=["nbr of visits", "accuracy", "recall", "f1", "idf1"])
                base_path_2 = base_path_datasets+dataset_name+'/'+dataset_name #gt_video' #'
                
                if os.path.exists(base_path_2+'accuracy_over_nbr_of_visits_with_track_helping_nr'+exp+'.csv'):
                    hmm_result_with_visits=pd.read_csv(base_path_2+'accuracy_over_nbr_of_visits_with_track_helping_nr'+exp+'.csv')
                if os.path.exists(base_path_2+'accuracy_Re_id_over_nbr_of_visits_with_track_helping_nr'+exp+'.csv'):
                    re_id_result_with_visits=pd.read_csv(base_path_2+'accuracy_Re_id_over_nbr_of_visits_with_track_helping_nr'+exp+'.csv')
                
                """
                #put down tracking result on a video
                track =read_data_str_frame(base_path_2+"_tracking_result.json")
                put_results_on_video ( track , video_path="/home/sophie/uncertain-identity-aware-tracking/Bytetrack/YOLOX_outputs/yolox_x_ablation/videos/MOT17-05-FRCNN/gt_video.mp4",  save_path= base_path_2+"_tracking_result.mp4")
                return 0"""
            
                #add error bar here 
                for j in range(0,1,1):
                    print('rep')
                    for i in range (5, 4000 , 2000):# (10, 5000 , 200)range (2, 200 , 10): #[10, 100]:#  [18]: # len(label_track.keys())            
                        #home_folder=home_folder#''#/home/sophie/uncertain-identity-aware-tracking/Bytetrack/'
                        observation_file=os.path.join(tmp_folder,exp+str(i)+"_"+str(j)+"_DBN_result_with_observations_visits_per_id.json")
                        Hmm_result_file=os.path.join(tmp_folder,exp+str(i)+"_"+str(j)+"_with_atq_tracking_with_HMM_result_per_id.json")
                        re_id_track_result_file = os.path.join(tmp_folder,exp+str(i)+"_"+str(j)+"_per_id_re_id.json")

                        nbr_visits=i
                        label_track = read_data(gt_path)

                        #convert_mot_format_to_json_format(gt_path='/home/sophie/uncertain-identity-aware-tracking/Bytetrack/datasets/mot/train/MOT17-04-FRCNN/gt/gt_val_half.txt',
                        #                          destination_path=gt_path )
                        
                        
                        kept_visit=adding_atq(nbr_visit=i,curated_artificial_visit=None, output_file=observation_file, feeder=False, 
                                    track_file= track_file,
                                    dbn_file= dbn_file,
                                    labels_file=gt_path,
                                is_it_random = is_it_random, model=False,fairmot=False,for_re_id=False)
                            
                        
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
                            
                        new_row= {'nbr of visits':nbr_visits, 'kept_visit':kept_visit, 'accuracy':acc, 'recall':rec, "f1":f1, "idf1":idf1}
                        print(new_row)
                        hmm_result_with_visits = pd.concat([hmm_result_with_visits, pd.DataFrame([new_row])], ignore_index=True)


                        ####Re_id_part of the work
                    
                        tracking_result =produce_re_id_results(track_with_observation_file =observation_file , tracking_file=track_file, re_id_track_result_file = re_id_track_result_file , save_video=True, video_path=gt_video_path)
                        re_id_track = read_data(re_id_track_result_file)
                        print("***Oki before re_id_accuracy")
                        acc, rec, f1,idf1= precise_accuracy_track(label_track, re_id_track,  gt_base_number=-2,basic_tracker=True)
                        new_row= {'nbr of visits':nbr_visits,'kept_visit':kept_visit, 'accuracy':acc, 'recall':rec, "f1":f1, "idf1":idf1}
                        print(new_row)
                        re_id_result_with_visits = pd.concat([re_id_result_with_visits, pd.DataFrame([new_row])], ignore_index=True)
                        
                        hmm_result_with_visits.to_csv(base_path_2+'accuracy_over_nbr_of_visits_with_track_helping_nr'+exp+'.csv')
                        re_id_result_with_visits.to_csv(base_path_2+'accuracy_Re_id_over_nbr_of_visits_with_track_helping_nr'+exp+'.csv')
                        
                        #put_results_on_video ( tracking_result , save_path="video/re_id_192.mp4" , video_path="videos/GR77_20200512_111314.mp4", track_with_observation_file=track_with_observation_file)
                    #break
                print(re_id_result_with_visits)        
                print(hmm_result_with_visits)        


            score_for_various_artificial_observations_mot()
            #break
        
    '''
        


    def score_for_various_artificial_observations_mot(dataset_name):
        print(f"Processing {dataset_name}...")

        base_path_2 = base_path_datasets + dataset_name + '/' + dataset_name
        gt_video_path = dct_tmp_match[home_folder + "YOLOX_outputs/yolox_x_mix_mot20_ch/track_results/videos/" + dataset_name + "/gt_video.mp4"]
        gt_path = dct_tmp_match[os.path.join(home_folder + "YOLOX_outputs/yolox_x_mix_mot20_ch/track_results", dataset_name + "/" + dataset_name + "_gt.json")]
        track_file = dct_tmp_match[base_path_2 + "_tracking_result.json"]
        dbn_file = dct_tmp_match[base_path_2 + "_DBN_result.json"]

        hmm_result_with_visits = pd.DataFrame(columns=["nbr of visits", "accuracy", "recall", "f1", "idf1"])
        re_id_result_with_visits = pd.DataFrame(columns=["nbr of visits", "accuracy", "recall", "f1", "idf1"])

        """if os.path.exists(base_path_2 + 'accuracy_over_nbr_of_visits_with_track_helping_nr' + exp + '.csv'):
            hmm_result_with_visits = pd.read_csv(base_path_2 + 'accuracy_over_nbr_of_visits_with_track_helping_nr' + exp + '.csv')
        if os.path.exists(base_path_2 + 'accuracy_Re_id_over_nbr_of_visits_with_track_helping_nr' + exp + '.csv'):
            re_id_result_with_visits = pd.read_csv(base_path_2 + 'accuracy_Re_id_over_nbr_of_visits_with_track_helping_nr' + exp + '.csv')
        """
        for j in range(0, 4, 1):
            for i in range(5, 4000, 200):
                observation_file = os.path.join(tmp_folder, exp + str(i) + "_" + str(j) + "_DBN_result_with_observations_visits_per_id.json")
                Hmm_result_file = os.path.join(tmp_folder, exp + str(i) + "_" + str(j) + "_with_atq_tracking_with_HMM_result_per_id.json")
                re_id_track_result_file = os.path.join(tmp_folder, exp + str(i) + "_" + str(j) + "_per_id_re_id.json")

                label_track = read_data(gt_path)
                kept_visit = adding_atq(nbr_visit=i, curated_artificial_visit=None, output_file=observation_file, feeder=False, 
                                        track_file=track_file, dbn_file=dbn_file, labels_file=gt_path,
                                        is_it_random=is_it_random, model=False, fairmot=False, for_re_id=False, min_observation_max_value=min_observation_max_value)

                try:
                    process_forwad_backward(observation_file, nbr_visit="per_id_2_", pigs_HMM=False, json_save_path=Hmm_result_file, video_path=gt_video_path)
                    hmm_track = read_data(Hmm_result_file)
                    acc, rec, f1, idf1 = precise_accuracy_track(label_track, hmm_track, gt_base_number=0, basic_tracker=True)
                except Exception as e:
                    print(f"Error in forward-backward processing: {e}")
                    acc, rec, f1, idf1 = 0, 0, 0, 0

                hmm_result_with_visits = pd.concat([hmm_result_with_visits, pd.DataFrame([{
                    'nbr of visits': i, 'kept_visit': kept_visit, 'accuracy': acc, 'recall': rec, "f1": f1, "idf1": idf1
                }])], ignore_index=True)

                tracking_result = produce_re_id_results(track_with_observation_file=observation_file, tracking_file=track_file, 
                                                        re_id_track_result_file=re_id_track_result_file, save_video=True, video_path=gt_video_path)
                re_id_track = read_data(re_id_track_result_file)
                acc, rec, f1, idf1 = precise_accuracy_track(label_track, re_id_track, gt_base_number=-2, basic_tracker=True)

                re_id_result_with_visits = pd.concat([re_id_result_with_visits, pd.DataFrame([{
                    'nbr of visits': i, 'kept_visit': kept_visit, 'accuracy': acc, 'recall': rec, "f1": f1, "idf1": idf1
                }])], ignore_index=True)

                hmm_result_with_visits.to_csv(base_path_2 + 'accuracy_over_nbr_of_visits_with_track_helping_nr' + exp + '.csv')
                re_id_result_with_visits.to_csv(base_path_2 + 'accuracy_Re_id_over_nbr_of_visits_with_track_helping_nr' + exp + '.csv')

        print(f"Finished processing {dataset_name}.")

    #if __name__ == "__main__":
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.map(score_for_various_artificial_observations_mot, folders)#in folders if 'video' not in d and d==dataset_interest])
