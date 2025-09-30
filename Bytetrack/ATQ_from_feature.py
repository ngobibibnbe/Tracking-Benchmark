import pandas as pd
import numpy as np 
import datetime as dt
import json 
import copy
import cv2
import math
import random
#random.seed(42)
import xml.etree.ElementTree as ET
#from ultralytics import YOLO

from datetime import timedelta

###########reading important files and setting the max number of frame ##########
home_folder= "/home/sophie/uncertain-identity-aware-tracking/Bytetrack"
visits_with_frame=[]
min_detection_for_artificial_visits=0.0005 #minimum iou of the corresponding detected  bbox in the tracker for an artificial visit
epsilone=0.00000000000001#000001 # to avoid null distance in the inverse calculus 
min_observation_max_value =0.0005 #minimum probability of the highest probable bbox corresponding to an artificial visit
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

def adding_atq(nbr_visit, output_file, feeder=False, 
               video_debut=dt.datetime(2020, 5, 12, 9, 0,0),
                video_fin= dt.datetime(2020, 5, 12, 9, 10,0), 
                track_file=home_folder+"/videos/GR77_20200512_111314tracking_result.json",
                dbn_file= home_folder+"/videos/GR77_20200512_111314DBN_result.json",
                
                labels_file=home_folder+"/videos/labels_with_atq.json",
                feeder_file=home_folder+"/videos/donnees_insentec_lot77_parc6.xlsx",
                water_file=home_folder+"/videos/eau_parc6.xlsx",
               is_it_random = True, model=True,curated_artificial_visit=None, fairmot=False, for_re_id=False):
   

    with open(track_file) as f:
            tracks = json.load(f) 
    with open(dbn_file) as f:
            dbn_infos = json.load(f) 
    

    max_frame=max([int(i) for i in list(dbn_infos.keys())])
        
    """ add atq depending on the labels file provided, and the number of observations we would like to have 
    
    Returns:
        write in a file: Bytetrack/videos/GR77_20200512_111314DBN_result_with_observations.json 
    """
    
    

    ########################defining utils functions#####################
    def convert_to_json(o):
        try:
            o=o.__dict__
        except:
            o=str(o)
        return o

    def iou (boxA,boxB=[580, 0, 90, 115+20]):
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

    def eucledian_distance(a,b):
        a=np.array(a)
        b=np.array(b)
        dist = np.linalg.norm(a-b)  
        return dist
    
    ##############################################################################""###  


    ########################Here we create the observations (HMM emission matrix) on visit at the feeders #############
    ##############################################################################################
    
    for key in dbn_infos.keys() :
        dbn_infos[key]["observation"]={}
    tracks_atq={}
    
    if feeder==True:
        water_visits=pd.read_excel(water_file)
        """feeder_visits=pd.read_excel(feeder_file)
        feeder_visits["debut"] = feeder_visits["Date_fin"].combine(feeder_visits["Tfin"], lambda d, t: pd.datetime.combine(d, t))
        feeder_visits['debut'] = [feeder_visits['debut'][idx] - timedelta(seconds=feeder_visits['Duree_s'][idx]) for idx in feeder_visits.index ]
        """
        water_center=[625,70]
        feeder_center=[90, 102]

        #on selectionne les visites qui sont sensées être dans la vidéo
        water_visits = water_visits.loc[(water_visits["debut"]>dt.datetime(2020, 5, 12, 9, 0,0)) & (water_visits["debut"]<dt.datetime(2020, 5, 12, 9, 9,59)) ]
        #feeder_visits = feeder_visits.loc[(feeder_visits["debut"]>dt.datetime(2020, 5, 12, 9, 0,0)) & (feeder_visits["debut"]<dt.datetime(2020, 5, 12, 9, 9,59)) ]
        #print(len(water_visits), len(feeder_visits))
        
        
        
      
        
        def add_observations(visits, feeder_center=[625,70]):
            nbr_of_visits=0
            for idx, visit in visits.iterrows(): 
                atq = float(visit["animal_num"])
                debut = visit['debut']
                fin = visit['fin']
                #on retrouve la frame de chaque visite 
                # de plus on rajoute une marge entre les visites pour éviter les problèmes de confusions 
                #d'identités quand deux animaux viennent bagarer à la mangeoire
                
                # je rajoute +50 frame de marge entre les debuts et fin de visites 
                frame_id_debut = int((debut-dt.datetime(2020, 5, 12, 9, 0,0)).total_seconds()*24.63666666666)+100 # +2 secondes
                frame_id_fin =  int((fin-dt.datetime(2020, 5, 12, 9, 0,0)).total_seconds()*24.63666666666)-100 #-2 secondes 
                frame_id=frame_id_debut+1
                flag=False
                while frame_id<frame_id_fin: 
                    frame_id=frame_id+1
                    if max_frame> frame_id:
                        frame=tracks[str(frame_id)]
                        max_d = 0
                        id_track_min =None          
                        """for track_id, track in frame.items(): 
                            # on calcule l'iou de chaque animal par rapport à la mangeoire et on vérifie qu'on a au moins un animal à la mangeoire
                            if iou(track, boxB=[feeder_center[0]-45, feeder_center[0]-70, 90, 115+20])>max_d:  #(x,y,w,h)
                                id_track_min=track_id
                                max_d =iou(track)"""
                        if max_d >=0 :
                            observation=[]
                            for track in dbn_infos[str(frame_id)]["current"]:
                                ### on  pourrait faire la gaussienne ici
                                tests= [[track["location"][0]+track["location"][2], track["location"][1]+track["location"][3]],[track["location"][0]+track["location"][2], track["location"][1]], [track["location"][0], track["location"][1]+track["location"][3]], [track["location"][0], track["location"][1]]]
                                #track_coin = [track["location"][0]+track["location"][2]/2, track["location"][1]+track["location"][3]/2]
                                min_dist=float('inf')
                                for coin in tests:
                                    if eucledian_distance(feeder_center, coin)<min_dist:
                                        min_dist=eucledian_distance(feeder_center, coin)
                                        track_coin=coin
                                        
                                observation.append(math.pow(max(eucledian_distance(feeder_center, track_coin),epsilone),2))
                                #l'observation est donné par la softmax sur les distance
                            ####transforming distances to probabilities  ****Our peut être remplacer par une gaussienne plus tard
                            observation = np.array(observation)
                            observation = 1/(1+observation)
                            observation = observation/sum(observation)
                            if max(observation)>=0.5:
                                dbn_infos[str(frame_id)]["observation"][atq]=observation
                                if is_it_random ==True and feeder==False: 
                                    if random.choice( [False, False, False,  True] ) ==True: # False, False, False, False,
                                        dbn_infos[str(frame_id)]["observation"][atq]=np.random(loc=0, scale=1, size=(len(observation))).tolist() #***random of lengt between 0 and 1 
                                #dbn_infos[str(frame_id)]["observed"]=atq
                                if flag==False:
                                    flag=True
                            else:
                                print("max observation is lower than 0")
                    else:
                        print("**not in the video",frame_id)
                
                if flag==True:
                    nbr_of_visits+=1
            print("**********nbr of rewarded visits",nbr_of_visits)
        #add_observations(feeder_visits, feeder_center)
        add_observations(water_visits, water_center)

        #####################################################################################################
        #############We create the observations based on a model#######
        """elif model==True:
            print('ok')
            video_path="Bytetrack/videos/GR77_20200512_111314.mp4"
            model = YOLO('fairmot/best.pt')  # load a pretrained model (recommended for training)
            cap = cv2.VideoCapture(video_path)
            
            def add__model_observations():
                frame_id = 0
                model_tracking={}
                while cap.isOpened():
                    model_tracking[frame_id]={}
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # Run detection
                    cv2.imwrite('tmp.jpg', frame)
                    results = model('tmp.jpg')  # predict on an image
                    #print(len(results))
                    names = results[0].names
                    for result in results:
                        boxes = result.boxes  # Boxes object for bounding box outputs
                        cls = boxes.cls  # Masks object for segmentation masks outputs
                        conf = boxes.conf  # Keypoints object for pose outputs
                        boxes = boxes.xywh  # Probs object for classification outputs
                        #print('probs', boxes)
                        for cl_idx, cl in enumerate(cls):
                            cl =names[int(cl)]
                            cl = float (cl.split("object_")[-1].split('-')[0])
                            if cl>4800 and conf[cl_idx]>0.0:
                                cl=str(cl)
                                max_one= 0
                                max_id =None
                                for track_idx, track in enumerate(dbn_infos[str(frame_id)]["current"]):
                                    if iou(list(boxes[cl_idx]),track["location"])>max_one:
                                        max_id = track_idx
                                        max_one = iou(list(boxes[cl_idx]),track["location"])
                                if cl not in dbn_infos[str(frame_id)]["observation"].keys():
                                    dbn_infos[str(frame_id)]["observation"][cl]=np.zeros((len(dbn_infos[str(frame_id)]["current"]),1)).tolist()
                                if max_id is not None:
                                    dbn_infos[str(frame_id)]["observation"][cl][max_id].append(conf[cl_idx].cpu() )
                    
                    #normaliser les observations et les ramener en une raw
                    observations= dbn_infos[str(frame_id)]["observation"]
                    detections= dbn_infos[str(frame_id)]["current"]
                    for cl  in observations.keys():
                        for track_idx in range(len(observations[cl])):
                            tmp=np.mean(np.array(observations[cl][track_idx]))
                            if tmp==0:
                                tmp=0.01
                            observations[cl][track_idx] =tmp
                        model_tracking[frame_id][cl]=detections[observations[cl].index(max(observations[cl]))]["location"]
                    dbn_infos[str(frame_id)]["observation"] =observations
                    
                    
                    frame_id += 1
                with open('Bytetrack/videos/model_yolo_result.json', 'w') as file:
                    json.dump(model_tracking, file)
            add__model_observations()        
            cap.release()
        """
        
        
        #####################################################################################################
        #############We create the observations using real labels ########### when we do random selection 
    else:
        with open(labels_file) as f:
            labels = json.load(f) 
        
        if curated_artificial_visit is None and feeder==True:
            #V=50#Nomber of random visits 
            idx_selection = [ i for i in range(0, len(labels.keys()), int(len(list(labels.keys()))/nbr_visit) ) ] #random.sample(list(labels.keys()), nbr_visit)
            random_selection = [list(labels.keys())[i] for i  in idx_selection ]
            
            selections_frame_id={}
            for frame_id in random_selection:
                selections_frame_id[frame_id]=random.sample(list(labels[frame_id].keys()), 1)
        
        elif curated_artificial_visit is None:
            #V=50#Nomber of random visits 
            idx_selection = [ i for i in range(0, len(labels.keys()), max(1,int(len(list(labels.keys()))/nbr_visit)) ) ] #random.sample(list(labels.keys()), nbr_visit)
            random_selection = [list(labels.keys())[i] for i  in idx_selection ]
            
            selections_frame_id={}
            for frame_id in random_selection:
                #print(len(labels[frame_id].keys()),max(1,int(nbr_visit/len(random_selection))  ))
                selections_frame_id[frame_id]=random.sample(list(labels[frame_id].keys()), min( len(labels[frame_id].keys()),max(1,int(nbr_visit/len(random_selection)) ) ) )
                
        
        if curated_artificial_visit is not None: 
            id_and_frames={}

            #we transform to have id as key s of our dictionnary
            for frame_id in labels.keys():
                for id in labels[frame_id].keys():
                    if id not in id_and_frames.keys():
                        id_and_frames [id]={}
                    id_and_frames [id][frame_id] = labels[frame_id][id]
                    
                    
            
            #print(id_and_frames.keys(), labels.keys())
            #we randomly selected x frames amongs frames on which the id is appearing in the label_track
            selections_id_frame={}
            for id in id_and_frames.keys():
                #print(id, list(id_and_frames[id].keys()))
                selections_id_frame[id] = random.sample(list(id_and_frames[id].keys()), min(curated_artificial_visit, len(id_and_frames[id])))
                if min(curated_artificial_visit, len(id_and_frames[id]))< curated_artificial_visit:
                    print(id, '****has less visits than expected , those are its visits',list(id_and_frames[id].keys()))
                
                
            #we transforms back to frame-id format
            selections_frame_id={}
            for id in selections_id_frame.keys():
                for frame_id in selections_id_frame[id]:
                    if frame_id not in selections_frame_id.keys():
                        selections_frame_id[frame_id]=[]
                    selections_frame_id[frame_id].append(id)
                
            print("those are frames and selected visitis",selections_frame_id)    
        #########################################################################
        not_in_video=0
        nbr_of_kept_visits=0
        kept_visits={}
        for frame_id in selections_frame_id.keys(): #random_selection:
            
            if str(frame_id) in tracks.keys():
                for visitor_id in selections_frame_id[frame_id]:
                    #########visitor_id = random.sample(list(labels[frame_id].keys()), 1)[0]
                    #visitor_id = list(labels[frame_id].keys())[0]
                    #print("***", frame_id, visitor_id)
                    if visitor_id!="observed": # and float(visitor_id)>15:  #####twick to modify later ..........................;;;;
                        #print(frame_id,labels[frame_id], visitor_id)
                        visitor_coordinate=labels[frame_id][visitor_id]
                        max_d = 0
                        id_track_min =None   
                        
                        ##we test if there is at least one object corresponding to the selected visitor in the track  
                        #we don't manage if there is not       
                        for  idx, track in enumerate(dbn_infos[str(frame_id)]["current"]): 
                            track_id = track['id_in_frame']
                            track=track["location"]
                            # on calcule l'iou de chaque animal par rapport à la mangeoire et on vérifie qu'on a au moins un animal à la mangeoire
                            if iou(track,visitor_coordinate)>max_d:
                                idx_min=idx
                                max_d =iou(track, visitor_coordinate)
                        
                                
                        if max_d >=min_detection_for_artificial_visits:#and max_d_x<= 620:
                            #instead of using the corresponding bbox in detection i will use the distance
                            visitor_coordinate =  dbn_infos[str(frame_id)]["current"][idx_min]["location"] #labels[frame_id][visitor_id] #
                            feeder_center = [visitor_coordinate[0], visitor_coordinate[1]]
                            ##print(max_d, "*****")
                            observation=[]
                            observations_iou=[]
                            
                            
                            #################################to be checked 
                            max_iou_with_visitor =0
                            track_of_max_iou_with_visitor=None
                            for track in dbn_infos[str(frame_id)]["current"]:
                                iou_track = iou(visitor_coordinate,track["location"])
                                if iou_track>max_iou_with_visitor:
                                    max_iou_with_visitor=iou_track
                                    track_of_max_iou_with_visitor =track

                                
                                observation.append(iou(visitor_coordinate,track["location"]))

                            observation = np.array(observation)
                            """observation = 1/(1+observation)"""
                            observation = observation/sum(observation)
                            if max(observation)>0:
                                observation=[]
                                
                                for track in dbn_infos[str(frame_id)]["current"]:                             
                                    distance_feature = eucledian_distance(track["detection_feat"], track_of_max_iou_with_visitor["detection_feat"])
                                    observation.append(distance_feature)
                                    #observation.append(eucledian_distance(feeder_center, track_coin))
                                    #observation.append(1-iou(track["location"], visitor_coordinate))
                                
                                observation = np.array(observation)
                                observation = observation/sum(observation)
                                dbn_infos[str(frame_id)]["observation"][visitor_id]=observation.tolist()
        
                                #########################################################end to be checked 
                                
                                if frame_id not in kept_visits.keys():
                                    kept_visits[frame_id]=[]
                                kept_visits[frame_id].append(visitor_id)
                                #dbn_infos[str(frame_id)]["observed"]=atq
                        else:
                            not_in_video+=1
                            print("we didnt find a detection with the minimum iou requirement for  visitor", visitor_id, " at frame", frame_id)
                            #print("id", visitor_id,"not detected by the tracker in the video at frame",frame_id, "total:",not_in_video)
                #if flag==True:
                #    nbr_of_visits+=1
            


    print("we finished adding observations")
    #####################################################################################################
    randoms_observation=0
    for frame_id,frame in dbn_infos.items():
        current_to_stay=[]
        ids_to_stay=[]
        ids_to_stay_prev=[]
        if int(frame_id)<max_frame and frame_id!="0" and frame_id!=list(dbn_infos.keys())[-1] and len(dbn_infos[str(frame_id)]["matrice"])!=0: ###???change t300 to -1   int(frame_id)<300 and 
            for idx,i in  enumerate(dbn_infos[str(frame_id)]["current"]):
                for id_prev,j in enumerate(dbn_infos[str(int(frame_id)+1)]["previous"]):
                    if  i["location"]==j["location"]: ## int(i["id_in_frame"])==int(j["id_in_frame"]) and
                        #current_to_stay.append(i)
                        ids_to_stay_prev.append(id_prev)
                        ids_to_stay.append(idx)
            
            assert len(ids_to_stay)==len(ids_to_stay_prev), "previous and current does not have the same size"

            """if  for_re_id:
                ids_to_stay_prev = [i for i in range(len(dbn_infos[str(int(frame_id)+1)]["previous"]))]#   if dbn_infos[str(int(frame_id)+1)]["previous"][i]["track_id"] is not None ]

                ids_to_stay =[i for i in range(len(dbn_infos[str(frame_id)]["current"]))]#  if dbn_infos[str(int(frame_id))]["current"][i]["track_id"] is not None]
            """

            #we make sure what is in current of frame t and in previous in frame t+1 are the same objects identicallly 
            #ids_to_stay_prev.sort()# on trie parce que dans strackpool les detection sont récupérés dans l'ordre des matching avec les frames précédentes dans la fonction update de bytetrack.py
            #print("****",frame_id, np.array(dbn_infos[str(frame_id)]["matrice"]).shape, len(ids_to_stay_prev),)

            dbn_infos[str(frame_id)]["current"]=np.array(dbn_infos[str(frame_id)]["current"])[ids_to_stay].tolist()#current_to_stay 

            dbn_infos[str(int(frame_id)+1)]["previous"]=np.array([dbn_infos[str(int(frame_id)+1)]["previous"][id_prev] for id_prev in ids_to_stay_prev]).tolist()#current_to_stay 
            
            keys_to_del=[]
            for key in  dbn_infos[str(frame_id)]["observation"].keys(): 
                #print(dbn_infos[str(frame_id)]["observation"][key])
                #print(np.array(dbn_infos[str(frame_id)]["observation"][key])[ids_to_stay].tolist())
                dbn_infos[str(frame_id)]["observation"][key]= np.array(dbn_infos[str(frame_id)]["observation"][key])[ids_to_stay].tolist()
                #print(type(dbn_infos[str(frame_id)]["observation"][key]))

                if max(dbn_infos[str(frame_id)]["observation"][key])<=min_observation_max_value:
                    keys_to_del.append(key)
                    not_in_video+=1
                    print(frame_id,"in this frame we will remove", key, "because of min_max_observation threshold on artificial informations" )
                else:
                    nbr_of_kept_visits+=1
                    observation=dbn_infos[str(frame_id)]["observation"][key]
                    if is_it_random ==True and feeder==False: 
                        #print("*************we enter in random")
                        if random.choice( [False, False, False, True] ) ==True:
                            random_values_uniform = np.random.rand(len(observation)).tolist()
                            # Normalize so that the sum is 1
                            normalized_values = [x / sum(random_values_uniform) for x in random_values_uniform]
                            dbn_infos[str(frame_id)]["observation"][key]=normalized_values
                            randoms_observation+=1
                          
                          
            for key in keys_to_del:
                del dbn_infos[str(frame_id)]["observation"][key]

            #if "observed" in list(dbn_infos[str(frame_id)].keys()):
            #    dbn_infos[str(frame_id)]["observation"]=np.array(dbn_infos[str(frame_id)]["observation"])[ids_to_stay].tolist()
            
            #on s'assure de transformer chaque ligne de la trice de transition en distribution de probabilités et se rassurer que current at t is previous at t+1
            #print(frame_id, [ids_to_stay] )

            tmp_var=dbn_infos[str(frame_id)]["matrice"]
            dbn_infos[str(frame_id)]["matrice"] = np.array(dbn_infos[str(frame_id)]["matrice"])[:,ids_to_stay]
            dbn_infos[str(int(frame_id)+1)]["matrice"] =np.array(dbn_infos[str(int(frame_id)+1)]["matrice"] )
            dbn_infos[str(int(frame_id)+1)]["matrice"] = dbn_infos[str(int(frame_id)+1)]["matrice"][ids_to_stay_prev,:]
            
            if not fairmot:
                dbn_infos[str(frame_id)]["matrice"] = 1- dbn_infos[str(frame_id)]["matrice"]
            else:
                dbn_infos[str(frame_id)]["matrice"] = 1/dbn_infos[str(frame_id)]["matrice"]
            
            
            for idx,_ in enumerate(dbn_infos[str(frame_id)]["matrice"]):
                if dbn_infos[str(frame_id)]["matrice"][idx].sum()!=0:
                    dbn_infos[str(frame_id)]["matrice"][idx]= dbn_infos[str(frame_id)]["matrice"][idx]/dbn_infos[str(frame_id)]["matrice"][idx].sum()
            
            dbn_infos[str(int(frame_id))]["matrice_inter"]=np.zeros((len(dbn_infos[str(int(frame_id))]["previous"]),len(dbn_infos[str(int(frame_id))]["current"])))
            for id_previous in range(len(dbn_infos[str(int(frame_id))]["previous"])):
                dbn_infos[str(int(frame_id))]["matrice_inter"][id_previous]= np.array([ iou(dbn_infos[str(int(frame_id))]["previous"][id_previous]["location"], dbn_infos[str(int(frame_id))]["current"][id_current]["location"]) for id_current in range(len(dbn_infos[str(int(frame_id))]["current"]))])
                dbn_infos[str(int(frame_id))]["matrice_inter"][id_previous] = dbn_infos[str(int(frame_id))]["matrice_inter"][id_previous]/dbn_infos[str(int(frame_id))]["matrice_inter"][id_previous].sum()
            
            dbn_infos[str(int(frame_id))]["matrice_difference"] = dbn_infos[str(int(frame_id))]["matrice_inter"] - dbn_infos[str(frame_id)]["matrice"]
            #if dbn_infos[str(int(frame_id))]["matrice_difference"].max()>0.3:
            #    print("grosse différence entre l'iou et le sort+appearance de bytetrack sur la frame",frame_id)


            dbn_infos[str(frame_id)]["matrice"] = np.array(dbn_infos[str(frame_id)]["matrice"]).tolist()
            dbn_infos[str(frame_id)]["matrice_inter"] = np.array(dbn_infos[str(frame_id)]["matrice_inter"]).tolist()
            
            ############this is necessary because previous has id but not current that is why we did matchings previously
            for idx, track in enumerate(dbn_infos[str(frame_id)]["current"]):
                #if idx <len(dbn_infos[str(int(frame_id)+1)]["previous"]):
                dbn_infos[str(frame_id)]["current"][idx]["track_id"]=dbn_infos[str(int(frame_id)+1)]["previous"][idx]["track_id"]
                #print("an error on current", idx, "which has id", track["track_id"], "not corresponding on previous")


    with open(output_file, 'w') as outfile:
        json.dump(dbn_infos, outfile, default=lambda o: convert_to_json(o), indent=1)
        #print("the results txt files are #printed in", outfile)
        
    print("****We kept :", nbr_of_kept_visits, "with:",randoms_observation, " random observations", "and left:",not_in_video, )#S, "visits. They are:", kept_visits)
    return nbr_of_kept_visits
#adding_atq(nbr_visit=0,output_file='test_model_observation.json')
"""adding_atq(nbr_visit=1, feeder=True, output_file="test2_to_del", labels_file=home_folder+"/videos/labels_with_atq.json")"""

"""dbn_file= "/home/sophie/uncertain-identity-aware-tracking/Bytetrack/YOLOX_outputs/yolox_x_ablation/track_results/MOT17-04-FRCNN/fairmot/MOT17-04-SDP.json"
tracking_result_file="/home/sophie/uncertain-identity-aware-tracking/Bytetrack/YOLOX_outputs/yolox_x_ablation/track_results/MOT17-04-FRCNN/fairmot/fairmot_MOT17_04.json"
dbn_file= dbn_file.split(".json")[0]+"_half_val.json"

base_path= home_folder+"YOLOX_outputs/yolox_x_ablation/track_results/MOT17-04-FRCNN/fairmot/MOT17-04-FRCNN"

hmm_result_with_visits=pd.DataFrame(columns=["nbr of visits", "accuracy", "recall", "f1"])
re_id_result_with_visits=pd.DataFrame(columns=["nbr of visits", "accuracy", "recall", "f1"])

#add error bar here 
for j in range(0,1):

    for i in range (2, 200 , 100):# range (2, 200 , 10): #[10, 100]:#  [18]: # len(label_track.keys())            
        #home_folder=home_folder#''#/home/sophie/uncertain-identity-aware-tracking/Bytetrack/'
        observation_file=base_path+"_DBN_result_with_observations_visits_per_id.json"
        gt_path="/home/sophie/uncertain-identity-aware-tracking/Bytetrack/YOLOX_outputs/yolox_x_ablation/track_results/MOT17-04-FRCNN/MOT17-04-FRCNN_val_gt.json" 
        video_path= '/home/sophie/uncertain-identity-aware-tracking/Bytetrack/YOLOX_outputs/yolox_x_ablation/videos/MOT17-04-FRCNN.mp4'  

        #convert_mot_format_to_json_format(gt_path='/home/sophie/uncertain-identity-aware-tracking/Bytetrack/datasets/mot/train/MOT17-04-FRCNN/gt/gt_val_half.txt',
        #                          destination_path=gt_path )
        adding_atq(nbr_visit=0, output_file=observation_file, feeder=False, 
                    track_file=tracking_result_file,#****replace with tracking of fairmot
                    dbn_file= dbn_file,
                    labels_file=gt_path,
                is_it_random = False, model=False,curated_artificial_visit=i, fairmot=True)"""