import cv2
import math
import os 
import cv2
import sys
from os import listdir
from os.path import isfile, join, isdir




def match_images_from_all_directories(directory):
    '''
    Go throughs all the manga in the directory, and tries to ensure they are aligned via simple 
    color matching
    '''
    onlydir = [f for f in listdir(directory) if isdir(directory)][1:]
    fails=[]
    success=[]
    empty=[]
    for manga in onlydir:
        manga_path:str=os.path.join(directory,manga)
        
      
        try:
            print(manga)
            results:list=match_images_from_directory(manga_path)
            if len(results)>0:
                success.append(results)
            else:
                empty.append(manga)
        except:
            fails.append(manga)
    
    
    return success,empty,fails



def match_images_from_directory(data_path:str):
    '''
    Matches files from english directory to the japanese, finding best match, path, eng file, ja match, and score
    data_path:str="/home/jupyter/ComicTransfer/data/Conan/"
    '''
    saved_pairs=[]
    print("OK")
    eng_path:str=os.path.join(data_path,"en")
    ja_path:str=os.path.join(data_path,"ja")
        
    onlyfilesen = [f for f in listdir(eng_path) if isfile(join(eng_path, f))]
    onlyfilesja = [f for f in listdir(ja_path) if isfile(join(ja_path, f))]
    
    if math.fabs(len(onlyfilesen)-len(onlyfilesja))>5:
        return
    matches_1={}
    for eng_file in onlyfilesen:
        min_score=999999
        best_pic=""

        eng_file_path=os.path.join(eng_path,eng_file)
        src_img = cv2.imread(eng_file_path, cv2.IMREAD_COLOR)  # trainImage
        for filepath in onlyfilesja:

            alt_image = cv2.imread(os.path.join(ja_path,filepath), cv2.IMREAD_COLOR)  # trainImage

            score=keyPointScore(src_img,alt_image)
            if score<min_score:
                min_score=score
                best_pic=filepath
        matches_1[eng_file]=best_pic
        print(min_score)
        if min_score >1000 and min_score < 175000:
            eng_index=int(eng_file.replace(".png",""))
            ja_index=int(best_pic.replace(".png",""))
            print(eng_file)
            print(best_pic)
            if math.fabs(eng_index-ja_index)<3:

                saved_pairs.append([data_path,eng_file,best_pic,min_score])
    return saved_pairs
            

def colorScore(img1, img2):
    abs_sum_error=0
    abs_sum_error+=math.fabs(img1.T[0].mean()-img2.T[0].mean())
    abs_sum_error+=math.fabs(img1.T[1].mean()-img2.T[1].mean())
    abs_sum_error+=math.fabs(img1.T[2].mean()-img2.T[2].mean())
    return abs_sum_error

def keyPointScore(img1,img2):
    orb = cv2.ORB_create() 
    queryKeypoints, queryDescriptors = orb.detectAndCompute(img1,None) 
    trainKeypoints, trainDescriptors = orb.detectAndCompute(img2,None) 
    matcher = cv2.BFMatcher() 
    matches = matcher.match(queryDescriptors,trainDescriptors) 
    sums=0.0
    for match in matches:
        sums+=match.distance
    return sums

def simpleDiff(img1,img2):
    abs_sum_error=0
    abs_sum_error+=math.fabs((img1.T[0]-img2.T[0]).sum())
    abs_sum_error+=math.fabs((img1.T[1]-img2.T[1]).sum())
    abs_sum_error+=math.fabs((img1.T[2]-img2.T[2]).sum())
    return abs_sum_error