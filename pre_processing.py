import glob
import cv2 as cv
import numpy as np
from intrinsic_matrix import compute_intrinsic_matrix
from info_track import ImageView, MetaInfo
from initialize import initialization
import inspect
from tqdm import tqdm 

def debug_info(Frames):
    for view in Frames:
        print(len(view.keypoints))
        print(len(view.descriptors))

def main_flow():
    Frames = []
    metainfo = MetaInfo()

    #Open Images and compute SIFT features 
    initialization(Frames, metainfo)
    print('Total Keypoints:', metainfo.total_feature_points) 
    
    
main_flow()
