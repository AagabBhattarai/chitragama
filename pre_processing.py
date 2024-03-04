import glob
import cv2 as cv
import numpy as np
from intrinsic_matrix import compute_intrinsic_matrix
from info_track import ImageView, MetaInfo
from initialize import initialization
import inspect
from tqdm import tqdm 

def debug_info(Views):
    for view in Views:
        print(len(view.keypoints))
        print(len(view.descriptors))

def main_flow():
    Views = []
    metainfo = MetaInfo()

    #Open Images and compute SIFT features 
    initialization(Views, metainfo)
    print('Total Keypoints:', metainfo.total_feature_points) 
    
    #Track 2D features for across all views
    feature_track = np.zeros((metainfo.total_views, metainfo.total_feature_points), dtype=bool)
    assert len(Views) == len(feature_track), "Assertion Error: metainfo.total_views not computed properly"
    

    
main_flow()
