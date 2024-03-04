import glob
import cv2 as cv
import numpy as np
from intrinsic_matrix import compute_intrinsic_matrix
from info_track import ImageView, MetaInfo, ImagePair
from initialize import initialization
from utilities import display_3d
import inspect
from tqdm import tqdm 

def debug_info(Views):
    for view in Views:
        print(len(view.keypoints))
        print(len(view.descriptors))

def find_match(query_view: ImageView, comp_view: ImageView, bf_matcher):
    matches = bf_matcher.knnMatch(query_view.descriptors, comp_view.descriptors, k=2)
    #Lowe's test
    good = []
    for m,n in matches:
        if m.distance < 0.5*n.distance:
            good.append(m)
    
    return good

def find_projection(qv: ImageView, tv: ImageView, matches):
    left=   np.float32([ qv.keypoints[m.queryIdx].pt for m in matches ]).reshape(-1,2)
    right = np.float32([ tv.keypoints[m.trainIdx].pt for m in matches ]).reshape(-1,2)
   
    E, mask = cv.findEssentialMat( left,right,qv.K,
                                    cv.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, fmask =  cv.recoverPose(E, left, right, qv.K, mask=mask)
    
    # m_shape = matches.shape
    i_matches = [m for i,m in enumerate(matches) if fmask[i]==1]
    # i_matches = matches[fmask.ravel() ==1]
    # assert m_shape == matches.shape, "Assertion Error, must reshape appropriately"
    
    temp = np.eye(4)
    proj2 = temp[:3, :4]
    proj2[:3, :3] = R
    proj2[0:3, 3:4] = t
    
    return proj2, i_matches

def to_camera_coordinate(K, point: list[float]) -> list[float]:
    normalized = [  (point[0] - K[0,2]) / K[0,0] ,  (point[1] - K[1,2])/K[1,1] ];
    return normalized

def find_average_depth(qv: ImageView, tv: ImageView, proj2, matches):
    temp = np.eye(4)
    sampling_rate = 30
    
    intrinsic_camera_matrix = qv.K
    proj1 = intrinsic_camera_matrix @ temp[:3, :4]
    p2 = intrinsic_camera_matrix @ proj2
    left=   np.float32([ qv.keypoints[m.queryIdx].pt for i,m in enumerate(matches) if i % sampling_rate ]).reshape(-1,2)
    right = np.float32([ tv.keypoints[m.trainIdx].pt for i,m in enumerate(matches) if i % sampling_rate ]).reshape(-1,2)
     
    recovered_3D_points_in_homogenous = cv.triangulatePoints(proj1, p2, left.T, right.T).T
    # Calculate norms for all 3D points 
    triangulated_points =  (recovered_3D_points_in_homogenous[:, 0:3]/recovered_3D_points_in_homogenous[:,3:]).reshape(-1,3)
    # display_3d(triangulated_points)
    norms = np.linalg.norm(triangulated_points[:, :3], axis=1)
    
    # Calculate average depth
    average_depth = np.mean(norms) 
    print("Frame:", qv.id, "\tFrame:", tv.id)
    print('Average depth:', average_depth)

    return average_depth

                                
    
def match_and_find_scene_graph_relation(query_view: ImageView, Views: list, Scene_graph, bf_matcher, metainfo):
    association = list()
    for i, comp_view in enumerate(Views):
        if query_view.id == comp_view.id:
            return
        
        matches = find_match(query_view, comp_view, bf_matcher)
        if(len(matches) > 30):
            proj2, i_matches = find_projection(query_view, comp_view, matches)
            avg_depth = find_average_depth(query_view, comp_view, proj2, i_matches)
            imgpair = ImagePair(query_view.id, comp_view.id, proj2, i_matches, avg_depth)
            for m in i_matches:
                query_view.global_descriptor[m.queryIdx] = comp_view.global_descriptor[m.trainIdx]
                query_view.global_descriptor_status[m.queryIdx] = True
        else:
            imgpair = ImagePair(query_view.id, comp_view.id, np.ones((3,4)), [], 1)
        association.append(imgpair)

    Scene_graph.append(association)
            
def main_flow():
    Views = []
    metainfo = MetaInfo()
    Scene_graph = []
    bf_matcher = cv.BFMatcher()
    #Open Images and compute SIFT features 
    initialization(Views, metainfo)
    print('Total Keypoints:', metainfo.total_feature_points) 
    
    #Track 2D features for across all views
    feature_track = np.zeros((metainfo.total_views, metainfo.total_feature_points), dtype=bool)
    assert len(Views) == len(feature_track), "Assertion Error: metainfo.total_views not computed properly"
    
    global_descriptor_index_value = 0 #No keypoint has been registered
    #find correspondences between two images
    for i,view in enumerate(Views):
        match_and_find_scene_graph_relation(view, Views, Scene_graph, bf_matcher, metainfo)
        for j in range(len(view.global_descriptor)):
            if view.global_descriptor[j] == -1:
                view.global_descriptor[j] = global_descriptor_index_value
                global_descriptor_index_value +=1 
        feature_track[i, view.global_descriptor] = True
        print("\nunique pts identified:", global_descriptor_index_value)
    
    print("Number of unique features:", global_descriptor_index_value)
    

    
main_flow()
