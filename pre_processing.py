import glob
import cv2 as cv
import numpy as np
from intrinsic_matrix import compute_intrinsic_matrix
from info_track import ImageView, MetaInfo, ImagePair
from initialize import initialization
from utilities import display_3d, display_plot_simple
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
    # if query_view.id == 2 and comp_view.id == 1:
    #     print(len(matches))
    #     print(len(good)) 
    return good

def find_projection(qv: ImageView, tv: ImageView, matches):
    left=   np.float32([ qv.keypoints[m.queryIdx].pt for m in matches ]).reshape(-1,2)
    right = np.float32([ tv.keypoints[m.trainIdx].pt for m in matches ]).reshape(-1,2)
   
    E, mask = cv.findEssentialMat( left,right,qv.K,
                                    cv.RANSAC, prob=0.999, threshold=1.0)
    # print("Essential", E)

    success, R, t, fmask =  cv.recoverPose(E, left, right, qv.K, mask=mask)
    
    # m_shape = matches.shape
    i_matches = [m for i,m in enumerate(matches) if fmask[i]==1]
    # i_matches = matches[fmask.ravel() ==1]
    # assert m_shape == matches.shape, "Assertion Error, must reshape appropriately"
    print("Success",success)
    # if qv.id == 2 and tv.id ==1:
    #     print("Mask", mask)
    #     print(len(i_matches))   
    #     print(i_matches)
    temp = np.eye(4)
    proj2 = temp[:3, :4]
    proj2[:3, :3] = R
    proj2[0:3, 3:4] = t
    # print('Proj2', proj2) 
    return proj2, i_matches

def find_average_depth_and_median_angle(qv: ImageView, tv: ImageView, proj2, matches):
    temp = np.eye(4)
    sampling_rate = 30

    intrinsic_camera_matrix = qv.K
    proj1 = intrinsic_camera_matrix @ temp[:3, :4]
    p2 = intrinsic_camera_matrix @ proj2
    left=   np.float32([ qv.keypoints[m.queryIdx].pt for i,m in enumerate(matches) if i % sampling_rate ]).reshape(-1,2)
    right = np.float32([ tv.keypoints[m.trainIdx].pt for i,m in enumerate(matches) if i % sampling_rate ]).reshape(-1,2)
     
    recovered_3D_points_in_homogenous = cv.triangulatePoints(proj1, p2, left.T, right.T).T
    triangulated_points = (recovered_3D_points_in_homogenous[:, 0:3] / recovered_3D_points_in_homogenous[:, 3:]).reshape(-1, 3)

    norms = np.linalg.norm(triangulated_points[:, :3], axis=1)
    average_depth = np.mean(norms)

    # Calculate camera centers
    camera_center_1 = np.array([0, 0, 0])  # Assuming the first camera is at the origin
    R2, t2 = proj2[:3, :3], proj2[:3, 3]
    camera_center_2 = -np.matmul(R2.T, t2)

    angles = []
    for point in triangulated_points:
        # Vectors from camera centers to the point
        vector1 = point - camera_center_1
        vector2 = point - camera_center_2

        # Normalize vectors
        vector1_norm = vector1 / np.linalg.norm(vector1)
        vector2_norm = vector2 / np.linalg.norm(vector2)

        # Calculate angle
        cos_angle = np.dot(vector1_norm, vector2_norm)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))  # Clip to handle numerical issues
        angle_degrees = np.degrees(angle)

        angles.append(angle_degrees)

    median_angle = np.median(angles)

    print("View:", qv.id, "\tView:", tv.id)
    print('Average depth:', average_depth)
    print('Median transformed triangulation angle:', median_angle)

    capped_angle = min(median_angle, 45)
    median_angle = 2 * np.sqrt(capped_angle)
    return average_depth, median_angle

def match_and_find_scene_graph_relation(query_view: ImageView, Views: list, Scene_graph, bf_matcher, metainfo:MetaInfo):
    association = list()
    for i, comp_view in enumerate(Views):
        if query_view.id == comp_view.id:
            if(len(association)>0):
                Scene_graph.append(association)
            return
        sad_exit = False 
        matches = find_match(query_view, comp_view, bf_matcher)
        if(len(matches) > 30):
            proj2, i_matches = find_projection(query_view, comp_view, matches)
            if(len(i_matches) > 30):
                avg_depth, median_angle = find_average_depth_and_median_angle(query_view, comp_view, proj2, i_matches)
                upd_matches = []
                for m in i_matches:

                    if metainfo.do_bundle_adjustment:
                        if comp_view.global_descriptor[m.trainIdx] in query_view.global_descriptor_set and comp_view.global_descriptor[m.trainIdx] != query_view.global_descriptor[m.queryIdx]:#if space is empty then new gd shouldn't be same to any of the old ones, otherwise it will implying that different keypoints are the same
                            continue
                    upd_matches.append(m)
                    query_view.global_descriptor[m.queryIdx] = comp_view.global_descriptor[m.trainIdx]
                    query_view.global_descriptor_set.add(query_view.global_descriptor[m.queryIdx])
                imgpair = ImagePair(query_view.id, comp_view.id, proj2, upd_matches, avg_depth, median_angle)
                association.append(imgpair)
            else: 
                sad_exit = True
        else:
            sad_exit = True
        if sad_exit:
            imgpair = ImagePair(query_view.id, comp_view.id, np.ones((3,4)), [], 1, 0)
            association.append(imgpair)
    
            
def main_flow(recons_images):
    Views = []
    metainfo = MetaInfo()
    Scene_graph = []
    bf_matcher = cv.BFMatcher()
    #Open Images and compute SIFT features 
    initialization(Views, metainfo, recons_images)
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
                view.global_descriptor_set.add(view.global_descriptor[j])
        feature_track[i, view.global_descriptor] = True
        print("\nunique pts identified:", global_descriptor_index_value)
    
    print("Number of unique features:", global_descriptor_index_value)
    metainfo.unique_feature_points =global_descriptor_index_value 
    print('Total Keypoints:', metainfo.total_feature_points)
    
    #find two views to initialize object scene
    feature_track = feature_track[:, :global_descriptor_index_value]
    initaization_ids = find_initialization_view_id(feature_track, Scene_graph)
    if metainfo.do_bundle_adjustment:
        debug_scene_g(Views)
    
    return Views, Scene_graph, initaization_ids, metainfo, feature_track
    

def find_initialization_view_id(feature_track, Scene_graph, maximum_depth_for_acceptable_baseline=10):
    feature_repeat_frequency = np.sum(feature_track, axis=0)
    view1 = 1
    view2 = 0
    maximum_track_sum = 0
    largest_angle = 0  # Initialize largest angle to 0

    for association in Scene_graph:
        for img_pair in association:
            if img_pair.average_depth != 1 and img_pair.average_depth < maximum_depth_for_acceptable_baseline and img_pair.median_angle > 5 :
                common_points = np.logical_and(feature_track[img_pair.view_1], feature_track[img_pair.view_2])
                track_sum = np.sum(feature_repeat_frequency[common_points])
                # Check if this pair has the largest number of matches and the largest triangulation angle
                if track_sum > maximum_track_sum or (track_sum == maximum_track_sum and img_pair.median_angle > largest_angle):
                    maximum_track_sum = track_sum
                    largest_angle = img_pair.median_angle  # Update the largest angle
                    view1, view2 = img_pair.view_1, img_pair.view_2

    print("Maximum track sum:", maximum_track_sum)
    print("Largest triangulation angle:", largest_angle)
    print("View Initialization:", view1, "\tand\t", view2)
    return view1, view2


def debug_scene_g(Views):
    for view in Views:
        print("View Id:", view.id)
        print("GD length:", len(view.global_descriptor))
        print("Set GD length:", len(set(view.global_descriptor)))
        assert len(view.global_descriptor) == len(set(view.global_descriptor)), "Assertion Error: Global descriptor repeatation"
