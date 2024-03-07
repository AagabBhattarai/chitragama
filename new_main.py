from pre_processing import main_flow
from compute_points import model_initialization, register_new_view, triangulate_new_points, find_new_viewid
from info_track import ObjectPoints, ImageView
import numpy as np
from write_to_ply import write_to_ply_file
import sys
from tqdm import tqdm
import cv2 as cv
from bundle_adjustment import Bundle_Adjusment
from bundle_setup import do_bundle_adjustment
def main():
    arguments = sys.argv[1:]
    if len(arguments) == 0:
        arg = 'def'
    else: 
        arg = arguments[0]
    
    views_processed = list()
    Views, Scene_graph, initialization_ids, metainfo, feature_track = main_flow()
    bundle_adjustment = Bundle_Adjusment()

    set_gd_dict(Views)
    
    print("Initialization ID:", initialization_ids)
    object_points:ObjectPoints = model_initialization(Views, Scene_graph, initialization_ids, metainfo)
    views_processed.extend(initialization_ids)
    for _ in tqdm(range(metainfo.total_views - 2), desc="Processing Views"):
        finished, viewid = find_new_viewid(Views, views_processed, object_points, feature_track)
        if finished:
            break 

        register_new_view(viewid,Views, object_points, feature_track)
        for processed_view in views_processed:
            triangulate_new_points(Views, Scene_graph, (processed_view, viewid), object_points, metainfo)
        views_processed.append(viewid)
        set_camera_params(Views, views_processed, object_points)
        if metainfo.bundle_adjustment_time:
            set_BA_points(Views, views_processed, object_points)
            do_bundle_adjustment(bundle_adjustment, object_points, Views[viewid].K)
            update_view_pose(Views, views_processed, object_points)
            metainfo.error_sum = 0
            metainfo.bundle_adjustment_time = False
        
    update_camera_path(object_points)
    write_to_ply_file(object_points,arg)

def set_camera_params(Views, views_processed, object_points:ObjectPoints):
    for i,id in enumerate(views_processed):
        if id in object_points.camera_params_map:
            continue
        view = Views[id]
        R = view.extrinsic_pose[:3,:3].copy()
        rvec, _ = cv.Rodrigues(R)
        t = view.extrinsic_pose[:3, 3]
        object_points.camera_params.extend(rvec.ravel().tolist())
        object_points.camera_params.extend(t.tolist()) 
        object_points.camera_params_map[id] = i

        
        
def set_BA_points(Views, views_processed, object_points:ObjectPoints):
    points_2d = list()
    point_indices = list()
    camera_indices = list()
    for view in Views:
        if view.id not in views_processed:
            continue
        for i,gd in enumerate(object_points.pts_3D_global_descriptor_value):
            if gd in view.global_descriptor:
                index_for_point_2d = view.global_descriptor_and_index[gd]
                point_2d = view.keypoints[index_for_point_2d].pt
                points_2d.append(point_2d)
                point_indices.append(i)
                camera_indices.append(object_points.camera_params_map[view.id])
    
    object_points.points_2d = points_2d
    object_points.point_indices = point_indices
    object_points.camera_indices = camera_indices

def update_view_pose(Views, views_processed, object_points:ObjectPoints):
    object_points.camera_params = np.float32(object_points.camera_params).reshape(-1, object_points.n_camera_params_ba)
    for id in views_processed:
        view:ImageView = Views[id]
        view_params = object_points.camera_params[object_points.camera_params_map[id]].ravel()
        rvec = view_params[ : 3]
        t = view_params[3:6]
        R, _ = cv.Rodrigues(rvec)
        temp = np.empty((3,4))
        temp[:3,:3] = R
        temp[:3, 3] = t
        view.extrinsic_pose = temp.copy()
    object_points.camera_params = object_points.camera_params.ravel().tolist()

def set_gd_dict(Views):
    for view in Views:
        assert len(view.global_descriptor) == len(set(view.global_descriptor)), "Assertion Error: Global descriptor repeatation"
        view.global_descriptor_and_index = {gd:i for i, gd in enumerate(view.global_descriptor)}
        

def update_camera_path(object_points: ObjectPoints):
    object_points.camera_params = np.float32(object_points.camera_params).reshape(-1, object_points.n_camera_params_ba)
    length = len(object_points.camera_params)
    for i in range(length):
        rvec = object_points.camera_params[i, :3]
        tvec = object_points.camera_params[i, 3:6]
        rmat, _ = cv.Rodrigues(rvec)
        cam_path = (-rmat.T @ tvec).tolist()
        # object_points.camera_path[i] = cam_path
        object_points.camera_path.append(cam_path)

if __name__== "__main__":
    main()