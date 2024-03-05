from pre_processing import main_flow
from compute_points import model_initialization, register_new_view, triangulate_new_points
from info_track import ObjectPoints
import numpy as np
from write_to_ply import write_to_ply_file
import sys
from tqdm import tqdm

def main():
    arguments = sys.argv[1:]
    if len(arguments) == 0:
        arg = 'def'
    else: 
        arg = arguments[0]
    
    views_processed = list()
    Views, Scene_graph, initialization_ids, metainfo, feature_track = main_flow()
    print("Initialization ID:", initialization_ids)
    object_points:ObjectPoints = model_initialization(Views, Scene_graph, initialization_ids, metainfo)
    views_processed.extend(initialization_ids)
    for _ in tqdm(range(metainfo.total_views - 2), desc="Processing Views"):
        finished, viewid = find_new_viewid(Views, views_processed, object_points, feature_track)
        if finished:
            break 

        register_new_view(viewid,Views, object_points, feature_track)
        for processed_view in views_processed:
            triangulate_new_points(Views, Scene_graph, (processed_view, viewid), object_points)
        views_processed.append(viewid)
    
    write_to_ply_file(object_points,arg)
    
        


def find_new_viewid(Views, views_processed, object_points:ObjectPoints, feature_track):
    max_track_sum =10
    view_n = -1
    for view in Views:
        if view.id not in views_processed:
            common_to_world = np.logical_and(feature_track[view.id], object_points.pts_3D_global_descriptor[:])
            track_sum = np.sum(common_to_world)    
            if track_sum > max_track_sum:
                view_n = view.id
                max_track_sum = track_sum
    finished = True if view_n == -1 else False
    print("New view:", view_n)
    return finished, view_n 

if __name__== "__main__":
    main()