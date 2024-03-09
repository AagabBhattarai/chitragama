from  bundle_adjustment import Bundle_Adjusment
import numpy as np
from info_track import ObjectPoints

def do_bundle_adjustment(bundle_adjustment:Bundle_Adjusment, object_points: ObjectPoints, intrinsic_camera_matrix):
    object_points.pts_3D = np.float32(object_points.pts_3D).reshape(-1,3)
    
    object_points.points_2d = np.float32(object_points.points_2d).reshape(-1,2)
    object_points.camera_params = np.float32(object_points.camera_params).reshape(-1, object_points.n_camera_params_ba)
    object_points.camera_indices = np.int32(object_points.camera_indices).ravel()
    object_points.point_indices = np.int32(object_points.point_indices).ravel()
    
    # if object_points.camera_path_added_till_first_ba == 0:
    #     object_points.camera_path_added_till_first_ba = len(object_points.camera_params)
    assert len(object_points.camera_indices) == len(object_points.point_indices), "Point inidices and camera indices not equal"
    assert len(object_points.points_2d) == len(object_points.point_indices), "Points 2D and camera indices not equal"

    if True: #internal camera calibration parameters fixed
        #give all points for optimization
        opt_camera_params, opt_pts_3D = bundle_adjustment.do_BA(
                                            object_points.pts_3D,
                                            object_points.camera_params,
                                            object_points.camera_indices,
                                            object_points.point_indices,
                                            object_points.points_2d,
                                            intrinsic_camera_matrix.copy()
                                            )
    else:
        #this never run for now
        opt_camera_params, opt_pts_3D = bundle_adjustment.do_BA(
                                                object_points.pts_3D[object_points.bundle_start:object_points.bundle_stop, :],
                                            object_points.camera_params[object_points.b_camera_params_start:object_points.b_camera_params_stop, :],
                                            # camera_indices,
                                            object_points.point_indices[object_points.ba_point2d_start:, ],
                                            object_points.points_2D[object_points.ba_point2d_start:, :],
                                            )
                
    
    object_points.pts_3D = opt_pts_3D.reshape(-1,3).copy()
    object_points.pts_3D = object_points.pts_3D.tolist()

    object_points.camera_params  = opt_camera_params.reshape(-1,object_points.n_camera_params_ba).copy()
    object_points.camera_params = object_points.camera_params.ravel().tolist()
    
    object_points.points_2d =  (object_points.points_2d).tolist()
    object_points.camera_indices = (object_points.camera_indices).ravel().tolist()
    object_points.point_indices = (object_points.point_indices).ravel().tolist()

