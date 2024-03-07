from info_track import ObjectPoints, ImagePair, MetaInfo, ImageView, Preferences
import numpy as np
import cv2 as cv
from utilities import outlier_filtering, display_3d

def model_initialization(Views, Scene_graph, initialization_ids,metainfo:MetaInfo):
    view_1, view_2 = initialization_ids
    img_pair:ImagePair = Scene_graph[view_1-1][view_2]
    assert view_1 == img_pair.view_1 and view_2 == img_pair.view_2, "Assertion Error, Scene graph associations and given index don't match"
    matches = img_pair.matches
    temp = np.eye(4)
    proj1 = temp[:3, :4]
    proj2 = img_pair.projection
    Views[view_1].extrinsic_pose = proj1
    Views[view_2].extrinsic_pose = proj2
    
    object_points = ObjectPoints(metainfo.unique_feature_points) 
    object_points = compute_3D_points(view_1, view_2, matches, Views, object_points, metainfo )
    return object_points

def triangulate_new_points(Views, Scene_graph, initialization_ids, object_points, metainfo):
    initialization_ids = tuple(sorted(initialization_ids, reverse=True))
    view_1, view_2 = initialization_ids
    img_pair:ImagePair = Scene_graph[view_1-1][view_2]
    matches = img_pair.matches
    if len(matches) < 30:
        return
    compute_3D_points(view_1, view_2, matches, Views, object_points, metainfo)
    
def compute_3D_points(view_1, view_2,matches, Views, object_points: ObjectPoints, metainfo:MetaInfo):
    unique_matches = list()
    for m in matches:
        if Views[view_1].global_descriptor[m.queryIdx] not in object_points.pts_3D_global_descriptor_value:
            unique_matches.append(m)
    if len(unique_matches) == 0:
        return object_points  
    points_1 =np.float32([ Views[view_1].keypoints[m.queryIdx].pt for m in unique_matches]).reshape(-1,2)
    points_2 =np.float32([ Views[view_2].keypoints[m.trainIdx].pt for m in unique_matches]).reshape(-1,2)
    camera_intrinsics1 = Views[view_1].K
    camera_intrinsics2 = Views[view_2].K
    p1 = camera_intrinsics1 @ Views[view_1].extrinsic_pose
    p2 = camera_intrinsics2 @ Views[view_2].extrinsic_pose
    
    start = len(object_points.pts_3D)
    recovered_3D_points_in_homogenous = cv.triangulatePoints(p1, p2, points_1.T, points_2.T).T
    triangulated_points =  recovered_3D_points_in_homogenous[:, 0:3]/recovered_3D_points_in_homogenous[:,3:]
    
    global_descriptors_for_triangulated_pts = [Views[view_1].global_descriptor[m.queryIdx] for m in unique_matches]
    stop = start + len(global_descriptors_for_triangulated_pts)
    stat_return= statistical_outlier_filtering(triangulated_points, points_1, points_2, global_descriptors_for_triangulated_pts)
    triangulated_points, points_1, points_2, global_descriptors_for_triangulated_pts = stat_return
    assert start != stop, "Assertion error" 
    if len(global_descriptors_for_triangulated_pts) == 0:
        return object_points
    object_points.pts_3D_global_descriptor[global_descriptors_for_triangulated_pts] = True
    object_points.pts_3D_global_descriptor_value.extend(global_descriptors_for_triangulated_pts)
    
    
    for pts in triangulated_points :
        object_points.pts_3D.append(pts)
    for (x,y) in points_1[:]:
        pixel_color = Views[view_1].bgr_img[int(y),int(x)][::-1]
        object_points.pts_3D_color.append(pixel_color)


    reprojection_error(object_points.pts_3D, points_1, start, stop, Views[view_1].extrinsic_pose, Views[view_1].K, metainfo)
    object_points.pts_3D = np.float32(object_points.pts_3D).reshape(-1,3)
    # display_3d(object_points.pts_3D)
    object_points.pts_3D = object_points.pts_3D.tolist()
    #set point indices for 3D points
    # self.set_point_indices()
    # self.add_points_2d()
    # setup_for_BA()
    return object_points
    
def statistical_outlier_filtering(points3d, pts1, pts2, gdi, preferences: Preferences = Preferences()):
    gdi = np.int32(gdi).ravel()
    if preferences.filtering_l:
        inliers_mask = outlier_filtering(points3d, method='l')
        points3d = points3d[inliers_mask]
        pts1 = pts1[inliers_mask]
        pts2 = pts2[inliers_mask]
        gdi = gdi[inliers_mask]
    
    if preferences.filtering_i:
        inliers_mask = outlier_filtering(points3d, method='i')
        points3d = points3d[inliers_mask]
        pts1 = pts1[inliers_mask]
        pts2 = pts2[inliers_mask]
        gdi = gdi[inliers_mask]
        
    gdi = gdi.tolist()
    return points3d, pts1, pts2, gdi

def setup_for_BA():
    pass

def register_new_view(viewid, Views, object_points:ObjectPoints, feature_track):
    view:ImageView = Views[viewid]
    common_3D_pts = []
    common_2D_pts = []
    
    obj_gd_dict = {gd: pt for gd, pt in zip(object_points.pts_3D_global_descriptor_value, object_points.pts_3D)}
    for i, view_gd in enumerate(view.global_descriptor):
        if view_gd in obj_gd_dict:
            common_3D_pts.append(obj_gd_dict[view_gd])
            common_2D_pts.append(view.keypoints[i].pt)
    
    common_2D_pts = np.float32(common_2D_pts).reshape(-1,2)
    common_3D_pts = np.float32(common_3D_pts).reshape(-1, 3)
    success, rvec, tvec, mask = cv.solvePnPRansac(  common_3D_pts,
                                                    common_2D_pts,
                                                    view.K,
                                                    view.distortion_coefficient,
                                                    cv.SOLVEPNP_EPNP)
    
    
    #Calculate Reprojection error for the overlapping points

    print("PNP")
    print("\nSCENE OVERLAPPING POINTS: ",mask.shape[0],"\n")
    pts_3D = common_3D_pts[mask]
    common_og_pts = common_2D_pts[mask].reshape(-1,2)
    reproj_pts, _ = cv.projectPoints(pts_3D, rvec, tvec, view.K,distCoeffs=None)
    reproj_pts = reproj_pts.reshape(-1,2) 
    # print(f"\n\nREPROJECTION ERROR FOR OVERLAPP: BA RESET: {self.ba_reset}")
    print(f"Reprojected points:\n{reproj_pts[:10,:]}")
    print(f"Original points:\n{common_og_pts[:10,:]}")
    error = cv.norm(common_og_pts, reproj_pts, normType=cv.NORM_L2)/reproj_pts.shape[0]
    print(f"Error: {error}")
    
    R = cv.Rodrigues(rvec)[0]
    # self.transformation_matrix = np.eye(4)
    view.extrinsic_pose[0:3, 0:3] = R
    view.extrinsic_pose[0:3, 3:4] = tvec
    # object_points.pts_3D = object_points.pts_3D.tolist()
    

def reprojection_error(points3d, pt1, start, stop, proj, K, metainfo:MetaInfo, preferences = Preferences() ):
    #takes newly calculated 3D pts and 2D correspondences and calculate reprojection error
    #3D points are assumed to be in Eucledian Space
    original_pts = pt1
    pts_3D = points3d[start: stop]
    pts_3D = np.float32(pts_3D).reshape(-1,3)
    R = proj[0:3, 0:3]
    t = proj[0:3, 3]
    
    rvec, _ = cv.Rodrigues(R)
    reproj_pts, _ = cv.projectPoints(pts_3D, rvec, t, K,distCoeffs=None)
    reproj_pts = reproj_pts.reshape(-1,2) 
    # print(f"Reprojected points:\n{reproj_pts[:10, :]}")
    # print(f"Original points:\n{original_pts[:10, :]}")
    # print(f"\nTransformation Matrix:\n{self.transformation_matrix}")

    error = cv.norm(original_pts, reproj_pts, normType=cv.NORM_L2)/len(pts_3D)
    # show_residual = (original_pts-reproj_pts).ravel()
    # plt.plot(show_residual)
    # plt.show()
    # original_pts = original_pts.ravel()
    # reproj_pts = reproj_pts.ravel()
    
    print(f"Reprojection for newly triangulated points Error: {error}")
    metainfo.error_sum += error
    print(f"ERROR SUM: {metainfo.error_sum}")
    if (metainfo.error_sum > preferences.error_threshold):
        metainfo.bundle_adjustment_time = True
    
    # return self.bundle_adjustment_time
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