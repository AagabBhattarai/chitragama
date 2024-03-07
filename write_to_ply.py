import numpy as np
from plyfile import PlyData, PlyElement
from utilities import outlier_filtering
from info_track import ObjectPoints

def write_to_ply_file(object_points: ObjectPoints, name):
    pcd_name = name + "point_cloud.ply"
    cam_name = name + "camera_path.ply"
    object_points.pts_3D = np.float32(object_points.pts_3D).reshape(-1,3)
    object_points.pts_3D_color = np.uint8(object_points.pts_3D_color).reshape(-1,3)
    statistical_outlier_filtering_with_whole(object_points)
    
    x,y,z =object_points.pts_3D[:, 0], object_points.pts_3D[:,1], object_points.pts_3D[:, 2]
    r, g, b= object_points.pts_3D_color[:, 0], object_points.pts_3D_color[:, 1], object_points.pts_3D_color[:, 2] 
    pts = list(zip(x,y,z,r,g,b))
    vertex = np.array(pts, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el]).write(pcd_name)
    
    #write camera path to ply file
    object_points.camera_path = np.float32(object_points.camera_path).reshape(-1,3)
    x,y,z =object_points.camera_path[:, 0], object_points.camera_path[:,1], object_points.camera_path[:, 2]
    pts = list(zip(x,y,z))
    vertex = np.array(pts, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el]).write(cam_name)

def statistical_outlier_filtering_with_whole(object_points):
    inliers_mask = outlier_filtering(object_points.pts_3D, 'i')
    object_points.pts_3D = object_points.pts_3D[inliers_mask]
    object_points.pts_3D_color = object_points.pts_3D_color[inliers_mask]
