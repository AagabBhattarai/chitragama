import numpy as np
from plyfile import PlyData, PlyElement
from utilities import outlier_filtering
import pyvista
import open3d as o3d
from tkinter import filedialog as tkfd

#functions to estimate normals 
def collect_cameras(object_points ):
    out_list = []
    for inx_3d, pt in enumerate(object_points.pts_3D):
        cam_list = []
        for inx_2d, ref_3d in enumerate(object_points.point_indices):
            if ref_3d == inx_3d :
                cam_inx = object_points.camera_indices[inx_2d]
                cam_obj = object_points.camera_params[cam_inx]
                cam_pos = cam_obj[3:] - pt
                cam_pos = cam_pos / np.linalg.norm(cam_pos)
                cam_list.append(cam_pos)
        out_list.append(cam_list)
    return out_list

def get_averaged_norms(vec_arr_list):
    out_list = []
    for arr in vec_arr_list:
        sum = [0, 0, 0]
        for v in arr:
            sum = sum + v
        sum = sum / np.linalg.norm(sum)
        out_list.append(sum)
    return out_list


def write_to_ply_file(object_points):
    from info_track import ObjectPoints
    #pcd_name = name + "point_cloud.ply"
    #cam_name = name + "camera_path.ply"
    pcd_name = tkfd.asksaveasfilename(title = "Save Sparse Point Cloud As",
                                      defaultextension = ".ply",
                                      filetypes = [("PLY file", "*.ply")])
    cam_name = tkfd.asksaveasfilename(title = "Save Camera Path As",
                                      defaultextension = ".ply",
                                      filetypes = [("PLY file", "*.ply")])
    object_points.pts_3D = np.float32(object_points.pts_3D).reshape(-1,3)
    object_points.pts_3D_color = np.uint8(object_points.pts_3D_color).reshape(-1,3)
    statistical_outlier_filtering_with_whole(object_points)
    
    x,y,z =object_points.pts_3D[:, 0], object_points.pts_3D[:,1], object_points.pts_3D[:, 2]
    r, g, b= object_points.pts_3D_color[:, 0], object_points.pts_3D_color[:, 1], object_points.pts_3D_color[:, 2]

    camera_per_pt = collect_cameras(object_points)
    normals = get_averaged_norms(camera_per_pt)
    normals = np.float32(normals)
    nx, ny, nz = normals[:, 0], normals[:, 1], normals[:, 2]
    
    pts = list(zip(x,y,z,r,g,b,nx, ny, nz))
    vertex = np.array(pts, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')])

    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el]).write(pcd_name)
    
    #write camera path to ply file
    object_points.camera_path = np.float32(object_points.camera_path).reshape(-1,3)
    x,y,z =object_points.camera_path[:, 0], object_points.camera_path[:,1], object_points.camera_path[:, 2]
    pts = list(zip(x,y,z))
    vertex = np.array(pts, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el]).write(cam_name)
    return pcd_name, cam_name

def statistical_outlier_filtering_with_whole(object_points):
    inliers_mask = outlier_filtering(object_points.pts_3D, 'i')
    object_points.pts_3D = object_points.pts_3D[inliers_mask]
    object_points.pts_3D_color = object_points.pts_3D_color[inliers_mask]
    
