import open3d as o3d
import pyvista as pv
import numpy as np
import tkinter as tk

# colors_file = "room1/defpoint_cloud.ply"
# normal_file = "room1/test_pt_cloud.ply"

# colors_file = "collected things/guire pts color.ply"
# normal_file = "collected things/guire pts.ply"

#colors_file = "collected things/another.ply"
#normal_file = "collected things/gustavpts.ply"
vis = None

#do ball pivoting for now
radii = [0.001, 0.05, 0.1]
pos_depth = 11
density_filter = 0.06
smoothing = 15
pt_cld = None

def slide_depth(val):
    global pos_depth
    pos_depth = int(val)
def slide_density(val):
    global density_filter
    density_filter = val
def slide_smoothing(val):
    global smoothing
    smoothing = int(val)
    
def redraw_poission(butt_state):
    global vis
    print('poission depth = ', pos_depth, 'density filter = ', density_filter, 'smoothing = ' ,smoothing)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pt_cld, depth = pos_depth)
    verts_to_remove = densities < np.quantile(densities, density_filter)
    mesh.remove_vertices_by_mask(verts_to_remove)
    mesh = mesh.filter_smooth_simple(number_of_iterations = smoothing)
    write_file = "temp_mesh.ply"
    o3d.io.write_triangle_mesh(write_file, mesh)
    vis.clear_actors()
    pv_mesh = pv.read(write_file)
    #vis.add_mesh(pv_pt)
    vis.add_mesh(pv_mesh, show_scalar_bar = False, color = '#777777')
    if tk.messagebox.askyesno("", "Save current mesh to a file ?"):
        from tkinter import filedialog as tkfd
        write_file = tkfd.asksaveasfilename(title = "Save Mesh",
                                            defaultextension = ".ply",
                                            filetypes =
                                            [("PLY file", "*.ply")])
        o3d.io.write_triangle_mesh(write_file, mesh)
        
def launch_meshing_window(pt_file_name):
    global vis
    global radii
    global pos_depth
    global density_filter
    global smoothing
    global pt_cld
    #pt_norms = o3d.io.read_point_cloud("collected things/guire pts.ply")
    #pt_cld = o3d.io.read_point_cloud("collected things/guire pts color.ply")
    pt_cld = o3d.io.read_point_cloud(pt_file_name)
    pt_cld.estimate_normals()
    #pv_pt = pv.read("collected things/guire pts color.ply")
    pv_pt = pv.read(pt_file_name)
    vis = pv.Plotter()
    normals = np.asarray(pt_cld.normals)
    pv_pt['vectors'] = normals
    arrows = pv_pt.glyph(
            orient = 'vectors',
            scale = False,
            factor = 0.01,
        )
    vis.add_mesh(pv_pt, show_scalar_bar = False)
    #vis.add_mesh(arrows, color = "lightblue")

    

    depth_slider = vis.add_slider_widget(slide_depth, (2, 20), value = pos_depth, title = "Depth Slider", pointa = (0.1, 0.1), pointb = (0.4, 0.1))
    density_slider = vis.add_slider_widget(slide_density, (0.001, 0.09), value = density_filter, title = "Density Filter Slider", pointa = (0.1, 0.3), pointb = (0.4, 0.3))
    pois_butt = vis.add_checkbox_button_widget(redraw_poission, value = False, position = (1, 50))
    smooth_slider = vis.add_slider_widget(slide_smoothing, (1, 40), value = smoothing, title = "Smoothing Iterations", pointa = (0.1, 0.5), pointb = (0.4, 0.5))    
    #sl1 = vis.add_slider_widget(slide1, (0.001, 0.5), value = 0.001, title = "Radius 1", pointa = (0.1, 0.1), pointb = (0.4, 0.1))
    #sl2 = vis.add_slider_widget(slide2, (0.01, 1), value = 0.05, title = "Radius 2", pointa = (0.1, 0.3), pointb = (0.4, 0.3))
    #sl3 = vis.add_slider_widget(slide3, (0.05, 2), value = 0.1, title = "Radius 3", pointa = (0.1, 0.5), pointb = (0.4, 0.5))
    
    #butt = vis.add_checkbox_button_widget(redraw, value = False, position = (1, 50))
    vis.show()
    vis.close()

# #o3d.visualization.draw_geometries([pcd])
# # alpha = 0.01

# # #mesh = o3d.geometry.TriangleMesh.create_from_
# # mesh.compute_vertex_normals()
# # o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

# # pcd = pcd.voxel_down_sample(voxel_size=0.005)
# # o3d.visualization.draw_geometries([pcd])
# # radii = [0.001,0.05,0.1]
# #mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
# #mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, 0.1)
# # print('run Poisson surface reconstruction')
# # with o3d.utility.VerbosityContextManager(
# #         o3d.utility.VerbosityLevel.Debug) as cm:
# #     mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
# #         pcd, depth=6)
# # print(mesh)
# # o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
# #o3d.visualization.draw_geometries([mesh],
# #                                  zoom=0.664,
# #                                  front=[-0.4761, -0.4698, -0.7434],
# #                                  lookat=[1.8900, 3.2596, 0.9284],
# #                                  up=[0.2304, -0.8825, 0.4101])
