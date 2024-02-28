# from two_view import TwoView
# import glob


#import open3d as o3d
from two_view import TwoView
import glob
import pyvista
import numpy as np
sfm = TwoView()

vis = pyvista.Plotter()
items = []
button = None
alpha_val = 0.1
iters_count = 1
make_mesh = False
def redraw_stuff():
    global alpha_val
    pyv_pt_cloud = pyvista.PolyData(sfm.pts_3D)
    vis.clear_actors()
    if make_mesh:
        surf = pyv_pt_cloud.delaunay_2d(alpha = alpha_val)
        vis.add_mesh(surf)
        print('Making mesh')
    else :
        vis.add_points(pyv_pt_cloud, show_scalar_bar = False, scalars = np.uint8(sfm.pts_3D_color).reshape(-1,3), rgb = True)
        print('Not mking mesh')

def mesh_make_func(state):
    global make_mesh
    make_mesh = state
    redraw_stuff()

        
def slider_func(slider_val):
    global alpha_val
    alpha_val = slider_val
    redraw_stuff()
    #print("slider changed", alpha_val)

def iter_slider(iter_val):
    global iters_count
    iters_count = iter_val

def iter_func(button_state):
    global items
    global sfm
    global vis
    global button
    count = iters_count
    while items and (count > 0):
        count = count - 1
        button.GetRepresentation().SetState(False)
        item = items[0]
        print("The item for iteration is ", item)
        i = item[0]
        im1 = item[1]
        im2 = item[2]
        items = items[1:]
        
        print("\n\nITERATION:", i)
        sfm.update_frame_no_value(i+2)
        sfm.process_image([im1, im2])
        matches = sfm.find_good_correspondences()
        sfm.find_inlier_points(matches)
        # sfm.find_extrinsics_of_camera()
        #register_new_view find pose of the new view wrt to present object scene
        sfm.find_overlap()
        sfm.register_new_view()
        sfm.find_3D_of_iniliers()
        sfm.reprojection_error()
        # sfm.display()
        sfm.store_for_next_registration()
        sfm.update_bundle_stop()
    redraw_stuff()
    
        
def run_sfm(filepaths, intrinsic_matrix):
    # global sfm
    global items
    global vis
    global button
    sfm = TwoView()
    sfm.intrinsic_camera_matrix = np.float32(intrinsic_matrix)
    #test_directory = "fountain"
    #test_directory = "GustavIIAdolf"
    #filepaths = glob.glob(f"{test_directory}/*.png")
    #filepaths = glob.glob(f"{test_directory}/*.jpg")
    # sfm.intrinsic_camera_matrix = np.float32(intrinsic_matrix) 
    # filepaths = ['fountain\\0005.png', 'fountain\\0004.png', 'fountain\\0006.png', 'fountain\\0007.png', 'fountain\\0003.png', 'fountain\\0002.png', 'fountain\\0001.png', 'fountain\\0000.png']#, 'fountain\\0008.png', 'fountain\\0009.png', 'fountain\\0010.png']
    #print(filepaths)
    
    #Use 2 views to create a object points
    intial_model = filepaths[:2]
    filepaths.remove(intial_model[0])
    # filepaths.remove(intial_model[1])
    # sfm = TwoView()
    sfm.process_image(intial_model)
    matches = sfm.find_good_correspondences()
    sfm.find_inlier_points(matches)
    sfm.find_extrinsics_of_camera()
    sfm.find_3D_of_iniliers()
    sfm.reprojection_error()
    # sfm.display()
    sfm.store_for_next_registration()


    
    # pcd = o3d.geometry.PointCloud()

    print(items)
    for i, (im1, im2) in enumerate(zip(filepaths, filepaths[1:])):
        items.append([i,im1, im2])
    
    print('Adding key event')
    #vis.add_checkbox_button_widget(mesh_make_func, value = True, position = (10,60))
    button = vis.add_checkbox_button_widget(iter_func, value = False, position = (1,50))
    #slider = vis.add_slider_widget(slider_func, (0.1,2), value = 0.5, title = "Delauny Alpha Value", pointa = (0, 1), pointb = (0.3, 1))

    slider2 = vis.add_slider_widget(iter_slider, (1, 20), value = 1, title = "Images to Process",
                                    pointa = (0.1, 0.1), pointb = (0.4, 0.1))
    #vis.add_key_event('s', iter_func)
    #vis.register_key_callback(32, iter_func)

    #da_app = o3d.visualization.gui.Application()
    #da_app.add_window(vis);
    #dat_button = o3d.visualization.gui.Button("Da Button")
    #da_button = vis.gui.Button("Da Button")
    print(vis.actors)
    vis.show()
    print('Closing')
    vis.close()
    #vis.destroy_window()
    
    
    sfm.start = 0
    #sfm.display()
    sfm.write_to_ply_file()
    #pts = sfm.get_pts_3D()
    #print(pts)

def main():
    test_directory = "fountain"
    test_directory_b = "GustavIIAdolf"
    # test_directory = "guerre"
    # test_directory = "eglise"
    # test_directory = "nikolaiI"   
    fountain_path = glob.glob(f"{test_directory}/*.png")
    gustav_path = glob.glob(f"{test_directory_b}/*.jpg")
    fountain_mat = [[620.87, 0, 380.17],[0, 691.04, 251.70],[0, 0, 1]]
    gustav_mat = [[2393.952166119461, -3.410605131648481e-13, 932.3821770809047], [0, 2398.118540286656, 628.2649953288065], [0, 0, 1]]
    
    # intrinsic_camera_matrix = [[2461.016, 0, 1936/2], [0, 2460, 1296/2], [0, 0, 1]] # camera matrix from sensor size database
    # run_sfm(gustav_path, gustav_mat)
    run_sfm(fountain_path, fountain_mat)

    
if __name__ == "__main__":
    main()