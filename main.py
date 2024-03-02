from two_view import TwoView
import glob

def main():
    # test_directory = "fountain"
    # test_directory = "GustavIIAdolf"
    test_directory = "guerre"
    # test_directory = "eglise"
    # test_directory = "nikolaiI"
    # filepaths = glob.glob(f"{test_directory}/*.png")
    filepaths = glob.glob(f"{test_directory}/*.jpg")
    
    # filepaths = ['fountain\\0005.png', 'fountain\\0004.png', 'fountain\\0006.png', 'fountain\\0007.png', 'fountain\\0003.png', 'fountain\\0002.png', 'fountain\\0001.png', 'fountain\\0000.png']#, 'fountain\\0008.png', 'fountain\\0009.png', 'fountain\\0010.png']
    print(filepaths)
    
    # filepaths = filepaths[16:]
    # filepaths = filepaths[:20]
    
    #Use 2 views to create a object points
    intial_model = filepaths[:2]
    filepaths.remove(intial_model[0])
    # filepaths.remove(intial_model[1])
    sfm = TwoView()
    sfm.process_image(intial_model)
    matches = sfm.find_good_correspondences()
    sfm.find_inlier_points(matches)
    sfm.find_extrinsics_of_camera()
    sfm.find_3D_of_iniliers()
    sfm.reprojection_error()
    # sfm.display()
    sfm.store_for_next_registration()
    
    for i, (im1, im2) in enumerate(zip(filepaths, filepaths[1:])):
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

    sfm.start = 0
    sfm.display()


    sfm.update_camera_path()
    sfm.write_to_ply_file()
    pts = sfm.get_pts_3D()
    print(pts)

if __name__ == "__main__":
    main()

