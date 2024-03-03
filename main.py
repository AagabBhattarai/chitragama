from two_view import TwoView
import sys
import glob

def main():

    arguments = sys.argv[1:]
    if len(arguments) == 0:
        arg = 'def'
    else: 
        arg = arguments[0]
    
    sfm = TwoView()
    filepaths = sfm.get_filepaths()
    print(filepaths)

    #Use 2 views to create a object points
    intial_model = filepaths[:2]
    filepaths.remove(intial_model[0])
    sfm.process_image(intial_model)
    sfm.initial_frame_no_setup()
    matches = sfm.find_good_correspondences()
    sfm.find_inlier_points(matches)
    sfm.find_extrinsics_of_camera()
    sfm.find_3D_of_iniliers()
    sfm.reprojection_error()
    sfm.display()
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
        ba_required = sfm.reprojection_error()
        if ba_required:
            sfm.do_bundle_adjustment()
        # sfm.display()
        sfm.store_for_next_registration()
        sfm.update_bundle_stop()

    sfm.start = 0
    sfm.display()


    sfm.update_camera_path()
    sfm.write_to_ply_file(arg)
    pts = sfm.get_pts_3D()
    print(pts)

if __name__ == "__main__":
    main()

