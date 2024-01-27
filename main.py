from two_view import TwoView
import glob
def main():
    test_directory = "fountain"
    # filepaths = ["fountain/0000.png","fountain/0001.png"]
    filepaths = glob.glob(f"{test_directory}/*.png")
    # filepaths = ['fountain\\0005.png', 'fountain\\0004.png', 'fountain\\0006.png', 'fountain\\0007.png', 'fountain\\0003.png', 'fountain\\0002.png', 'fountain\\0001.png', 'fountain\\0000.png']#, 'fountain\\0008.png', 'fountain\\0009.png', 'fountain\\0010.png']
    print(filepaths)
    
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
    sfm.display()
    sfm.store_for_next_registration()
    
    for im1, im2 in zip(filepaths, filepaths[1:]):
        
        sfm.process_image([im1, im2])
        matches = sfm.find_good_correspondences()
        sfm.find_inlier_points(matches)
        # sfm.find_extrinsics_of_camera()
        #register_new_view find pose of the new view wrt to present object scene
        sfm.find_overlap()
        sfm.register_new_view()
        sfm.find_3D_of_iniliers()
        # sfm.display()
        sfm.store_for_next_registration()

    sfm.start = 0
    sfm.display()
    sfm.write_to_ply_file()

if __name__ == "__main__":
    main()
