from two_view import TwoView
import glob
def main():
    test_directory = "fountain"
    # filepaths = ["fountain/0000.png","fountain/0001.png"]
    filepaths = glob.glob(f"{test_directory}/*.png")
    # filepaths = ['fountain\\0005.png', 'fountain\\0004.png', 'fountain\\0006.png', 'fountain\\0007.png', 'fountain\\0003.png', 'fountain\\0002.png', 'fountain\\0001.png', 'fountain\\0000.png']#, 'fountain\\0008.png', 'fountain\\0009.png', 'fountain\\0010.png']

    print(filepaths)
    sfm = TwoView()
    for im1, im2 in zip(filepaths, filepaths[1:]):
        
        sfm.process_image([im1, im2])
        matches = sfm.find_good_correspondences()
        sfm.find_inlier_points(matches)
        sfm.find_extrinsics_of_camera()
        sfm.find_3D_of_iniliers()
        # sfm.display()

    sfm.start = 0
    sfm.display()
    sfm.write_to_ply_file()

if __name__ == "__main__":
    main()
