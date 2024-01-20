from two_view import TwoView

def main():
    filepaths = ["fountain/0003.png","fountain/0004.png"]
    sfm = TwoView(filepaths)
    matches = sfm.find_good_correspondences()
    sfm.find_inlier_points(matches)
    sfm.find_extrinsics_of_camera()
    sfm.find_3D_of_iniliers()
    sfm.display()
    
if __name__ == "__main__":
    main()
