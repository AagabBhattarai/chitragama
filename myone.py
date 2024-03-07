
from two_view import TwoView
import glob
import pyvista
import numpy as np

#So, 1 image requires, gray image, and a list of sift features
#Make a list of such for every available images
#Now for each image pair in cosecutive order only
#Compue common good correspondences, find the whole fundamental shinanegans
#And compute the 3d points, and find inliers
#Now find the indexes of such inliers in pairs for each two images
#For that pair, those are matches
#Also for those matches, store an array of 3D points
#Also store a relative camera for the pair
#So each entry contains : pair of matching inxes of sift features
#                         for each of those, a corresponding 3D point
#                         a relative camera matrix
#
#Now collection of points:
#For each point, starting from first pair
#Keep collecting the 3D points in chain, and average them, push in new array
#then pop each from original images
#keep on till none is left


        
def main():
    #test_directory = "fountain"
    test_directory = "GustavIIAdolf"
    #filepaths = glob.glob(f"{test_directory}/*.png")
    filepaths = glob.glob(f"{test_directory}/*.jpg")

    sfms = []
    for i, (im1, im2) in enumerate(zip(filepaths, filepaths[1:-2])):
        sfm = TwoView()
        #Use 2 views to create a object points
        intial_model = [im1, im2]

        sfm.process_image(intial_model)
        matches = sfm.find_good_correspondences()
        sfm.find_inlier_points(matches)
        sfm.find_extrinsics_of_camera()
        sfm.find_3D_of_iniliers()
        sfm.reprojection_error()
        sfm.store_for_next_registration()
        sfms.append(sfm)




    #Now, just display combined one
    vis = pyvista.Plotter()
    mat = np.eye(4,4)
    print(mat)
    gpts = np.array([])
    for sfm in sfms:
        pts = np.array(sfm.pts_3D)
        #print(pts)
        pts = np.insert(pts, 3, 1, axis=1)
        #print(pts)        
        tpts = np.matmul( mat[0:3:1, 0:4:1] , pts.transpose())
        #print(tpts)
        #break
        #pt_cloud = pyvista.PolyData(tpts.transpose())
        gpts = np.append(gpts,tpts.transpose())
        mat = np.matmul(mat, np.linalg.inv(sfm.transformation_matrix))
        #pt_cloud = pyvista.PolyData(sfm.pts_3D)
        # pt_cloud.plot(style = 'points_gaussian')
        # vis.add_points(pyvista.PolyData(sfm.pts_3D),
        #                style='points_gaussian')

    ptcld = pyvista.PolyData(gpts)
    vis.add_points(ptcld, style='points_gaussian')
    vis.show()
    vis.close()

#if __name__ == "__main__":
main()
