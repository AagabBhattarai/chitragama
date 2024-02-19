import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement

class TwoView:
    def __init__(self) -> None:
        self.rgb_img1 = None
        self.rgb_img2 = None
        self.gray_img1  = None
        self.gray_img2 = None
        self.sift = cv.SIFT_create()
        self.bf_matcher = cv.BFMatcher()
        #change how you take matrix K, read it from a file or something
        # intrinsic_camera_matrix = [[689.87, 0, 380.17],[0, 691.04, 251.70],[0, 0, 1]]
        #K for GUSTAV
        intrinsic_camera_matrix = [[2393.952166119461, -3.410605131648481e-13, 932.3821770809047], [0, 2398.118540286656, 628.2649953288065], [0, 0, 1]]
        self.distortion_coefficients = np.zeros(4, dtype=np.float32).reshape(1,4)
        self.intrinsic_camera_matrix = np.float32(intrinsic_camera_matrix)
        temp = np.eye(4)
        self.proj1 =self.intrinsic_camera_matrix @ temp[:3,:4]
        self.transformation_matrix = None # this contains R|t for frame being registered
        self.proj1_alt = temp[:3,:4]
        
        self.kp1 = None
        self.kp2 = None
        self.des1 = None
        self.des2 = None
        #Here left == smaller indexed image and right == higher indexed image 
        self.inliers_left = None
        self.inliers_right = None
        self.left = None
        self.right = None
        self.unique_pts_left = None
        self.unique_pts_right = None
        #overlapping pts for newly registered image
        self.overlapping_pts_nri = None
        self.common_pts_index_nri = None
        
        self.pts_3D = []
        self.pts_3D_color = []
        self.camera_path = []   
        self.camera_path.append([0,0,0])
        self.stride = []
        self.start = 0
        self.stop = 0
        
        #store location of keypoints from previously triangulated image pair
        self.potential_overlaping_img_pts = None
        #misnomer since this variable is used for storing overlapping 3D points
        self.potential_overlapping_object_pts = None
    
    def store_for_next_registration(self):
        self.potential_overlaping_img_pts = self.unique_pts_right.copy()
        #pts_3D.append is used so; converting between list to numpy array and then back
        self.pts_3D = np.float32(self.pts_3D).reshape(-1,3)
        self.potential_overlapping_object_pts = self.pts_3D[ self.start: self.stop, :]
        self.pts_3D = self.pts_3D.tolist() 
        #setting for next set of registraion????
        self.start = self.stop 
    
    def find_overlap(self) :
        #so at this stage of the pipeline:
        #self.inliers_left contains correspondance for the view with new registration 
        #   which should also have overlap with self.potential_overlapping_img_pts
        #self.inliers_right contains newly registered image points 
        #Now we find overlapping image points for new registration with object points
        
        #find rows/pt's index where overlap
        # all this because of broadcasting rule where newaxis is added index of that array is given first in the tuple
        #okay after some testing i am still confused so this might be the source of bugs
        # index where duplicated
        if (self.inliers_left.shape[0] > self.potential_overlaping_img_pts.shape[0]):
            common_pts_index_prev, common_pts_index = np.where((self.inliers_left == self.potential_overlaping_img_pts[:, None]).all(-1))
        elif (self.inliers_left.shape[0] < self.potential_overlaping_img_pts.shape[0]):
            common_pts_index, common_pts_index_prev = np.where((self.inliers_left[:, None] == self.potential_overlaping_img_pts).all(-1))
        else:
            common_pts_index_prev, common_pts_index = np.where((self.inliers_left == self.potential_overlaping_img_pts[:, None]).all(-1))
        
        assert common_pts_index.shape[0] == common_pts_index_prev.shape[0], "Index lengths for overlapp don't match"
        #same thing as above but readable
        # common_pts_index = []
        # common_pts_index_prev = []
        # for (pt_index, pt) in enumerate(self.potential_overlaping_img_pts):
        #     match_index = np.where((self.inliers_left == pt).all(axis=1))[0]
        #     if match_index.shape[0] != 0: 
        #         common_pts_index.append(match_index[0])
        #         common_pts_index_prev.append(pt_index)
        
        self.overlapping_pts_nri = self.inliers_right[common_pts_index]
        # self.potential_overlapping_object_pts = self.potential_overlapping_object_pts[common_pts_index_prev]
        #to find unique pts
        mask = np.ones(self.inliers_left.shape[0], dtype=bool)
        mask[common_pts_index] = False
        self.unique_pts_left = self.inliers_left[mask] 
        self.unique_pts_right = self.inliers_right[mask]   
        # a = np.ones(self.inliers_left.shape[0])
        # a = a[mask]
        self.common_pts_index_nri = common_pts_index
        self.common_pts_index_prev = common_pts_index_prev
        

    def process_image(self, filepaths: list[str]):
        print(type(filepaths))
        self.rgb_img1 = cv.imread(filepaths[0], cv.IMREAD_COLOR)
        assert self.rgb_img1 is not None, "Image not found in specified file path"
        
        self.rgb_img2 = cv.imread(filepaths[1], cv.IMREAD_COLOR)
        assert self.rgb_img2 is not None, "Image not found in specified file path"
        # self.rgb_img1 = cv.cvtColor(self.rgb_img1, cv.COLOR_BGR2RGB)
        # self.gray_img1 = cv.cvtColor(self.rgb_img1, cv.COLOR_BGR2GRAY)
        # self.gray_img2 = cv.cvtColor(self.rgb_img2, cv.COLOR_BGR2GRAY)
        self.gray_img1 = cv.imread(filepaths[0], cv.IMREAD_GRAYSCALE)     
        self.gray_img2 = cv.imread(filepaths[1], cv.IMREAD_GRAYSCALE)     

    def find_good_correspondences(self) -> list[cv.DMatch]:
        #find features in each of the images
        self.kp1, self.des1 = self.sift.detectAndCompute(self.gray_img1,None)
        self.kp2, self.des2 = self.sift.detectAndCompute(self.gray_img2,None)
        #find correspondences between the two images
        matches = self.bf_matcher.knnMatch(self.des1, self.des2, k=2)
        #Lowe's test
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)
        
        return good
    
    def find_inlier_points(self, matches: list[cv.DMatch]) -> None:
        assert len(matches) > 10, "Not enough matches between the two images"
        
        matching_points_from_left = np.float32([ self.kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
        matching_points_from_right = np.float32([ self.kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
        
        #here USAC_DEFAULT or USAC_MAGSAC are other flag options to play with
        # M, mask = cv.findHomography(matching_points_from_left, matching_points_from_right, cv.RANSAC,5.0)
        F, mask = cv.findFundamentalMat(matching_points_from_left, matching_points_from_right, cv.FM_RANSAC)
        matching_points_from_left.reshape(matching_points_from_left.size//2,2)
        matching_points_from_right.reshape(matching_points_from_right.size//2,2)
        self.inliers_left = matching_points_from_left[mask.ravel() == 1].reshape(-1,2)
        self.inliers_right = matching_points_from_right[mask.ravel() == 1].reshape(-1,2)
        #maybe throw this variable out/delete these variables
        self.left = matching_points_from_left.reshape(-1,2)
        self.right = matching_points_from_right.reshape(-1,2)

        self.unique_pts_left = self.inliers_left
        self.unique_pts_right = self.inliers_right
        print(len(matching_points_from_left)) 
    
    def register_new_view(self) :
        #Wait, to use pnpRansac we don't have to provide overlapping image points and 3D scene it can figure it out itself?
        #well think so
        #let's test it
        a= self.potential_overlapping_object_pts.shape
        overlapping_object_pts = self.potential_overlapping_object_pts[self.common_pts_index_prev].reshape(-1, 3)
        success, rvec, tvec, mask = cv.solvePnPRansac(overlapping_object_pts,
                                                        self.overlapping_pts_nri,
                                                        self.intrinsic_camera_matrix,
                                                        self.distortion_coefficients,
                                                        cv.SOLVEPNP_EPNP)
        
        #Calculate Reprojection error for the overlapping points
    
        print("PNP")
        pts_3D = overlapping_object_pts[mask]
        common_og_pts = self.overlapping_pts_nri[mask].reshape(-1,2)
        reproj_pts, _ = cv.projectPoints(pts_3D, rvec, tvec, self.intrinsic_camera_matrix,distCoeffs=None)
        reproj_pts = reproj_pts.reshape(-1,2) 
        print(f"Reprojected points:\n{reproj_pts[:10,:]}")
        print(f"Original points:\n{common_og_pts[:10,:]}")
        error = cv.norm(common_og_pts, reproj_pts, normType=cv.NORM_L2)/reproj_pts.shape[0]
        print(f"Error: {error}")

        original_length = overlapping_object_pts.shape[0]
        # outlier_index = [x for x in range(original_length) if x not in mask]
        # print(outlier_index)
        print(mask.shape[0])
        # print(original_length)
        # outlier_pts = overlapping_object_pts[outlier_index]
        
        outlier_mask = np.ones(overlapping_object_pts.shape[0], dtype=bool)
        outlier_mask[mask] = False;
        outlier_pts = overlapping_object_pts[outlier_mask]
        if len(outlier_pts) != 0: 
            self.update_known_outlier_pts(outlier_pts)
        # i = input("wait") 
        #so giving all 3D points doesn't work
        #maybe to run RANSAC you need to have atleast 4 correspondance with any scene point??????
        #so maybe give 3D object points as per some distance threshold
       ################################################## 
        # self.overlapping_pts_nri = self.overlapping_pts_nri.astype('float32')
        # rvec, tvec, success, inliers = cv.solvePnPRansac(self.pts_3D.T,
        #                                                  self.overlapping_pts_nri.T,
        #                                                  self.intrinsic_camera_matrix,
        #                                                  self.distortion_coefficients,
        #                                                  cv.SOLVEPNP_EPNP)
       ################################################## 
        R = cv.Rodrigues(rvec)[0]
        self.transformation_matrix = np.eye(4)
        self.transformation_matrix[0:3, 0:3] = R
        self.transformation_matrix[0:3, 3:4] = tvec
        # self.unique_pts_left = self.inliers_left[mask].tolist()
        # self.unique_pts_right = self.inliers_right[mask].tolist()
    
    def update_known_outlier_pts(self, outlier_pts):
        self.pts_3D = np.float32(self.pts_3D)
        self.pts_3D_color = np.float32(self.pts_3D_color)
        
        print(self.pts_3D.shape)
        print(outlier_pts.shape)
        
        #here global means that this is index of outlier points for whole of object points
        _, global_outlier_indices= np.where((self.pts_3D == outlier_pts[:, None]).all(-1))
        length = self.pts_3D.shape[0]
        global_inlier_mask = np.ones(length, dtype=bool)
        global_inlier_mask[global_outlier_indices] = False
        self.pts_3D = self.pts_3D[global_inlier_mask]
        self.pts_3D_color = self.pts_3D_color[global_inlier_mask]
        change = self.stop - self.pts_3D.shape[0]
        self.pts_3D = self.pts_3D.tolist()
        self.pts_3D_color = self.pts_3D_color.tolist()
        print((outlier_pts))
        
        # global_inlier_indices = [x for x in range(length) if x not in global_outlier_indices]
        # unique_outlier_indices = set(global_outlier_indices.tolist())
        assert self.start==self.stop, "Stride variables unmatched"
        self.stop = self.stop - change
        self.start = self.stop
         
    
    def find_extrinsics_of_camera(self) -> None:
        E, mask = cv.findEssentialMat(self.inliers_left,
                                         self.inliers_right,
                                         self.intrinsic_camera_matrix,
                                         cv.RANSAC, prob=0.999, threshold=1.0)
        j = [x for x in mask if x ==1]
        print(len(j))
        _, R, t, _ =  cv.recoverPose(E,  self.inliers_left,
                                            self.inliers_right,
                                            self.intrinsic_camera_matrix,
                                            mask=mask)
        
        # for (x,y) in self.inliers_left[:]:
        #     pixel_color = self.rgb_img1[int(y),int(x)]
        #     pixel_color = pixel_color[::-1]
        #     self.pts_3D_color.append(pixel_color)
        
        self.transformation_matrix = np.eye(4)
        self.transformation_matrix[0:3, 0:3] = R
        self.transformation_matrix[0:3, 3:4] = t

    def to_camera_coordinate(self, K, point: list[float]) -> list[float]:
        normalized = [  (point[0] - K[0,2]) / K[0,0] ,  (point[1] - K[1,2])/K[1,1] ];
        return normalized
    def find_3D_of_iniliers(self) -> list[list[float]]:
        pts_left_camera_space = list()
        pts_right_camera_space = list()
        for ptl, ptr in zip(self.unique_pts_left, self.unique_pts_right):
            pts_left_camera_space.append(self.to_camera_coordinate(self.intrinsic_camera_matrix, ptl))
            pts_right_camera_space.append(self.to_camera_coordinate(self.intrinsic_camera_matrix, ptr))
        pts_left_camera_space = np.float32(pts_left_camera_space).T
        pts_right_camera_space = np.float32(pts_right_camera_space).T
        
        proj1 = self.proj1_alt 
        # proj1 is 3*4 matrix that muls 4*4 matrix
        proj2 = (self.transformation_matrix)[0:3, 0:4] 
        # proj2 = (self.transformation_matrix)[0:3, 0:4] 
        proj3 = np.linalg.inv(self.transformation_matrix)
        
        recovered_3D_points_in_homogenous = cv.triangulatePoints(proj1, proj2, pts_left_camera_space, pts_right_camera_space).T
        self.proj1_alt = proj2
        self.camera_path.append((-proj2[:3,:3].T @ proj2[:3,3]).tolist())
        self.stop += recovered_3D_points_in_homogenous.shape[0] 
        
        for pts in recovered_3D_points_in_homogenous[:, 0:3]/recovered_3D_points_in_homogenous[:,3:]:
            self.pts_3D.append(pts)
        for (x,y) in self.unique_pts_left[:]:
            pixel_color = self.rgb_img1[int(y),int(x)][::-1]
            self.pts_3D_color.append(pixel_color)
    
    def find_3D_of_iniliers_alt(self) -> None:
        temp: np.ndarray = np.eye(4)
        # x = K * [R|t] X; proj1 = K * [R|t] 
        # proj1 = self.intrinsic_camera_matrix @ temp[:3, :4]
        proj1 = self.proj1
        # x' = K*[R|t] X'; X= X';;;TO DEFINE PROPERLY  
        proj2 = proj1 @ self.transformation_matrix
        
        recovered_3D_points_in_homogenous = cv.triangulatePoints(proj1, proj2, self.inliers_left.T, self.inliers_right.T).T
        recovered_3D_points_in_homogenous = cv.triangulatePoints(proj1, proj2, self.inliers_left.T, self.inliers_right.T).T
        
        self.proj1 = proj2
        self.stop += recovered_3D_points_in_homogenous.shape[0] 
        for pts in recovered_3D_points_in_homogenous[:, 0:3]/recovered_3D_points_in_homogenous[:,3:]:
            self.pts_3D.append(pts)
        
    
    def reprojection_error(self):
        #takes newly calculated 3D pts and 2D correspondences and calculate reprojection error
        #3D points are assumed to be in Eucledian Space
        original_pts = self.unique_pts_right
        pts_3D = self.pts_3D[self.start: self.stop]
        pts_3D = np.float32(pts_3D).reshape(-1,3)
        R = self.transformation_matrix[0:3, 0:3]
        t = self.transformation_matrix[0:3, 3]
        
        rvec, _ = cv.Rodrigues(R)
        reproj_pts, _ = cv.projectPoints(pts_3D, rvec, t, self.intrinsic_camera_matrix,distCoeffs=None)
        reproj_pts = reproj_pts.reshape(-1,2) 
        # print(f"Reprojected points:\n{reproj_pts}")
        # print(f"Original points:\n{original_pts}")
        error = cv.norm(original_pts, reproj_pts, normType=cv.NORM_L2)
        # print(f"Error: {error}")
        # i = input("wait") 
        
        
    
    
    def display(self):
        self.pts_3D = np.float32(self.pts_3D).reshape(-1,3)
        self.camera_path = np.float32(self.camera_path).reshape(-1,3)
        fig = plt.figure(figsize = (10,10))
        ax = plt.axes(projection='3d')
        
        # ax.scatter(self.pts_3D[self.start:self.stop, 0], self.pts_3D[self.start:self.stop,1], self.pts_3D[self.start:self.stop, 2],s= 1, c= self.pts_3D_color[self.start:self.stop])
        ax.scatter(self.pts_3D[self.start:self.stop, 0], self.pts_3D[self.start:self.stop,1], self.pts_3D[self.start:self.stop, 2],s= 10) 
        # ax.scatter(self.camera_path[:,0],self.camera_path[:,1],self.camera_path[:,2])
        ax.set_title('3D Parametric Plot')
        ax.set_xlabel('x', labelpad=20)
        ax.set_ylabel('y', labelpad=20)
        ax.set_zlabel('z', labelpad=20)
        plt.show()
            
        # self.start = self.stop
        # plt.scatter(self.pts_3D[:, 0], self.pts_3D[:,1], marker='.')
        self.pts_3D = self.pts_3D.tolist()
        self.camera_path = self.camera_path.tolist()
        # plt.show()

    def write_to_ply_file(self):
        self.pts_3D = np.float32(self.pts_3D).reshape(-1,3)
        x,y,z =self.pts_3D[:, 0], self.pts_3D[:,1], self.pts_3D[:, 2]
        self.pts_3D_color = np.uint8(self.pts_3D_color).reshape(-1,3)
        r, g, b= self.pts_3D_color[:, 0], self.pts_3D_color[:, 1], self.pts_3D_color[:, 2] 
        pts = list(zip(x,y,z,r,g,b))
        vertex = np.array(pts, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

        el = PlyElement.describe(vertex, 'vertex')
        PlyData([el]).write("point_cloud.ply")
        self.camera_path = np.float32(self.camera_path).reshape(-1,3)
        x,y,z =self.camera_path[:, 0], self.camera_path[:,1], self.camera_path[:, 2]
        pts = list(zip(x,y,z))
        vertex = np.array(pts, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        el = PlyElement.describe(vertex, 'vertex')
        PlyData([el]).write("camera_path.ply")
    
    def get_pts_3D(self) -> np.ndarray:
        return self.pts_3D
        