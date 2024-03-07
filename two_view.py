import glob
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement
from frames import Frames
from bundle_adjustment import Bundle_Adjusment
from utilities import outlier_filtering
from intrinsic_matrix import compute_intrinsic_matrix
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
        # intrinsic_camera_matrix = [[2393.952166119461, -3.410605131648481e-13, 932.3821770809047], [0, 2398.118540286656, 628.2649953288065], [0, 0, 1]]
        self.test_directory = "GustavIIAdolf"
        # self.test_directory = "nikolaiI"
        # self.test_directory = "guerre"
        # self.test_directory = "eglise"
        # self.test_directory = "room1"
        self.filepaths = glob.glob(f"{self.test_directory}/*.jpg")
        self.database_path = "sensor_width_camera_database.txt"
        intrinsic_camera_matrix = compute_intrinsic_matrix(self.filepaths[0], self.database_path)
        # self.test_directory = "chair"
        # self.filepaths = glob.glob(f"{self.test_directory}/*.jpg")
        # intrinsic_camera_matrix = [[3123.392, 0, 4080/2],[0, 3123.118, 3060/2],[0, 0, 1]]
        self.distortion_coefficients = np.zeros(4, dtype=np.float32).reshape(1,4)
        self.intrinsic_camera_matrix = np.float32(intrinsic_camera_matrix)
        temp = np.eye(4)
        self.proj1 =self.intrinsic_camera_matrix @ temp[:3,:4]
        self.transformation_matrix = np.eye(4) # this contains R|t for frame being registered
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
        
        self.frame_info_handler:Frames = Frames()
        self.frame_info_handler.frame1_no = 0
        self.frame_info_handler.frame2_no = 0
        # self.store_camera_param()
        self.store_camera_param_fix()

        
        self.bundle_adjuster = Bundle_Adjusment()
        self.bundle_start = 0
        self.bundle_stop = 0 #equates to no. of 3D observations
        self.bundle_adjustment_time = False
        # self.bundle_adjustment_time = False
        self.b_camera_params_start = 0 #bundle camera start
        self.b_camera_params_stop = 3 # value 3 because only 3 cameras are used for BA
        self.n_unique_pts_prev_frame = 0
        self.ba_point2d_start = 0
        # self.ba_point2d_stop = 0
        self.bundle_size = 3
        self.ba_reset = False
        self.n_camera_params_ba = 6
        self.fix_calib = True
        self.offset_for_adding_point_indices= 0 #because 0 3D points have been triangulated till now
        self.error_sum =0
        self.camera_path_added_till_first_ba =0

    def get_filepaths(self):
        return self.filepaths
    
    def update_frame_no_value(self, n):
        self.frame_info_handler.frame1_no = n-1
        self.frame_info_handler.frame2_no = n
        
        
    def update_bundle_stop(self):
        self.bundle_stop = self.stop
    
    def update_camera_intrinsic(self):
        #maybe as this is monocular sfm but here we have BA as if we have multiple cameras
        #so here lazily average value of from each optimization is taken
        fx = 0
        fy = 0
        cx = 0
        cy = 0
        self.frame_info_handler.camera_params = np.float32(self.frame_info_handler.camera_params).reshape(-1,10)
        intrinsic_param_sum  = np.sum(self.frame_info_handler.camera_params[:, 6:], axis=0)
        new_intrinsic_params = (intrinsic_param_sum/self.bundle_size).copy()
        self.intrinsic_camera_matrix[0][0] = new_intrinsic_params[0]
        self.intrinsic_camera_matrix[0][2] = new_intrinsic_params[1]
        self.intrinsic_camera_matrix[1][1] = new_intrinsic_params[2]
        self.intrinsic_camera_matrix[1][2] = new_intrinsic_params[3]
        self.frame_info_handler.camera_params = self.frame_info_handler.camera_params.ravel().tolist()
        
        
    def do_bundle_adjustment(self):
        self.pts_3D = np.float32(self.pts_3D).reshape(-1,3)
        
        self.frame_info_handler.points_2D = np.float32(self.frame_info_handler.points_2D).reshape(-1,2)
        self.frame_info_handler.camera_params = np.float32(self.frame_info_handler.camera_params).reshape(-1,self.n_camera_params_ba)
        self.frame_info_handler.camera_indices = np.int32(self.frame_info_handler.camera_indices).ravel()
        self.frame_info_handler.point_indices = np.int32(self.frame_info_handler.point_indices).ravel()
        
        if self.camera_path_added_till_first_ba == 0:
            self.camera_path_added_till_first_ba = len(self.frame_info_handler.camera_params)
        assert len(self.frame_info_handler.camera_indices) == len(self.frame_info_handler.point_indices), "Point inidices and camera indices not equal"
        assert len(self.frame_info_handler.points_2D) == len(self.frame_info_handler.point_indices), "Points 2D and camera indices not equal"

        if self.fix_calib:
            #give all points for optimization
            opt_camera_params, opt_pts_3D = self.bundle_adjuster.do_BA(
                                                self.pts_3D,
                                                self.frame_info_handler.camera_params,
                                                self.frame_info_handler.camera_indices,
                                                self.frame_info_handler.point_indices,
                                                self.frame_info_handler.points_2D,
                                                self.intrinsic_camera_matrix.copy()
                                                )
        else:
            #this never run for now
            opt_camera_params, opt_pts_3D = self.bundle_adjuster.do_BA(
                                                    self.pts_3D[self.bundle_start:self.bundle_stop, :],
                                                self.frame_info_handler.camera_params[self.b_camera_params_start:self.b_camera_params_stop, :],
                                                # camera_indices,
                                                self.frame_info_handler.point_indices[self.ba_point2d_start:, ],
                                                self.frame_info_handler.points_2D[self.ba_point2d_start:, :],
                                                )
                    
        
        self.pts_3D = opt_pts_3D.reshape(-1,3).copy()
        self.pts_3D = self.pts_3D.tolist()

        self.frame_info_handler.camera_params  = opt_camera_params.reshape(-1,self.n_camera_params_ba).copy()
        self.frame_info_handler.camera_params = self.frame_info_handler.camera_params.ravel().tolist()
        
        self.frame_info_handler.points_2D =  (self.frame_info_handler.points_2D).tolist()
        self.frame_info_handler.camera_indices = (self.frame_info_handler.camera_indices).tolist()
        self.frame_info_handler.point_indices = (self.frame_info_handler.point_indices).ravel().tolist()
    
        self.setup_for_next_BA_iteration()
 
    def setup_for_next_BA_iteration(self):
        self.bundle_adjustment_time = False
        self.error_sum = 0
    
    def store_for_next_registration(self):
        self.potential_overlaping_img_pts = self.unique_pts_right.copy()
        #pts_3D.append is used so; converting between list to numpy array and then back
        self.pts_3D = np.float32(self.pts_3D).reshape(-1,3)
        self.potential_overlapping_object_pts = self.pts_3D[ self.start: self.stop, :]
        self.pts_3D = self.pts_3D.tolist() 
        #setting for next set of registraion????
        self.start = self.stop 
        # #store camera param of frame with higher index or right frame
        # self.store_camera_param()
        #store both frames/i.e feature points are stored
        self.store_points2d()
    
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
        
        self.overlapping_pts_nri = self.inliers_right

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
            if m.distance < 0.5*n.distance:
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
    

    def find_extrinsics_of_camera(self) -> None:
        E, mask = cv.findEssentialMat(self.inliers_left,
                                         self.inliers_right,
                                         self.intrinsic_camera_matrix,
                                         cv.RANSAC, prob=0.999, threshold=1.0)
        j = [x for x in mask if x ==1]
        print(len(j))
        # self.unique_pts_left = self.unique_pts_left[mask.ravel() == 1].reshape(-1,2)
        # self.unique_pts_right = self.unique_pts_right[mask.ravel() == 1 ].reshape(-1,2)
        _, R, t, fmask =  cv.recoverPose(E,  self.inliers_left,
                                            self.inliers_right,
                                            self.intrinsic_camera_matrix,
                                            mask=mask)
        
        self.unique_pts_left = self.unique_pts_left[fmask.ravel() == 1].reshape(-1,2)
        self.unique_pts_right = self.unique_pts_right[fmask.ravel() == 1 ].reshape(-1,2)
        print("No. of unique pts:", len(self.unique_pts_left)) 
        
        # self.transformation_matrix = np.eye(4)
        self.transformation_matrix[0:3, 0:3] = R
        self.transformation_matrix[0:3, 3:4] = t
        #store camera param of frame with higher index or right frame
        # self.store_camera_param()
        self.store_camera_param_fix()

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
        proj2 = (self.transformation_matrix)[0:3, 0:4] 
        
        recovered_3D_points_in_homogenous = cv.triangulatePoints(proj1, proj2, pts_left_camera_space, pts_right_camera_space).T
        self.proj1_alt = proj2.copy()
        self.camera_path.append((-proj2[:3,:3].T @ proj2[:3,3]).tolist())
        
        triangulated_points =  recovered_3D_points_in_homogenous[:, 0:3]/recovered_3D_points_in_homogenous[:,3:]
        triangulated_points= self.statistical_outlier_filtering(triangulated_points)
        self.stop += triangulated_points.shape[0] 
        
        for pts in triangulated_points :
            self.pts_3D.append(pts)
        for (x,y) in self.unique_pts_left[:]:
            pixel_color = self.rgb_img1[int(y),int(x)][::-1]
            self.pts_3D_color.append(pixel_color)
        
        #set point indices for 3D points
        # self.set_point_indices()
        # self.add_points_2d()
        self.setup_for_BA()

    def statistical_outlier_filtering(self, points3d):
        inliers_mask = outlier_filtering(points3d, method='l')
        points3d = points3d[inliers_mask]
        self.unique_pts_left = self.unique_pts_left[inliers_mask]
        self.unique_pts_right = self.unique_pts_right[inliers_mask]
        
        inliers_mask = outlier_filtering(points3d, method='i')
        points3d = points3d[inliers_mask]
        self.unique_pts_left = self.unique_pts_left[inliers_mask]
        self.unique_pts_right = self.unique_pts_right[inliers_mask]
        return points3d
        
        
    def register_new_view(self) :
        #Wait, to use pnpRansac we don't have to provide overlapping image points and 3D scene it can figure it out itself?
        #well think so
        #let's test it
        # a= self.potential_overlapping_object_pts.shape
        self.overlapping_pts_nri = self.inliers_right[self.common_pts_index_nri]
        overlapping_object_pts = self.potential_overlapping_object_pts[self.common_pts_index_prev].reshape(-1, 3)
        success, rvec, tvec, mask = cv.solvePnPRansac(overlapping_object_pts,
                                                        self.overlapping_pts_nri,
                                                        self.intrinsic_camera_matrix,
                                                        self.distortion_coefficients,
                                                        cv.SOLVEPNP_EPNP)
        
        #Calculate Reprojection error for the overlapping points
    
        
        print("PNP")
        print("\nSCENE OVERLAPPING POINTS: ",mask.shape[0],"\n")
        pts_3D = overlapping_object_pts[mask]
        common_og_pts = self.overlapping_pts_nri[mask].reshape(-1,2)
        reproj_pts, _ = cv.projectPoints(pts_3D, rvec, tvec, self.intrinsic_camera_matrix,distCoeffs=None)
        reproj_pts = reproj_pts.reshape(-1,2) 
        print(f"\n\nREPROJECTION ERROR FOR OVERLAPP: BA RESET: {self.ba_reset}")
        print(f"Reprojected points:\n{reproj_pts[:10,:]}")
        print(f"Original points:\n{common_og_pts[:10,:]}")
        error = cv.norm(common_og_pts, reproj_pts, normType=cv.NORM_L2)/reproj_pts.shape[0]
        print(f"Error: {error}")
        
        R = cv.Rodrigues(rvec)[0]
        # self.transformation_matrix = np.eye(4)
        self.transformation_matrix[0:3, 0:3] = R
        self.transformation_matrix[0:3, 3:4] = tvec
        
        self.update_points2d()
        self.store_camera_param_fix() #while not changing internal camera parameters
        
        
         
    def update_points2d(self):
            inlier_for_overlapp = self.common_pts_index_prev
            point_indices= inlier_for_overlapp.ravel() + self.offset_for_adding_point_indices
            self.add_point_indices(point_indices)
            self.offset_for_adding_point_indices += self.n_unique_pts_prev_frame #so from now on  for next add point indices, N + [2, 3, 7,...] will happen
            
            self.add_frame(self.overlapping_pts_nri.reshape(-1,2))
            self.add_points2d(self.frame_info_handler.frame2)
            n = len(self.frame_info_handler.frame2)
            self.add_camera_indices(self.frame_info_handler.frame2_no, n)
    
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
        # print(f"Reprojected points:\n{reproj_pts[:10, :]}")
        # print(f"Original points:\n{original_pts[:10, :]}")
        # print(f"\nTransformation Matrix:\n{self.transformation_matrix}")

        error = cv.norm(original_pts, reproj_pts, normType=cv.NORM_L2)/len(pts_3D)
        # show_residual = (original_pts-reproj_pts).ravel()
        # plt.plot(show_residual)
        # plt.show()
        # original_pts = original_pts.ravel()
        # reproj_pts = reproj_pts.ravel()
        
        print(f"Reprojection for newly triangulated points Error: {error}")
        self.error_sum += error
        print(f"ERROR SUM: {self.error_sum}")
        if (self.error_sum > 0.5):
            self.bundle_adjustment_time = True
        
        return self.bundle_adjustment_time
        
    
    def initial_frame_no_setup(self):
        self.frame_info_handler.frame1_no = 0
        self.frame_info_handler.frame2_no = 1
    def setup_for_BA(self):
        self.set_points_2d()
        self.set_point_indices()
        
    def store_camera_param(self):
        R = self.transformation_matrix[ :3, :3]
        rvec, _ = cv.Rodrigues(R)
        t = self.transformation_matrix[:3, 3]

        intrinsics = [self.intrinsic_camera_matrix[0][0], self.intrinsic_camera_matrix[0][2]
                    , self.intrinsic_camera_matrix[1][1],self.intrinsic_camera_matrix[1][2]]
        
        self.frame_info_handler.camera_params.extend(rvec.ravel().tolist())
        self.frame_info_handler.camera_params.extend(t.tolist())  
        self.frame_info_handler.camera_params.extend(intrinsics) 
    
    def store_camera_param_fix(self):
        R = self.transformation_matrix[ :3, :3]
        rvec, _ = cv.Rodrigues(R)
        t = self.transformation_matrix[:3, 3]

        # intrinsics = [self.intrinsic_camera_matrix[0][0], self.intrinsic_camera_matrix[0][2]
        #             , self.intrinsic_camera_matrix[1][1],self.intrinsic_camera_matrix[1][2]]
        
        self.frame_info_handler.camera_params.extend(rvec.ravel().tolist())
        self.frame_info_handler.camera_params.extend(t.tolist())  
        # self.frame_info_handler.camera_params.extend(intrinsics) 
    
    
    def store_points2d(self):
        self.frame_info_handler.frame1 = self.unique_pts_left.copy()
        self.frame_info_handler.frame2 = self.unique_pts_right.copy()
    def set_points_2d(self):
        self.store_points2d()
        self.add_points2d(self.frame_info_handler.frame1)
        n = len(self.frame_info_handler.frame1)
        self.add_camera_indices(self.frame_info_handler.frame1_no,n)
        
        self.add_points2d(self.frame_info_handler.frame2)
        n = len(self.frame_info_handler.frame2)
        self.add_camera_indices(self.frame_info_handler.frame2_no, n)
        self.n_unique_pts_prev_frame = n
            

    def add_frame(self, frame):
        self.frame_info_handler.frame2 = frame.copy()

    def add_points2d(self,frame):
        self.frame_info_handler.points_2D.extend(frame.tolist())
        
    def set_point_indices(self):
        assert self.start != self.stop, "Stride variables are equal, they should be different"
        point_indices = [(x) for x in range(self.start, self.stop)]
        self.frame_info_handler.point_indices.extend(point_indices)
        self.frame_info_handler.point_indices.extend(point_indices)
    
    def add_point_indices(self, index):
        self.frame_info_handler.point_indices.extend(index)
    
    def add_camera_indices(self, index, n):
        self.frame_info_handler.camera_indices.extend([index] * n)

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


    def statistical_outlier_filtering_with_whole(self):
        self.pts_3D_color = np.uint8(self.pts_3D_color).reshape(-1,3)
        self.pts_3D = np.float32(self.pts_3D).reshape(-1,3)
        inliers_mask = outlier_filtering(self.pts_3D, 'i')
        self.pts_3D = self.pts_3D[inliers_mask]
        self.pts_3D_color = self.pts_3D_color[inliers_mask]
        self.pts_3D = self.pts_3D.tolist()
        self.pts_3D_color = self.pts_3D_color.tolist()

    def write_to_ply_file(self, name):
        pcd_name = name + "point_cloud.ply"
        cam_name = name + "camera_path.ply"
        self.statistical_outlier_filtering_with_whole()
        
        self.pts_3D = np.float32(self.pts_3D).reshape(-1,3)
        x,y,z =self.pts_3D[:, 0], self.pts_3D[:,1], self.pts_3D[:, 2]
        self.pts_3D_color = np.uint8(self.pts_3D_color).reshape(-1,3)
        r, g, b= self.pts_3D_color[:, 0], self.pts_3D_color[:, 1], self.pts_3D_color[:, 2] 
        pts = list(zip(x,y,z,r,g,b))
        vertex = np.array(pts, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

        el = PlyElement.describe(vertex, 'vertex')
        PlyData([el]).write(pcd_name)
        self.camera_path = np.float32(self.camera_path).reshape(-1,3)
        
        x,y,z =self.camera_path[:, 0], self.camera_path[:,1], self.camera_path[:, 2]
        pts = list(zip(x,y,z))
        vertex = np.array(pts, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        el = PlyElement.describe(vertex, 'vertex')
        PlyData([el]).write(cam_name)
    
    def get_pts_3D(self) -> np.ndarray:
        return self.pts_3D
    
    
    def update_camera_path(self):
        self.frame_info_handler.camera_params = np.float32(self.frame_info_handler.camera_params).reshape(-1, self.n_camera_params_ba)
        for i in range(self.camera_path_added_till_first_ba):
            rvec = self.frame_info_handler.camera_params[i, :3]
            tvec = self.frame_info_handler.camera_params[i, 3:6]
            rmat, _ = cv.Rodrigues(rvec)
            cam_path = (-rmat.T @ tvec).tolist()
            self.camera_path[i] = cam_path
        