import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement
from frames import Frames
from bundle_adjustment import Bundle_Adjusment
from utilities import outlier_filtering
class TwoView:
    def __init__(self) -> None:
        self.rgb_img1 = None
        self.rgb_img2 = None
        self.gray_img1  = None
        self.gray_img2 = None
        self.sift = cv.SIFT_create()
        self.bf_matcher = cv.BFMatcher()
        #change how you take matrix K, read it from a file or something
        intrinsic_camera_matrix = [[689.87, 0, 380.17],[0, 691.04, 251.70],[0, 0, 1]]
        #K for GUSTAV
        # intrinsic_camera_matrix = [[2393.952166119461, -3.410605131648481e-13, 932.3821770809047], [0, 2398.118540286656, 628.2649953288065], [0, 0, 1]]
        # intrinsic_camera_matrix = [[2461.016, 0, 1936/2], [0, 2460, 1296/2], [0, 0, 1]]
        # intrinsic_camera_matrix = [[2393.95216, 0, 932.3821], [0, 2393.9521, 628.2649], [0, 0, 1]]
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
        self.bundle_adjustment_time = True
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


    def update_frame_no_value(self, n):
        self.frame_info_handler.frame1_no = n-2
        self.frame_info_handler.frame2_no = n-1
        self.frame_info_handler.frame3_no = n
        
        
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
        
        
    def reset_for_BA(self):
        #3D points
        self.bundle_start = self.bundle_stop - self.n_unique_pts_prev_frame
        #2D points and indices for BA access
        total_2D_points = len(self.frame_info_handler.points_2D)
        self.ba_point2d_start = total_2D_points - (2*self.n_unique_pts_prev_frame)
        #camera params
        # self.update_camera_intrinsic()
        #update 2D points and indices for Repose calculation for overlapp region
        self.update_pts2d_pindex_cindex()
        self.b_camera_params_start +=1
        self.b_camera_params_stop +=1
    
    def update_pts2d_pindex_cindex(self):
        # self.frame_info_handler.point_indices = self.frame_info_handler.point_indices[:self.ba_point2d_start]
        self.frame_info_handler.points_2D = self.frame_info_handler.points_2D[:self.ba_point2d_start]
        self.frame_info_handler.camera_indices = self.frame_info_handler.camera_indices[:self.ba_point2d_start]
        
    def alter_camera_indices_pre_ba(self, indices):
        indices = indices.ravel().tolist()
        uniq_indices = sorted(set(indices))
        mapping = {value: i for i, value in enumerate(uniq_indices)}
        new_indices = [mapping[x] for x in indices]
        new_indices = np.int32(new_indices).ravel()
        return new_indices

    def do_bundle_adjustment(self):
        self.pts_3D = np.float32(self.pts_3D).reshape(-1,3)
        
        self.frame_info_handler.points_2D = np.float32(self.frame_info_handler.points_2D).reshape(-1,2)
        self.frame_info_handler.camera_params = np.float32(self.frame_info_handler.camera_params).reshape(-1,self.n_camera_params_ba)
        self.frame_info_handler.camera_indices = np.int32(self.frame_info_handler.camera_indices).ravel()
        self.frame_info_handler.point_indices = np.int32(self.frame_info_handler.point_indices).ravel()
        
        camera_indices = self.alter_camera_indices_pre_ba(self.frame_info_handler.camera_indices[self.ba_point2d_start:, ])
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
        # if self.fix_calib:
        #     opt_camera_params, opt_pts_3D = self.bundle_adjuster.do_BA(
        #                                         self.pts_3D[self.bundle_start:self.bundle_stop, :],
        #                                         self.frame_info_handler.camera_params[self.b_camera_params_start:self.b_camera_params_stop, :],
        #                                         camera_indices,
        #                                         self.frame_info_handler.point_indices[self.ba_point2d_start:, ],
        #                                         self.frame_info_handler.points_2D[self.ba_point2d_start:, :],
        #                                         self.intrinsic_camera_matrix.copy()
        #                                         )
        else:
            opt_camera_params, opt_pts_3D = self.bundle_adjuster.do_BA(
                                                    self.pts_3D[self.bundle_start:self.bundle_stop, :],
                                                self.frame_info_handler.camera_params[self.b_camera_params_start:self.b_camera_params_stop, :],
                                                camera_indices,
                                                self.frame_info_handler.point_indices[self.ba_point2d_start:, ],
                                                self.frame_info_handler.points_2D[self.ba_point2d_start:, :],
                                                )
                    
        
        self.pts_3D[self.bundle_start: self.bundle_stop] = opt_pts_3D.reshape(-1,3).copy()
        self.pts_3D = self.pts_3D.tolist()

        self.frame_info_handler.camera_params[self.b_camera_params_start: self.b_camera_params_stop,]   = opt_camera_params.reshape(-1,self.n_camera_params_ba).copy()
        self.frame_info_handler.camera_params = self.frame_info_handler.camera_params.ravel().tolist()
        
        self.frame_info_handler.points_2D =  (self.frame_info_handler.points_2D).tolist()
        self.frame_info_handler.camera_indices = (self.frame_info_handler.camera_indices).tolist()
        self.frame_info_handler.point_indices = (self.frame_info_handler.point_indices).ravel().tolist()
    
        self.setup_for_next_BA_iteration()
 
    def setup_for_next_BA_iteration(self):
        #set start for points that will be used for next BA iteration
        #set start for camera params that will be BA for next BA iteration
        self.reset_for_BA()
        self.bundle_adjustment_time = False
        #for recalculatoion of external pose for newly registered image using imporved points
        self.potential_overlapping_object_pts = np.float32(self.pts_3D[self.bundle_start:]).reshape(-1,3)
        
        self.ba_reset = True
        self.register_new_view()

        
         

    
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
        print("No. of unique pts:", len(mask)) 
    

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
        # proj1 is 3*4 matrix that muls 4*4 matrix
        proj2 = (self.transformation_matrix)[0:3, 0:4] 
        # proj2 = (self.transformation_matrix)[0:3, 0:4] 
        # proj3 = np.linalg.inv(self.transformation_matrix)
        
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
        self.set_point_indices()
        #bundle adjustment done after going through two triangulations
        self.bundle_adjustment_time = not self.bundle_adjustment_time
        # self.bundle_adjustment_time = False

    def statistical_outlier_filtering(self, points3d):
        inliers_mask = outlier_filtering(points3d)
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

        #remove outliers from 2D observations as well

        outlier_mask = np.ones(overlapping_object_pts.shape[0], dtype=bool)
        outlier_mask[mask] = False;
        outlier_pts = overlapping_object_pts[outlier_mask]
        # print(outlier_pts)
        if len(outlier_pts) != 0: 
            self.update_known_outlier_pts(outlier_pts)
       
        R = cv.Rodrigues(rvec)[0]
        # self.transformation_matrix = np.eye(4)
        self.transformation_matrix[0:3, 0:3] = R
        self.transformation_matrix[0:3, 3:4] = tvec
        
        #store camera param of frame with higher index or right frame
        
        self.update_points2d(mask)
        if not self.ba_reset:
            # self.store_camera_param()
            self.store_camera_param_fix() #while not changing internal camera parameters
        else: 
            self.ba_reset = False
        
        
    
    def update_known_outlier_pts(self, outlier_pts):
        self.pts_3D = np.float32(self.pts_3D)
        self.pts_3D_color = np.uint8(self.pts_3D_color)
        
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
        # print((outlier_pts))
        
        #use remove the outlier indices from point_indices 
        self.update_point_indices(global_outlier_indices)
        # global_inlier_indices = [x for x in range(length) if x not in global_outlier_indices]
        # unique_outlier_indices = set(global_outlier_indices.tolist())
        assert self.start==self.stop, "Stride variables unmatched"
        self.stop = self.stop - change
        self.start = self.stop
        self.update_bundle_stop()
         
    def update_point_indices(self, global_outlier_indices):
        global_outlier_indices = set(global_outlier_indices)
        # n = sum(1 for j in global_outlier_indices if j < 630)
        self.frame_info_handler.point_indices = [x - sum(1 for j in global_outlier_indices if j < x) 
                                             for x in self.frame_info_handler.point_indices 
                                             if x not in global_outlier_indices]

    
    def update_points2d(self, inlier_mask_index):
        length = len(self.common_pts_index_prev)
        outlier_mask = np.ones(length, dtype=bool)
        outlier_mask[inlier_mask_index] = False
        outlier_index = self.common_pts_index_prev[outlier_mask]
        #now find total inliers using outlier_index ( inlier for overlapp -> outlier for whole -> inlier for whole)
        length_i = len(self.potential_overlaping_img_pts)
        global_inlier_mask = np.ones(length_i, dtype=bool)
        global_inlier_mask[outlier_index] = False
        self.frame_info_handler.frame1 = self.frame_info_handler.frame1[global_inlier_mask].reshape(-1,2)
        self.frame_info_handler.frame2 = self.frame_info_handler.frame2[global_inlier_mask].reshape(-1,2)

        self.add_points2d(self.frame_info_handler.frame1)
        n = len(self.frame_info_handler.frame1)
        self.add_camera_indices(self.frame_info_handler.frame1_no,n)
        
        self.add_points2d(self.frame_info_handler.frame2)
        n = len(self.frame_info_handler.frame2)
        self.add_camera_indices(self.frame_info_handler.frame2_no, n)

        #to be used for BA reset
        self.n_unique_pts_prev_frame = n
        
        if(self.bundle_adjustment_time):
            self.potential_overlaping_img_pts = self.potential_overlaping_img_pts[global_inlier_mask]
            self.common_pts_index_nri = self.common_pts_index_nri[inlier_mask_index]
            self.common_pts_index_prev = self.common_pts_index_prev[inlier_mask_index]
            self.do_bundle_adjustment()
        else: 
            #inlier within overlapp indexes
            inlier_for_overlapp = self.common_pts_index_prev[inlier_mask_index]
            self.add_point_indices(inlier_for_overlapp.ravel().tolist())
            
            self.add_frame(self.overlapping_pts_nri[inlier_mask_index].reshape(-1,2))
            self.add_points2d(self.frame_info_handler.frame3)
            n = len(self.frame_info_handler.frame3)
            self.add_camera_indices(self.frame_info_handler.frame3_no, n)
        
        
    
    
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

        error = cv.norm(original_pts, reproj_pts, normType=cv.NORM_L2)
        show_residual = (original_pts-reproj_pts).ravel()
        # plt.plot(show_residual)
        # plt.show()
        # original_pts = original_pts.ravel()
        # reproj_pts = reproj_pts.ravel()
        
        print(f"Reprojection for newly triangulated points Error: {error}")
        # i = input("wait") 
        
        
    
    
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
    def add_frame(self, frame):
        self.frame_info_handler.frame3 = frame.copy()

    def add_points2d(self,frame):
        self.frame_info_handler.points_2D.extend(frame.tolist())
        
    def set_point_indices(self):
        assert self.start != self.stop, "Stride variables are equal, they should be different"
        point_indices = [(x-self.bundle_start) for x in range(self.start, self.stop)]
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
        inliers_mask = outlier_filtering(self.pts_3D)
        self.pts_3D = self.pts_3D[inliers_mask]
        self.pts_3D_color = self.pts_3D_color[inliers_mask]
        self.pts_3D = self.pts_3D.tolist()
        self.pts_3D_color = self.pts_3D_color.tolist()

    def write_to_ply_file(self):
        self.statistical_outlier_filtering_with_whole()
        
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
    
        