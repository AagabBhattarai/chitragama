import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

class TwoView:
    def __init__(self, filepaths: list[str]) -> None:
        self.rgb_img1 = cv.imread(filepaths[0], cv.IMREAD_COLOR)
        assert self.rgb_img1 is not None, "Image not found in specified file path"
        self.rgb_img2 = cv.imread(filepaths[1], cv.IMREAD_COLOR)
        assert self.rgb_img2 is not None, "Image not found in specified file path"
        self.rgb_img1 = cv.cvtColor(self.rgb_img1, cv.COLOR_BGR2RGB)
        # self.gray_img1 = cv.cvtColor(self.rgb_img1, cv.COLOR_BGR2GRAY)
        # self.gray_img2 = cv.cvtColor(self.rgb_img2, cv.COLOR_BGR2GRAY)
        self.gray_img1 = cv.imread(filepaths[0], cv.IMREAD_GRAYSCALE)     
        self.gray_img2 = cv.imread(filepaths[1], cv.IMREAD_GRAYSCALE)     
        self.sift = cv.SIFT_create()
        self.bf_matcher = cv.BFMatcher()
        #change how you take matrix K, read it from a file or something
        intrinsic_camera_matrix = [[689.87, 0, 380.17],[0, 691.04, 251.70],[0, 0, 1]]
        self.intrinsic_camera_matrix = np.float32(intrinsic_camera_matrix)
        self.transfomation_matrix = None

        self.kp1 = None
        self.kp2 = None
        self.des1 = None
        self.des2 = None
        self.inliers_left = None
        self.inliers_right = None
        
        self.pts_3D = None
        self.pts_3D_color = list()
    def find_good_correspondences(self) -> list[cv.DMatch]:
        #find features in each of the images
        self.kp1, self.des1 = self.sift.detectAndCompute(self.gray_img1,None)
        self.kp2, self.des2 = self.sift.detectAndCompute(self.gray_img2,None)
        #find correspondences between the two images
        matches = self.bf_matcher.knnMatch(self.des1, self.des2, k=2)
        #Lowe's test
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append(m)
        
        return good
    
    def find_inlier_points(self, matches: list[cv.DMatch]) -> None:
        assert len(matches) > 10, "Not enough matches between the two images"
        
        matching_points_from_left = np.float32([ self.kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
        matching_points_from_right = np.float32([ self.kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
        
        #here USAC_DEFAULT or USAC_MAGSAC are other flag options to play with
        M, mask = cv.findHomography(matching_points_from_left, matching_points_from_right, cv.RANSAC,5.0)
        matching_points_from_left.reshape(matching_points_from_left.size//2,2)
        matching_points_from_right.reshape(matching_points_from_right.size//2,2)
        self.inliers_left = matching_points_from_left[mask.ravel() == 1].reshape(-1,2)
        self.inliers_right = matching_points_from_right[mask.ravel() == 1].reshape(-1,2)
        print(len(matching_points_from_left)) 
    
    def find_extrinsics_of_camera(self) -> None:
        E, mask = cv.findEssentialMat(self.inliers_left,
                                         self.inliers_right,
                                         self.intrinsic_camera_matrix,
                                         cv.RANSAC, prob=0.999, threshold=1.0)

        _, R, t, _ =  cv.recoverPose(E,  self.inliers_left,
                                            self.inliers_right,
                                            self.intrinsic_camera_matrix,
                                            mask=mask)
        
        for (x,y) in self.inliers_left[:]:
            pixel_color = self.rgb_img1[int(y),int(x)]
            pixel_color = list(map(lambda x: float(x)/255, pixel_color))
            self.pts_3D_color.append(pixel_color)
        
        self.transfomation_matrix = np.eye(4)
        self.transfomation_matrix[0:3, 0:3] = R
        self.transfomation_matrix[0:3, 3:4] = t

    def to_camera_coordinate(self, K, point: list[float]) -> list[float]:
        normalized = [  (point[0] - K[0,2]) / K[0,0] ,  (point[1] - K[1,2])/K[1,1] ];
        return normalized
    def find_3D_of_iniliers(self):
        pts_left_camera_space = list()
        pts_right_camera_space = list()
        for ptl, ptr in zip(self.inliers_left, self.inliers_right):
            pts_left_camera_space.append(self.to_camera_coordinate(self.intrinsic_camera_matrix, ptl))
            pts_right_camera_space.append(self.to_camera_coordinate(self.intrinsic_camera_matrix, ptr))
        pts_left_camera_space = np.float32(pts_left_camera_space).T
        pts_right_camera_space = np.float32(pts_right_camera_space).T
        
        temp = np.eye(4)
        proj1 = temp[0:3, 0:4]
        proj2 = (self.transfomation_matrix)[0:3, 0:4] 
        
        recovered_3D_points_in_homogenous = cv.triangulatePoints(proj1, proj2, pts_left_camera_space, pts_right_camera_space).T
        self.pts_3D = recovered_3D_points_in_homogenous[:, 0:3]/recovered_3D_points_in_homogenous[:,3:]

    def display(self):
        fig = plt.figure(figsize = (10,10))
        ax = plt.axes(projection='3d')
        
        ax.scatter(self.pts_3D[:, 0], self.pts_3D[:,1], self.pts_3D[:, 2], c= self.pts_3D_color)
        ax.set_title('3D Parametric Plot')
        ax.set_xlabel('x', labelpad=20)
        ax.set_ylabel('y', labelpad=20)
        ax.set_zlabel('z', labelpad=20)
        plt.show()
        plt.plot(self.pts_3D[:, 0], self.pts_3D[:,1], '.')
        plt.show()