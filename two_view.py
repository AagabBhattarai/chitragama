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
        intrinsic_camera_matrix = [[689.87, 0, 380.17],[0, 691.04, 251.70],[0, 0, 1]]
        self.intrinsic_camera_matrix = np.float32(intrinsic_camera_matrix)
        temp = np.eye(4)
        self.proj1 =self.intrinsic_camera_matrix @ temp[:3,:4]
        self.transfomation_matrix = None
        self.proj1_alt = temp[:3,:4]
    
        
        self.kp1 = None
        self.kp2 = None
        self.des1 = None
        self.des2 = None
        self.inliers_left = None
        self.inliers_right = None
        self.left = None
        self.right = None
        
        self.pts_3D = []
        self.pts_3D_color = []
        self.camera_path = []   
        self.camera_path.append([0,0,0])
        self.stride = []
        self.start = 0
        self.stop = 0
    
    def process_image(self, filepaths: list[str]):
        print(type(filepaths))
        self.rgb_img1 = cv.imread(filepaths[0], cv.IMREAD_COLOR)
        assert self.rgb_img1 is not None, "Image not found in specified file path"
        
        self.rgb_img2 = cv.imread(filepaths[1], cv.IMREAD_COLOR)
        assert self.rgb_img2 is not None, "Image not found in specified file path"
        self.rgb_img1 = cv.cvtColor(self.rgb_img1, cv.COLOR_BGR2RGB)
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
            if m.distance < 0.75*n.distance:
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
        self.left = matching_points_from_left.reshape(-1,2)
        self.right = matching_points_from_right.reshape(-1,2)
        print(len(matching_points_from_left)) 
    
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
        
        for (x,y) in self.inliers_left[:]:
            pixel_color = self.rgb_img1[int(y),int(x)]
            # pixel_color = list(map(lambda x: float(x)/255, pixel_color))
            # pixel_color = np.array([0,255,0], np.uint8)
            self.pts_3D_color.append(pixel_color)
        
        self.transfomation_matrix = np.eye(4)
        self.transfomation_matrix[0:3, 0:3] = R
        self.transfomation_matrix[0:3, 3:4] = t

    def to_camera_coordinate(self, K, point: list[float]) -> list[float]:
        normalized = [  (point[0] - K[0,2]) / K[0,0] ,  (point[1] - K[1,2])/K[1,1] ];
        return normalized
    def find_3D_of_iniliers(self) -> list[list[float]]:
        pts_left_camera_space = list()
        pts_right_camera_space = list()
        for ptl, ptr in zip(self.inliers_left, self.inliers_right):
            pts_left_camera_space.append(self.to_camera_coordinate(self.intrinsic_camera_matrix, ptl))
            pts_right_camera_space.append(self.to_camera_coordinate(self.intrinsic_camera_matrix, ptr))
        pts_left_camera_space = np.float32(pts_left_camera_space).T
        pts_right_camera_space = np.float32(pts_right_camera_space).T
        
        proj1 = self.proj1_alt 
        # proj1 is 3*4 matrix that muls 4*4 matrix
        proj2 = (proj1 @ self.transfomation_matrix)[0:3, 0:4] 
        # proj2 = (self.transfomation_matrix)[0:3, 0:4] 
        proj3 = np.linalg.inv(self.transfomation_matrix)
        
        recovered_3D_points_in_homogenous = cv.triangulatePoints(proj1, proj2, pts_left_camera_space, pts_right_camera_space).T
        self.proj1_alt = proj2
        self.camera_path.append((proj2[:3,3].ravel()).tolist())
        for pts in recovered_3D_points_in_homogenous[:, 0:3]/recovered_3D_points_in_homogenous[:,3:]:
            self.pts_3D.append(pts)
        
    def find_3D_of_iniliers_alt(self) -> None:
        temp: np.ndarray = np.eye(4)
        # x = K * [R|t] X; proj1 = K * [R|t] 
        # proj1 = self.intrinsic_camera_matrix @ temp[:3, :4]
        proj1 = self.proj1
        # x' = K*[R|t] X'; X= X';;;TO DEFINE PROPERLY  
        proj2 = proj1 @ self.transfomation_matrix
        
        recovered_3D_points_in_homogenous = cv.triangulatePoints(proj1, proj2, self.inliers_left.T, self.inliers_right.T).T
        recovered_3D_points_in_homogenous = cv.triangulatePoints(proj1, proj2, self.inliers_left.T, self.inliers_right.T).T
        
        self.proj1 = proj2
        self.stop += recovered_3D_points_in_homogenous.shape[0] 
        for pts in recovered_3D_points_in_homogenous[:, 0:3]/recovered_3D_points_in_homogenous[:,3:]:
            self.pts_3D.append(pts)
        
    def display(self):
        self.pts_3D = np.float32(self.pts_3D).reshape(-1,3)
        self.camera_path = np.float32(self.camera_path).reshape(-1,3)
        fig = plt.figure(figsize = (10,10))
        ax = plt.axes(projection='3d')
        
        start = self.start
        stop = self.stop
        # ax.scatter(self.pts_3D[self.start:self.stop, 0], self.pts_3D[self.start:self.stop,1], self.pts_3D[self.start:self.stop, 2],s= 1, c= self.pts_3D_color[self.start:self.stop])
        ax.scatter(self.pts_3D[self.start:self.stop, 0], self.pts_3D[self.start:self.stop,1], self.pts_3D[self.start:self.stop, 2],s= 1) 
        ax.scatter(self.camera_path[:,0],self.camera_path[:,1],self.camera_path[:,2])
        self.start = self.stop
        ax.set_title('3D Parametric Plot')
        ax.set_xlabel('x', labelpad=20)
        ax.set_ylabel('y', labelpad=20)
        ax.set_zlabel('z', labelpad=20)
        plt.show()
            
        # plt.scatter(self.pts_3D[:, 0], self.pts_3D[:,1], marker='.')
        self.pts_3D = self.pts_3D.tolist()
        self.camera_path = self.camera_path.tolist()
        plt.show()

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
    # def write_to_ply_file(self, filename="test.ply"):
    #     self.pts_3D = np.float32(self.pts_3D).reshape(-1,3)
    #     self.pts_3D_color = np.uint8(self.pts_3D_color).reshape(-1,3)
    #     x, y, z = self.pts_3D[:, 0], self.pts_3D[:, 1], self.pts_3D[:, 2]
    #     r, g, b = self.pts_3D_color[:, 0], self.pts_3D_color[:, 1], self.pts_3D_color[:, 2]

    #     pts = list(zip(x, y, z, r, g, b))

    #     vertex = PlyElement.describe(
    #         [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')],
    #         'vertex'
    #     )

    #     plydata = PlyData([vertex])

    #     for point in pts:
    #         plydata['vertex'].data.append(point)

    #     plydata.write(filename)