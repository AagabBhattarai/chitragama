import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from scipy.spatial.distance import cdist

def filter_with_local_neighbour(points3d, n_neighbours=9, threshold=1):
    #pairwise distances for all points
    distances = cdist(points3d, points3d, 'euclidean')
    mean_distances = []
    num_pts = len(points3d)
    for i  in range(num_pts):
        nearest_indices = np.argsort(distances[i])[1:n_neighbours+1]
        mean_distance = np.mean(distances[i][nearest_indices])
        mean_distances.append(mean_distance)
    
    mean_mean_distances = np.mean(mean_distances)
    std_dev_mean_distances = np.std(mean_distances, ddof=1)
    threshold = mean_mean_distances + threshold * std_dev_mean_distances
    
    inliers_mask = np.array(mean_distances) < threshold
    return inliers_mask

def filtering_with_zscore(points3d, threshold=5):
    mean = np.mean(points3d, axis=0)
    std_dev = np.std(points3d, axis=0)
    
    z_scores = np.abs((points3d - mean) / std_dev)
    inliers_mask =  (z_scores < threshold).all(axis=1) 
    return inliers_mask

def filtering_with_iqr(points3d, k=1.5):
    # Calculate quartiles (Q1, Q3) and IQR for each dimension
    q1 = np.percentile(points3d, 25, axis=0)
    q3 = np.percentile(points3d, 75, axis=0)
    iqr = q3 - q1
    # boundaries for inliers
    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr
    
    inliers_mask = np.all((points3d > lower_bound) & (points3d < upper_bound), axis=1) 
    
    return inliers_mask 

def outlier_filtering(points3d, method='l'):
    if method == 'i':
        inliers_mask = filtering_with_iqr(points3d)
    elif method == 'z':
        inliers_mask = filtering_with_zscore(points3d)
    elif method =='l':
        inliers_mask = filter_with_local_neighbour(points3d)
    return inliers_mask

def display(points3D, camera_param, transform_cam_pose = False):
        camera_path = list()
        if transform_cam_pose:
            for orientation in camera_param:
                rvec = orientation[:3]
                tvec = orientation[3:6]
                Rmat, _ = cv.Rodrigues(rvec)
                camera_path.append((-Rmat.T @ tvec).tolist())
                
        points3D = np.float32(points3D).reshape(-1,3)
        camera_path = np.float32(camera_path).reshape(-1,3)
        
        ax = plt.axes(projection='3d')
        ax.scatter(points3D[:, 0], points3D[:,1], points3D[:, 2],s= 10) 
        # ax.scatter(camera_path[:,0],camera_path[:,1],camera_path[:,2])

        ax.set_title('3D Parametric Plot')
        ax.set_xlabel('x', labelpad=20)
        ax.set_ylabel('y', labelpad=20)
        ax.set_zlabel('z', labelpad=20)
        plt.show()