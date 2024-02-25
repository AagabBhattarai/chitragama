import cv2 as cv
import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

class Bundle_Adjusment:
    def __init__(self):
        opitmal = None
    
    def rotate(self,points, rot_vecs):
        """Rotate points by given rotation vectors.
    
        Rodrigues' rotation formula is used.
        """
        theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
        with np.errstate(invalid='ignore'):
            v = rot_vecs / theta
            v = np.nan_to_num(v)
        dot = np.sum(points * v, axis=1)[:, np.newaxis]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v
    
    def project(self, points, camera_params):
        """Convert 3-D points to 2-D by projecting onto images."""
        points_proj = self.rotate(points, camera_params[:, :3])
        points_proj += camera_params[:, 3:6]
        # points_proj = -points_proj[:, :2] / points_proj[:, 2, np.newaxis]
        points_proj = points_proj[:, :2] / points_proj[:, 2, np.newaxis]
        fx = camera_params[:, 6]
        cx = camera_params[:, 7]
        fy = camera_params[:, 8]
        cy = camera_params[:, 9]
 
        points_proj[:, 0] = points_proj[:, 0] * fx + cx
        points_proj[:, 1] = points_proj[:, 1] * fy + cy

        return points_proj
        
    def fun(self,params, n_cameras, n_points, camera_indices, point_indices, points_2d):
        """Compute residuals.

        `params` contains camera parameters and 3-D coordinates.
        """
        camera_params = params[:n_cameras * 10].reshape((n_cameras, 10))
        points_3d = params[n_cameras * 10:].reshape((n_points, 3))
        points_proj = self.project(points_3d[point_indices], camera_params[camera_indices])
        return (points_proj - points_2d).ravel()

    def bundle_adjustment_sparsity(self,n_cameras, n_points, camera_indices, point_indices):
        m = camera_indices.size * 2
        n = n_cameras * 10 + n_points * 3
        A = lil_matrix((m, n), dtype=int)

        i = np.arange(camera_indices.size)
        for s in range(10):
            A[2 * i, camera_indices * 10 + s] = 1
            A[2 * i + 1, camera_indices * 10 + s] = 1

        for s in range(3):
            A[2 * i, n_cameras * 10 + point_indices * 3 + s] = 1
            A[2 * i + 1, n_cameras * 10 + point_indices * 3 + s] = 1

        return A 
    
    
    def do_BA(self, points_3d, camera_params, camera_indices, point_indices, points_2d) -> tuple:
        n_cameras = camera_params.shape[0]
        n_points = points_3d.shape[0]
        n = 10 * n_cameras + 3 * n_points
        m = 2 * points_2d.shape[0]

        print("n_cameras: {}".format(n_cameras))
        print("n_points: {}".format(n_points))
        print("Total number of parameters: {}".format(n))
        print("Total number of residuals: {}".format(m))
        print("3D points\n", points_3d[point_indices].shape)
        print("Camera params\n", camera_params[camera_indices].shape)
        
        x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))
        f0 = self.fun(x0, n_cameras, n_points, camera_indices, point_indices, points_2d)
        print("\nRes Error:\n",(sum(f0**2)**0.5)/m)
        plt.plot(f0)
        plt.show()
        
        A = self.bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)
        res = least_squares(self.fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                    args=(n_cameras, n_points, camera_indices, point_indices, points_2d))
        

        print("\nOptimized Res Error:\n",(sum(res.fun**2)**0.5)/m)
        plt.plot(res.fun)
        for pt in res.x[:10]:
            print(pt)
        for pt in res.x[10:20]:
            print(pt)
        for pt in res.x[20:30]:
            print(pt)
        camera_params = res.x[:10*n_cameras]
        points_3d = res.x[10*n_cameras:]
        plt.show()
        return camera_params, points_3d
        
        # def reprojection_error(self):
        #     #takes newly calculated 3D pts and 2D correspondences and calculate reprojection error
        #     #3D points are assumed to be in Eucledian Space
        #     original_pts = self.unique_pts_right
        #     pts_3D = self.pts_3D[self.start: self.stop]
        #     pts_3D = np.float32(pts_3D).reshape(-1,3)
        #     R = self.transformation_matrix[0:3, 0:3]
        #     t = self.transformation_matrix[0:3, 3]

        #     rvec, _ = cv.Rodrigues(R)
        #     reproj_pts, _ = cv.projectPoints(pts_3D, rvec, t, self.intrinsic_camera_matrix,distCoeffs=None)
        #     reproj_pts = reproj_pts.reshape(-1,2) 
        #     # print(f"Reprojected points:\n{reproj_pts[:10, :]}")
        #     # print(f"Original points:\n{original_pts[:10, :]}")
        #     # print(f"\nTransformation Matrix:\n{self.transformation_matrix}")
        #     error = cv.norm(original_pts, reproj_pts, normType=cv.NORM_L2)