import glob
import cv2 as cv
import numpy as np
from intrinsic_matrix import compute_intrinsic_matrix
from info_track import ImageView, MetaInfo
import inspect
from tqdm import tqdm 

def debug_info(Views):
    for view in Views:
        print(len(view.keypoints))
        print(len(view.descriptors))

def set_img_values(Views:list, filepath, K: np.ndarray, distc: np.ndarray ):
    for i,path in tqdm(enumerate(filepath), desc="Loading images"):
        view = ImageView()
        view.id = i
        view.gray_img = cv.imread(path, cv.IMREAD_GRAYSCALE)
        assert view.gray_img is not None, f"AssertionError: gray_img is None at line {inspect.currentframe().f_lineno}"
        
        view.bgr_img = cv.imread(path, cv.IMREAD_COLOR)
        assert view.bgr_img is not None, f"AssertionError: bgr_img is None at line {inspect.currentframe().f_lineno}"
        
        view.K = K.copy()
        view.distortion_coefficient = distc.copy() 

        Views.append(view)

def find_feature_points(Views: list, metainfo:MetaInfo):
    sift = cv.SIFT_create()
    for  view in tqdm((Views), total=len(Views), desc="Computing SIFT feature points"): 
        view.keypoints, view.descriptors = sift.detectAndCompute(view.gray_img,None)
        metainfo.total_feature_points += len(view.keypoints)
        view.set_initial_id()

 
def initialization(Views: list, metainfo: MetaInfo):
    directory = "GustavIIAdolf"
    # directory = "nikolaiI"
    # directory = "guerre"
    # directory = "eglise"
    # directory = "room1"
    filepaths = glob.glob(f"{directory}/*.jpg")
    filepaths = filepaths[:10]
    database_path = "sensor_width_camera_database.txt"
    intrinsic_camera_matrix = compute_intrinsic_matrix(filepaths[0], database_path)
    distortion_coefficients = np.zeros(4, dtype=np.float32).reshape(1,4)

    directory = "fountain"
    intrinsic_camera_matrix = [[689.87, 0, 380.17],[0, 691.04, 251.70],[0, 0, 1]]
    intrinsic_camera_matrix = np.float32(intrinsic_camera_matrix)
    filepaths = glob.glob(f"{directory}/*.png")
    #open images
    set_img_values(Views, filepaths, intrinsic_camera_matrix, distortion_coefficients)
    assert len(Views) == len(filepaths), f"AssertionError: Views not initialized properly Line:{inspect.currentframe().f_lineno}"
    #Compute SIFT features
    find_feature_points(Views, metainfo)
    metainfo.total_views = len(Views)


