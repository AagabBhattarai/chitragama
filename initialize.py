import glob
import cv2 as cv
import numpy as np
from intrinsic_matrix import compute_intrinsic_matrix
from info_track import ImageView, MetaInfo
import inspect
from tqdm import tqdm 

def debug_info(Frames):
    for view in Frames:
        print(len(view.keypoints))
        print(len(view.descriptors))

def set_img_values(Frames:list, filepath, K: np.ndarray, distc: np.ndarray ):
    for path in tqdm(filepath, desc="Loading images"):  # Wrap filepath with tqdm
        view = ImageView()
        
        view.gray_img = cv.imread(path, cv.IMREAD_GRAYSCALE)
        assert view.gray_img is not None, f"AssertionError: gray_img is None at line {inspect.currentframe().f_lineno}"
        
        view.bgr_img = cv.imread(path, cv.IMREAD_COLOR)
        assert view.bgr_img is not None, f"AssertionError: bgr_img is None at line {inspect.currentframe().f_lineno}"
        
        view.K = K.copy()
        view.distortion_coefficient = distc.copy() 

        Frames.append(view)

def find_feature_points(Frames: list, metainfo:MetaInfo):
    sift = cv.SIFT_create()
    for  view in tqdm((Frames), total=len(Frames), desc="Computing SIFT feature points"): 
        view.keypoints, view.descriptors = sift.detectAndCompute(view.gray_img,None)
        metainfo.total_feature_points += len(view.keypoints)

 
def initialization(Frames: list, metainfo: MetaInfo):
    directory = "GustavIIAdolf"
    # directory = "nikolaiI"
    # directory = "guerre"
    # directory = "eglise"
    filepaths = glob.glob(f"{directory}/*.jpg")
    database_path = "sensor_width_camera_database.txt"
    intrinsic_camera_matrix = compute_intrinsic_matrix(filepaths[0], database_path)
    distortion_coefficients = np.zeros(4, dtype=np.float32).reshape(1,4)
    intrinsic_camera_matrix = np.float32(intrinsic_camera_matrix)
    
    #open images
    set_img_values(Frames, filepaths, intrinsic_camera_matrix, distortion_coefficients)
    assert len(Frames) == len(filepaths), f"AssertionError: Frames not initialized properly Line:{inspect.currentframe().f_lineno}"
    #Compute SIFT features
    find_feature_points(Frames, metainfo)


