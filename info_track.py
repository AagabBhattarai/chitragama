import numpy as np
class ImageView:
    def __init__(self):
        self.id = None   
        self.gray_img = None
        self.bgr_img = None 
        self.keypoints = None
        self.descriptors = None

        self.global_descriptor = list()#this stores "global descriptor for the keypoints" --just a number value
        self.global_descriptor_and_index = dict() # Key = global_descriptor, Value= Index for Keypoints
        self.global_descriptor_set = set()
        temp = np.eye(4)
        self.extrinsic_pose = temp[:3,:4]
        self.K = None   
        self.distortion_coefficient = None

        self.is_processed = False
    def set_initial_id(self):
        self.global_descriptor = [-1] * len(self.keypoints)

class MetaInfo:
    def __init__(self) -> None:
        self.total_feature_points = 0
        self.unique_feature_points = 0
        self.total_views = 0
        self.bundle_adjustment_time = False
        self.error_sum = 0
        self.do_bundle_adjustment = True
        self.views_used = set() 
        
class ImagePair:
    def __init__(self, l, r, p, i_m, average_depth, median_triangulation_angle):
        self.view_1 = l
        self.view_2 = r

        self.projection = p # actually transformation matrix form cam1 wc to cam2 wc 
        self.matches = i_m
        self.median_angle = median_triangulation_angle
        self.average_depth = average_depth


class ObjectPoints:
    def __init__(self, unique_feature_points):
        self.pts_3D = list()
        self.pts_3D_color = list()
        self.camera_path = list()
        self.pts_3D_global_descriptor = np.zeros(unique_feature_points, dtype=bool) #use to and with feature track to get overlapp view fast
        self.pts_3D_global_descriptor_value = list() #AS 3D points added stores which 2D point it was triangulated from by using global descriptor

        #BA requirements
        self.point_indices = list() #expands pts_3D list as per point_2d correspondence with pts_3D
        self.points_2d = list() #Keypoint location/image pixel point storage
        self.camera_indices = list() #points_2d to camera pose map
        self.camera_params = list() #extrinsic pose
        self.camera_params_map = dict() 
        self.n_camera_params_ba = 6 #only extrinsic pose will get modified

        self.new_descriptor_addition_start_index = 0
        self.new_descriptor_addition_stop_index = 0
        self.incremental = False
class Preferences:
    def __init__(self):
        self.filtering_l = True
        self.filtering_i = False
        self.error_threshold = 10 
        