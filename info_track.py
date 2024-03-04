
class ImageView:
    def __init__(self):
        self.id = None   
        self.gray_img = None
        self.bgr_img = None 
        self.keypoints = None
        self.descriptors = None

        self.global_descriptor = list()#this stores "global descriptor for the keypoints" --just a number value
        # self.global_descriptor_status = dict() # Key = global_descriptor, Value= True/False (Status if the feature point can be triangulated)
        self.global_descriptor_status = list() # Key = global_descriptor, Value= True/False (Status if the feature point can be triangulated)

        self.extrinsic_pose = None
        self.K = None   
        self.distortion_coefficient = None
    def set_initial_id(self):
        self.global_descriptor = [-1] * len(self.keypoints)
        self.global_descriptor_status = [False] * len(self.keypoints)

class MetaInfo:
    def __init__(self) -> None:
        self.total_feature_points = 0
        self.unique_feature_points = 0
        self.total_views = 0
        self.global_descriptor_value = 0
        
        
class ImagePair:
    def __init__(self, l, r, p, i_m, baseline_depth_ratio):
        self.frame_1 = l
        self.frame_2 = r

        self.projection = p # actually transformation matrix form frame 1 wc to frame 2 wc 
        self.inlier_matches = i_m
        self.baseline_depth = baseline_depth_ratio
        
        
        
        