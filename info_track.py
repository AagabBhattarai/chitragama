
class ImageView:
    def __init__(self):
        self.id = None   
        self.gray_img = None
        self.bgr_img = None 
        self.keypoints = None
        self.descriptors = None

        self.id_of_unique_feature = None #this stores "global descriptor for the keypoints" --just a number value
        self.unique_feature_status = dict() # Key = global_descriptor, Value= True/False (Status if the feature point can be triangulated)

        self.extrinsic_pose = None
        self.K = None   
        self.distortion_coefficient = None
    def set_initial_id(self):
        self.id_of_unique_feature = -1 * len(self.keypoints)

class MetaInfo:
    def __init__(self) -> None:
        self.total_feature_points = 0
        self.unique_feature_points = 0
        self.total_views = 0
        