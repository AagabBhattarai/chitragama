import numpy as np

class Frame_info:
    def __init__(self,n):
        self.frame_no = n
    def get_frame_no(self) -> int:
        return self.frame_no
    
class Frames:
    def __init__(self):
        self.frame1 = None
        self.frame2 = None
        self.frame3 = None
        
        #refactoring required
        self.frame1_no=None
        self.frame2_no=None
        self.frame3_no=None
        
        self.point_indices = list()
        self.camera_indices = list()
        self.points_2D = list()
        self.points_3D = list()
        self.camera_params = list()
    
        
