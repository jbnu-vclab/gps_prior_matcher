import cv2
import math

def array_to_keypoint(arr):
    fkp = feature_keypoint(arr[0], arr[1], arr[2], arr[3], arr[4], arr[5])
    return fkp

def arrays_to_keypoints(li):
    fkps = []
    for kp in li:
        fkp = array_to_keypoint(kp)
        fkps.append(fkp.to_cv_keypoint())
        
    return fkps

class feature_keypoint:
    def __init__(self, x, y, a11, a12, a21, a22):
        self.x = x
        self.y = y
        self.a11 = a11;
        self.a12 = a12;
        self.a21 = a21;
        self.a22 = a22;
        self.scale = self.compute_scale()
        self.orientation = self.compute_orientation()
    
    def get_x(self):
        return self.x

    def to_cv_keypoint(self):
        return cv2.KeyPoint(x=self.x, y=self.y, size=self.scale, angle=self.orientation)
    
    def compute_scale(self):
        return (self.compute_scale_x() + self.compute_scale_y()) / 2.0

    def compute_scale_x(self):
        return math.sqrt(self.a11 * self.a11 + self.a21 * self.a21)

    def compute_scale_y(self):
        return math.sqrt(self.a12 * self.a12 + self.a22 * self.a22)

    def compute_orientation(self):
        return math.atan2(self.a21, self.a11)
    
    def compute_shear(self):
        return math.atan2(-self.a12, self.a22) - self.compute_orientation()
    