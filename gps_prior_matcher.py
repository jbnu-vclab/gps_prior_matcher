import os
import sys
import argparse
import sqlite3
import cv2
import numpy as np
from pymap3d import ned
import collections
import feature_keypoint as fkp
import visualization as vis

IS_PYTHON3 = sys.version_info[0] >= 3
MAX_IMAGE_ID = 2**31 - 1

def normalize(vec: np.array) -> np.array:
    norm = np.linalg.norm(vec)
    norm = np.finfo(vec.dtype).eps if norm == 0 else norm
    return vec / norm

def cosine_similarity(vec1: np.array, vec2: np.array) -> float:
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def get_adjacent_vector(p1: np.array, p2: np.array) -> np.array:
    return normalize(p2 - p1)
    # return p2 - p1

def blob_to_array(blob, dtype, shape=(-1,)):
    #! np.fromstring causes deprecation warning(expected to use 'frombuffer') despite using python3
    # if IS_PYTHON3:
    #     return np.fromstring(blob, dtype=dtype).reshape(*shape)
    # else:
    return np.frombuffer(blob, dtype=dtype).reshape(*shape)

def array_to_blob(array):
    #! same reason of blob_to_array
    # if IS_PYTHON3:
    #     return array.tostring()
    # else:
    return np.getbuffer(array)

class GPSPriorMatcher:
    OVERLAP = 0
    NO_OVERLAP = 1

    def __init__(self):
        self.args = self.parse_args()
        connection = sqlite3.connect(self.args.database_path)
        self.image_path = self.args.image_path
        self.distance_threshold = self.args.distance_threshold
        self.overlap = self.args.overlap
        self.cursor = connection.cursor()
        self.read_cameras()
        self.read_images()
        self.read_keypoints()
        self.read_descriptors()

        self.image_width = self.cameras[0]['width']
        self.image_height = self.cameras[0]['height']

        self.init_matcher()
        self.sort_keypoints_and_descriptors()
    
    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--database_path", required=True)
        parser.add_argument("--image_path", required=True)
        # parser.add_argument("--output_path", required=True)
        parser.add_argument("--min_num_matches", type=int, default=15)
        parser.add_argument("--num_neighbors", type=int, default=10)
        parser.add_argument("--overlap", type=float, default=1.0)
        parser.add_argument("--distance_threshold", type=float, default=0.60)
        args = parser.parse_args()
        return args

    def pair_id_to_image_ids(self, pair_id):
        image_id2 = pair_id % MAX_IMAGE_ID
        image_id1 = (pair_id - image_id2) / MAX_IMAGE_ID
        return image_id1, image_id2

    def image_ids_to_pair_id(self, image_id1, image_id2):
        if image_id1 > image_id2:
            image_id1, image_id2 = image_id2, image_id1
        return image_id1 * MAX_IMAGE_ID + image_id2
    
    def geodetic_to_ned(self, lat:float, lon:float, alt:float) -> list:
        n, e, d = ned.geodetic2ned(lat, lon, alt, \
                                  self.observer[0], self.observer[1], self.observer[2])
        return [n, e, d]
    
    def init_matcher(self):
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        # FLANN_INDEX_LSH = 6
        # index_params = dict(algorithm = FLANN_INDEX_LSH,
        #                 table_number = 6,
        #                 key_size = 12,
        #                 multi_probe_level = 1
        #                 )
        search_params = dict(checks=32,
                        cross_check=True)

        # self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        self.matcher = cv2.BFMatcher()
    
    def read_cameras(self):
        self.cameras = []
        self.cursor.execute("SELECT camera_id, model, width, height, params, prior_focal_length FROM cameras;")
        for row in self.cursor:
            self.cameras.append(
                {
                    "camera_id" : int(row[0]),
                    "model" : int(row[1]),
                    "width" : int(row[2]),
                    "height" : int(row[3]),
                    "params" : blob_to_array(row[4], dtype=np.float64),
                    "prior_focal_length" : row[5]
                }
            )
        self.cameras = sorted(self.cameras, key=lambda d: d['camera_id']) 
        return self.cameras
    
    def read_images(self):
        self.images = []
        self.cursor.execute("SELECT image_id, name, camera_id, prior_qw, prior_qx, prior_qy, prior_qz, prior_tx, prior_ty, prior_tz FROM images;")
        for row in self.cursor:
            self.images.append(
                {
                    "image_id" : int(row[0]),
                    "name" : row[1],
                    "camera_id" : row[2],
                    "prior_qw" : row[3],
                    "prior_qx" : row[4],
                    "prior_qy" : row[5],
                    "prior_qz" : row[6],
                    "prior_tx" : float(row[7]),
                    "prior_ty" : float(row[8]),
                    "prior_tz" : float(row[9])
                }
            )
        self.images = sorted(self.images, key=lambda d: d['image_id'])
        self.last_image_id = len(self.images) - 1
        print(f'Last image ID : {self.last_image_id}')

        self.observer = (self.images[0]['prior_tx'], self.images[0]['prior_ty'], self.images[0]['prior_tz'])

        return self.images

    #* use 6 col keypoints
    def read_keypoints(self):
        self.keypoints = []
        self.cursor.execute("SELECT image_id, rows, cols, data FROM keypoints;")
        for row in self.cursor:
            self.keypoints.append(
                {
                    "image_id" : int(row[0]),
                    "row" : row[1],
                    "cols" : row[2],
                    "data" : blob_to_array(row[3], dtype=np.float32, shape=(row[1], row[2]))
                }
            )
        self.keypoints = sorted(self.keypoints, key=lambda d: d['image_id'])
        return self.keypoints
    
    def read_descriptors(self):
        self.descriptors = []
        self.cursor.execute("SELECT image_id, rows, cols, data FROM descriptors;")
        for row in self.cursor:
            self.descriptors.append(
                {
                    "image_id" : int(row[0]),
                    "row" : row[1],
                    "cols" : row[2],
                    "data" : blob_to_array(row[3], dtype=np.uint8, shape=(row[1], row[2]))
                }
            )
        self.descriptors = sorted(self.descriptors, key=lambda d: d['image_id'])
        return self.descriptors
    
    def read_two_view_geometries(self):
        self.two_view_geometries = []
        self.cursor.execute("SELECT pair_id, rows, cols, data, config, F, E, H, qvec, tvec FROM two_view_geometries;")
        for row in self.cursor:
            self.two_view_geometries.append(
                {
                    "pair_id" : int(row[0]),
                    "row" : row[1],
                    "cols" : row[2],
                    "data" : blob_to_array(row[3], dtype=np.uint32, shape=(row[1], row[2])),
                }
            )
        # self.two_view_geometries = sorted(self.two_view_geometries, key=lambda d: d['image_id'])
        return self.two_view_geometries
    
    def get_gps_from_image(self, image: dict) -> np.array:
        return np.array(self.geodetic_to_ned(image['prior_tx'], image['prior_ty'], image['prior_tz']))
    
    def get_cos_sim_from_images(self, image_id1: int, image_id2: int) -> float:
        image1 = self.images[image_id1]
        next_image1 = self.images[image_id1 + 1]

        #* use previous adjacent vector if image_id2 is last index
        image2 = self.images[image_id2 if image_id2 < self.last_image_id else image_id2 - 1]
        next_image2 = self.images[image_id2 + 1 if image_id2 < self.last_image_id else image_id2]
        
        gps1, next_gps1 = self.get_gps_from_image(image1), self.get_gps_from_image(next_image1)
        gps2, next_gps2 = self.get_gps_from_image(image2), self.get_gps_from_image(next_image2)

        vec1, vec2 = get_adjacent_vector(gps1, next_gps1), get_adjacent_vector(gps2, next_gps2)

        cos_sim = cosine_similarity(vec1, vec2)

        return cos_sim
    
    def sort_keypoints_and_descriptors(self):
        ''' sort by x axis
        '''    
        for image_id, keypoint in enumerate(self.keypoints):
            kp = keypoint['data']
            desc = self.descriptors[image_id]['data']

            sorted_pairs = sorted(zip(kp, desc), key=lambda d: d[0][0])
            kp, desc = zip(*sorted_pairs)

    def match_one_pair(self, image_id1: int, image_id2: int):
        cos_sim = self.get_cos_sim_from_images(image_id1, image_id2)
        print(f'Cosine similarity : {cos_sim}')

        #* no overlap
        if cos_sim <= 0:
            return self.NO_OVERLAP

        keypoints1, descriptors1 = self.keypoints[image_id1]['data'], self.descriptors[image_id1]['data']
        keypoints2, descriptors2 = self.keypoints[image_id2]['data'], self.descriptors[image_id2]['data']

        if 0.0 < cos_sim < 0.1:
            cos_sim = 0.1
        overlap_begin = int((1 - cos_sim) * self.image_width / 2)
        overlap_end = self.image_width - overlap_begin

        overlap_indices1 = [i for i, kp in enumerate(keypoints1) if overlap_begin < kp[0] < overlap_end]
        overlap_indices2 = [i for i, kp in enumerate(keypoints2) if overlap_begin < kp[0] < overlap_end]

        overlap_keypoints1 = np.array([kp for i, kp in enumerate(keypoints1) if i in overlap_indices1])
        overlap_keypoints2 = np.array([kp for i, kp in enumerate(keypoints2) if i in overlap_indices2])
        overlap_descriptors1 = np.array([desc for i, desc in enumerate(descriptors1) if i in overlap_indices1])
        overlap_descriptors2 = np.array([desc for i, desc in enumerate(descriptors2) if i in overlap_indices2])

        overlap_keypoints1 = fkp.arrays_to_keypoints(overlap_keypoints1)
        overlap_keypoints2 = fkp.arrays_to_keypoints(overlap_keypoints2)
        
        matches = self.matcher.knnMatch(overlap_descriptors1, overlap_descriptors2, 2)

        good_matches = [match[0] for match in matches if match[0].distance / match[1].distance < self.distance_threshold]

        # img1 = cv2.imread(self.image_path + '/' + self.images[image_id1]['name'])
        # img1 = vis.draw_left_right_arrow(img1, overlap_begin, 800, overlap_end, 800)
        # img2 = cv2.imread(self.image_path + '/' + self.images[image_id2]['name'])
        # img2 = vis.draw_left_right_arrow(img2, overlap_begin, 800, overlap_end, 800)

        # res = vis.draw_matches_pair(img1, img2, overlap_keypoints1, overlap_keypoints2, good_matches)

        # cv2.imshow(f'{image_id1}_{image_id2}', res)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        return good_matches

    def gps_prior_matching(self):
        self.match_one_pair(20, 40)

        for image in self.images:
            image_id = image['image_id']



if __name__ == "__main__":
    gps_matcher = GPSPriorMatcher()
    gps_matcher.gps_prior_matching()
