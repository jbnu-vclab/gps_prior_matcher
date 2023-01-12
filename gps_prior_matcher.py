import os
import sys
import argparse
import sqlite3
import cv2
import numpy as np
import collections

IS_PYTHON3 = sys.version_info[0] >= 3
MAX_IMAGE_ID = 2**31 - 1

def normalize(vec: np.array) -> np.array:
    norm = np.linalg.norm(vec)
    norm = np.finfo(vec.dtype).eps if norm == 0 else norm
    return vec / norm

def cosine_similarity(vec1: np.array, vec2: np.array) -> float:
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def get_adjacent_vector(p1: np.array, p2: np.array):
    return normalize(p2 - p1)

class GPSPriorMatcher:
    def __init__(self):
        self.args = self.parse_args()
        connection = sqlite3.connect(self.args.database_path)
        self.cursor = connection.cursor()
        self.read_images()
        self.read_keypoints()
        self.read_descriptors()
    
    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--database_path", required=True)
        parser.add_argument("--output_path", required=True)
        parser.add_argument("--min_num_matches", type=int, default=15)
        parser.add_argument("--num_neighbors", type=int, default=10)
        args = parser.parse_args()
        return args
    
    def blob_to_array(self, blob, dtype, shape=(-1,)):
        #! np.fromstring causes deprecation warning(expected to use 'frombuffer') despite using python3
        # if IS_PYTHON3:
        #     return np.fromstring(blob, dtype=dtype).reshape(*shape)
        # else:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)

    def array_to_blob(self, array):
        #! same reason of blob_to_array
        # if IS_PYTHON3:
        #     return array.tostring()
        # else:
        return np.getbuffer(array)

    def pair_id_to_image_ids(self, pair_id):
        image_id2 = pair_id % MAX_IMAGE_ID
        image_id1 = (pair_id - image_id2) / MAX_IMAGE_ID
        return image_id1, image_id2

    def image_ids_to_pair_id(self, image_id1, image_id2):
        if image_id1 > image_id2:
            image_id1, image_id2 = image_id2, image_id1
        return image_id1 * MAX_IMAGE_ID + image_id2
    
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
                    "data" : self.blob_to_array(row[3], dtype=np.float32)
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
                    "data" : self.blob_to_array(row[3], dtype=np.uint8)
                }
            )
        self.descriptors = sorted(self.descriptors, key=lambda d: d['image_id']) 
        return self.descriptors
    
    def get_gps_from_image(self, image: dict) -> np.array:
        return np.array([image['prior_tx'], image['prior_ty']])
    
    def get_cos_sim(self, image_id1: int, image_id2: int) -> float:
        image1 = self.images[image_id1]
        next_image1 = self.images[image_id1 + 1]

        #* use previous vector if image_id2 is last index
        image2 = self.images[image_id2 if image_id2 < self.last_image_id else image_id2 - 1]
        next_image2 = self.images[image_id2 + 1 if image_id2 < self.last_image_id else image_id2]
        
        gps1, next_gps1 = self.get_gps_from_image(image1), self.get_gps_from_image(next_image1)
        gps2, next_gps2 = self.get_gps_from_image(image2), self.get_gps_from_image(next_image2)

        vec1, vec2 = get_adjacent_vector(gps1, next_gps1), get_adjacent_vector(gps2, next_gps2)

        cos_sim = cosine_similarity(vec1, vec2)

        return cos_sim

    def match_one_pair(self, image_id1, image_id2):
        keypoint1, keypoint2 = self.keypoints[image_id1], self.keypoints[image_id2]
        descriptor1, descriptor2 = self.descriptors[image_id1], self.descriptors[image_id2]
        
        cos_sim = self.get_cos_sim(image_id1, image_id2)

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

    def gps_prior_matching(self):
        for i, image in enumerate(self.images):
            image_id = image['image_id']



if __name__ == "__main__":
    gps_matcher = GPSPriorMatcher()
