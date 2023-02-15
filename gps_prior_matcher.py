import argparse
import sqlite3
import cv2
import math
import time
import numpy as np
import feature_keypoint as fkp
import visualization as vis
from pymap3d import ned
from tqdm import tqdm

MAX_IMAGE_ID = 2**31 - 1

def distance(p1: np.array, p2: np.array) -> float:
    return np.linalg.norm(p2-p1)

def gaussian(x):
    mu, sigma = 0.0, 1.0
    return (1 / np.sqrt(2 * np.pi * sigma**2)) * np.exp(-(x-mu)**2 / (2 * sigma**2))

def normalize(vec: np.array) -> np.array:
    norm = np.linalg.norm(vec)
    norm = np.finfo(vec.dtype).eps if norm == 0 else norm
    return vec / norm

def cosine_similarity(v1: np.array, v2: np.array) -> float:
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def get_adjacent_vector(p1: np.array, p2: np.array) -> np.array:
    return normalize(p2 - p1)

def get_signed_angle(v1: np.array, v2: np.array) -> float:
    ''' calculate signed angle of two vectors using only x and y elements.
    '''
    v1_u = normalize(v1)
    v2_u = normalize(v2)
    signed_angle = math.atan2(v2_u[1],v2_u[0]) - math.atan2(v1_u[1],v1_u[0])
    return signed_angle

def blob_to_array(blob, dtype, shape=(-1,)):
    return np.frombuffer(blob, dtype=dtype).reshape(*shape)

def array_to_blob(array):
    return array.tobytes()

class GPSPriorMatcher:
    OVERLAP = 0
    NO_OVERLAP = 1

    def __init__(self):
        self.args = self.parse_args()
        self.connection = sqlite3.connect(self.args.database_path)
        self.image_path = self.args.image_path
        self.distance_threshold = self.args.distance_threshold
        self.num_neighbors = self.args.num_neighbors
        self.min_match_pairs = self.args.min_match_pairs
        self.overlap = self.args.overlap
        self.fov = self.args.fov
        self.max_distance = self.args.max_distance
        
        self.cursor = self.connection.cursor()
        self.read_cameras()
        self.read_images()
        self.read_keypoints()
        self.read_descriptors()

        self.image_width = self.cameras[0]['width']
        self.image_height = self.cameras[0]['height']

        self.init_matcher()
        self.sort_keypoints_and_descriptors()
        self.precompute_gps()
        self.precompute_accumulated_distance()
        self.precompute_adjacent_vectors()

        # vis.draw_gps_data_with_distance(self.gps, self.accumulated_dist)
    
    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--database_path", required=True)
        parser.add_argument("--image_path", required=True)
        parser.add_argument("--num_neighbors", type=int, default=15)
        parser.add_argument("--min_match_pairs", type=int, default=20)
        parser.add_argument("--overlap", type=float, default=1.0)
        parser.add_argument("--fov", type=float, default=2.2)
        parser.add_argument("--distance_threshold", type=float, default=0.6)
        parser.add_argument("--max_distance", type=float, default=28.0)
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
        search_params = dict(checks=32,
                        cross_check=True)
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

    def get_gps_from_image(self, image: dict) -> np.array:
        return np.array(self.geodetic_to_ned(image['prior_tx'], image['prior_ty'], image['prior_tz']))
    
    def get_vector_from_image(self, image_id:int) -> np.array:
        #* use previous adjacent vector if image_id2 is last index
        vec = self.get_n_mean_vector(image_id, 8)
        return vec
    
    def precompute_gps(self):
        self.gps = [self.get_gps_from_image(img) for img in self.images]

    def precompute_accumulated_distance(self):
        self.accumulated_dist = [0]
        prev_dist = 0.0
        prev_gps = self.gps[0]
        for cur_gps in self.gps:
            prev_dist = distance(prev_gps, cur_gps) + prev_dist
            prev_gps = cur_gps
            self.accumulated_dist.append(prev_dist)
    
    def precompute_adjacent_vectors(self):
        self.adjacent_vectors = []
        for i in range(self.last_image_id):
            vec = get_adjacent_vector(self.gps[i], self.gps[i+1])
            self.adjacent_vectors.append(vec)
        self.adjacent_vectors.append(self.adjacent_vectors[-1])

    def get_n_mean_vector(self, image_id:int, n:int):
        before = max(image_id - int(n / 2), 0)
        after = min(image_id + int(n / 2) + 1, self.last_image_id)

        stack_list = []
        for i in range(before, after):
            stack_list.append(self.adjacent_vectors[i])
        
        mean_vec = np.mean(np.vstack(stack_list), axis=0)

        return mean_vec

    def get_accumulated_distance(self, image_id1, image_id2):
        if image_id1 > image_id2:
            image_id1, image_id2 = image_id2, image_id1

        return self.accumulated_dist[image_id2] - self.accumulated_dist[image_id1]
            
    def get_signed_angle_from_images(self, image_id1: int, image_id2: int) -> float:
        vec1, vec2 = self.get_vector_from_image(image_id1), self.get_vector_from_image(image_id2)
        signed_angle = get_signed_angle(vec1, vec2)

        return signed_angle

    def get_cos_sim_from_images(self, image_id1: int, image_id2: int) -> float:
        vec1, vec2 = self.get_vector_from_image(image_id1), self.get_vector_from_image(image_id2)
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
        
    def change_indices_for_db(self, matches, overlap_indices1, overlap_indices2):
        '''Change keypoint indices in match results to database indices
        '''
        db_indices = []
        for dmatch in matches:
            dmatch.queryIdx = overlap_indices1[dmatch.queryIdx]
            dmatch.trainIdx = overlap_indices2[dmatch.trainIdx]
            db_indices.append([dmatch.queryIdx, dmatch.trainIdx])

        return np.array(db_indices, dtype=np.uint32)

    def match_single_pair(self, image_id1: int, image_id2: int, signed_angle:float, accumulated_dist: float):
        keypoints1, descriptors1 = self.keypoints[image_id1]['data'], self.descriptors[image_id1]['data']
        keypoints2, descriptors2 = self.keypoints[image_id2]['data'], self.descriptors[image_id2]['data']

        overlap_begin1, overlap_end1, overlap_begin2, overlap_end2 = self.get_overlap_indices(signed_angle, accumulated_dist)

        overlap_indices1 = [i for i, kp in enumerate(keypoints1) if overlap_begin1[0] <= kp[0] <= overlap_end1[0] and overlap_begin1[1] <= kp[1] <= overlap_end1[1]]
        overlap_indices2 = [i for i, kp in enumerate(keypoints2) if overlap_begin2[0] <= kp[0] <= overlap_end2[0] and overlap_begin2[1] <= kp[1] <= overlap_end2[1]]

        overlap_keypoints1 = np.array([kp for i, kp in enumerate(keypoints1) if i in overlap_indices1])
        overlap_keypoints2 = np.array([kp for i, kp in enumerate(keypoints2) if i in overlap_indices2])
        overlap_descriptors1 = np.array([desc for i, desc in enumerate(descriptors1) if i in overlap_indices1])
        overlap_descriptors2 = np.array([desc for i, desc in enumerate(descriptors2) if i in overlap_indices2])

        overlap_keypoints1 = fkp.arrays_to_keypoints(overlap_keypoints1)
        overlap_keypoints2 = fkp.arrays_to_keypoints(overlap_keypoints2)
        origin_keypoints1 = fkp.arrays_to_keypoints(keypoints1)
        origin_keypoints2 = fkp.arrays_to_keypoints(keypoints2)
    
        raw_matches = self.matcher.knnMatch(overlap_descriptors1, overlap_descriptors2, 2)

        good_matches = [match[0] for match in raw_matches if match[0].distance / match[1].distance < self.distance_threshold]

        img1 = cv2.imread(self.image_path + '/' + self.images[image_id1]['name'])
        img2 = cv2.imread(self.image_path + '/' + self.images[image_id2]['name'])

        img1 = cv2.rectangle(img1, (int(overlap_begin1[0]), int(overlap_begin1[1])), (int(overlap_end1[0]), int(overlap_end1[1])), (80, 248, 146), 8)
        img2 = cv2.rectangle(img2, (int(overlap_begin2[0]), int(overlap_begin2[1])), (int(overlap_end2[0]), int(overlap_end2[1])), (80, 248, 146), 8)
        res = vis.draw_matches_pair(img1, img2, overlap_keypoints1, overlap_keypoints2, good_matches)

        matches = self.change_indices_for_db(good_matches, overlap_indices1, overlap_indices2)

        return matches
    
    def get_overlap_indices(self, signed_angle:float, accumulated_dist:float) -> tuple:
        w, h, half_w, half_h = self.image_width, self.image_height, self.image_width/2, self.image_height/2

        #* bigger alpha means shorter distance between image1 and image2
        alpha = 1 - (accumulated_dist / self.max_distance)
        beta = (signed_angle / self.fov) * w * alpha * 2.4
        assert(alpha <= 1.0)

        init_box = np.float32([[0, 0, 1],[w, h, 1]]).transpose()
    
        transform_mat = np.float32([[alpha, 0, half_w * (1 - alpha) + beta], [0, alpha, half_h * (1 - alpha)], [0, 0, 1]])

        moved_box = transform_mat @ init_box

        xmin_1, ymin_1, xmax_1, ymax_1 = 0, 0, w, h
        xmin_2, ymin_2, xmax_2, ymax_2 = moved_box[0][0], moved_box[1][0], moved_box[0][1], moved_box[1][1]

        intersection1 = ((max(xmin_1, xmin_2), max(ymin_1, ymin_2)), (min(xmax_1, xmax_2), min(ymax_1, ymax_2)))

        overlap_begin1, overlap_end1 = intersection1

        overlap_begin2 = ((1 / alpha) * (overlap_begin1[0] - xmin_2), 0)
        overlap_end2 = ((1 / alpha) * (overlap_end1[0] - xmin_2), h)
        
        return overlap_begin1, overlap_end1, overlap_begin2, overlap_end2
            
    def gps_prior_matching(self):
        match_result = []
        match_info = []
        for image_id1, image1 in enumerate(tqdm(self.images)):
            for i in range(1, self.num_neighbors + 1):
                image_id2 = image_id1 + i
                if image_id2 > self.last_image_id:
                    break
                
                image1_name = self.images[image_id1]['name']
                image2_name = self.images[image_id2]['name']
                # print(f'- Matching {image1_name} & {image2_name}...')

                signed_angle = self.get_signed_angle_from_images(image_id1, image_id2)
                unsigned_angle = abs(signed_angle)

                if unsigned_angle >= self.fov:
                    # print(f'zero matching expected (angle is bigger than fov, {signed_angle}, {unsigned_angle})')
                    break
                
                dist = gps_matcher.get_accumulated_distance(image_id1, image_id2)

                matches = self.match_single_pair(image_id1, image_id2, signed_angle, dist)
                shape = matches.shape

                if shape[0] < self.min_match_pairs:
                    # print('zero matching expected (too few matches)')
                    break

                match_result.append({
                    'pair_id': self.image_ids_to_pair_id(image_id1+1, image_id2+1),
                    'shape': shape,
                    'blob': array_to_blob(matches)
                })
                match_info.append((image1_name, image2_name))

        return match_result, match_info

    def write_single_match(self, match):
        pair_id, shape, blob = match['pair_id'], match['shape'], match['blob']

        assert(len(shape) == 2)
        assert(shape[1] == 2)

        self.cursor.execute(
            "INSERT INTO matches VALUES (?, ?, ?, ?)",
            (pair_id,) + shape + (blob,))
    
    def write_match_result(self, match_result):
        for match in match_result:
            self.write_single_match(match)

    def write_match_info(self, match_info):
        with open('./match_info.txt', 'w') as f:
            for match in match_info:
                f.write(f'{match[0]} {match[1]}\n')
    
    def commit_and_close_connection(self):
        self.connection.commit()
        self.connection.close()


def debug(gps_matcher):
    s, e = 0, 5 

    signed_angle = gps_matcher.get_signed_angle_from_images(s,e)
    print(signed_angle)

    distance = gps_matcher.get_accumulated_distance(s, e)
    print(distance)

    gps_matcher.match_single_pair(s, e, signed_angle, distance)

if __name__ == "__main__":
    t1 = time.time()
    gps_matcher = GPSPriorMatcher()

    # debug(gps_matcher)

    match_result, match_info = gps_matcher.gps_prior_matching()
    gps_matcher.write_match_result(match_result)
    gps_matcher.write_match_info(match_info)
    gps_matcher.commit_and_close_connection()

    print(f'Elapsed time: {(time.time() - t1)/60.0} min' )
