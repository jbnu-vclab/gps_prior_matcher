import cv2

def draw_left_right_arrow(img, x1, y1, x2, y2, color=(255,0,0)):
    cv2.arrowedLine(img, (x1, y1), (x2, y2), color=color, thickness=2)
    cv2.arrowedLine(img, (x2, y2), (x1, y1), color=color, thickness=2)
    return img

def draw_matches_pair(img1, img2, keypoints1, keypoints2, matches):
    res = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None, \
                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                        # flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    
    return res
