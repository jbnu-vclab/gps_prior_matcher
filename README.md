# GPS Prior Matcher
## About
This software is a feature matcher with gps prior (overlap zone acquisition).  
We use COLMAP for feature extraction, geometric verification, 3d reconstruction, and its database format to handle data.

## Dataset
The riding video frames which only moves to forward.  
Every frames must have gps information with EXIF.

## How to use
1. Extract SIFT features (COLMAP)
2. **Run this code**
   * Check gps_prior_matcher.py for various options.
3. Do Geometric verification (COLMAP)
   * COLMAP only uses match pairs in two_view_geometries table.
   * You can run custom feature match with match_info.txt to get two_view_geometries.
4. Run 3D reconstruction (COLMAP)
   
## License

