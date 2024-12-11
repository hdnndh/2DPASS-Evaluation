import os
import numpy as np

WAYMO_TO_SEMANTIC_KITTI = {
    0: 0,  # TYPE_UNDEFINED -> unlabeled
    1: 10,  # TYPE_CAR -> car
    2: 18,  # TYPE_TRUCK -> truck
    3: 13,  # TYPE_BUS -> bus (mapped to "other-vehicle")
    4: 20,  # TYPE_OTHER_VEHICLE -> other-vehicle
    5: 32,  # TYPE_MOTORCYCLIST -> motorcyclist
    6: 31,  # TYPE_BICYCLIST -> bicyclist
    7: 30,  # TYPE_PEDESTRIAN -> person
    8: 81,  # TYPE_SIGN -> traffic-sign
    9: 81,  # TYPE_TRAFFIC_LIGHT -> traffic-sign
    10: 80,  # TYPE_POLE -> pole
    11: 49,  # TYPE_CONSTRUCTION_CONE -> other-ground
    12: 11,  # TYPE_BICYCLE -> bicycle
    13: 15,  # TYPE_MOTORCYCLE -> motorcycle
    14: 50,  # TYPE_BUILDING -> building
    15: 70,  # TYPE_VEGETATION -> vegetation
    16: 71,  # TYPE_TREE_TRUNK -> trunk
    17: 9,  # TYPE_CURB -> road
    18: 9,  # TYPE_ROAD -> road
    19: 60,  # TYPE_LANE_MARKER -> lane-marking
    20: 49,  # TYPE_OTHER_GROUND -> other-ground
    21: 48,  # TYPE_WALKABLE -> sidewalk
    22: 48,  # TYPE_SIDEWALK -> sidewalk
}

def convert_directory(input_dir, output_dir):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".label"):
            input_file_path = os.path.join(input_dir, file_name)
            output_file_path = os.path.join(output_dir, file_name)
            
            waymo_labels = np.fromfile(input_file_path, dtype=np.int32)
            
            semantic_kitti_labels = np.array([WAYMO_TO_SEMANTIC_KITTI.get(label, 0) for label in waymo_labels], dtype=np.int32)
            
            semantic_kitti_labels.tofile(output_file_path)
            
            print(f"Converted {input_file_path} to {output_file_path}")

# Example usage

input_directory = "./dataset/SemanticKitti/dataset/sequences/09/labels"  
output_directory = "./dataset/SemanticKitti/dataset/sequences/09/labels" 

convert_directory(input_directory, output_directory)
