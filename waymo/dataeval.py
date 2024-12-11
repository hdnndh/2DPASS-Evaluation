import os
import numpy as np
import glob

def read_pointcloud(bin_path):

    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return points

def read_labels(label_path):

    labels = np.fromfile(label_path, dtype=np.int32)
    return labels

def main():
    label_distance_sum = {}
    label_count = {}
    for s in range(10,22):
        sequence_dir = "./dataset/SemanticKitti/dataset/sequences/" + str(s)
        velodyne_dir = os.path.join(sequence_dir, "velodyne")
        label_dir = os.path.join(sequence_dir, "labels")

        bin_files = sorted(glob.glob(os.path.join(velodyne_dir, "*.bin")))



        for bin_file in bin_files:
            base_name = os.path.basename(bin_file)  
            label_file = os.path.join(label_dir, base_name.replace(".bin", ".label"))

            if not os.path.isfile(label_file):
                print(f"No label file found for {bin_file}, skipping...")
                continue

            points = read_pointcloud(bin_file)
            labels = read_labels(label_file)

            if len(points) != len(labels):
                print(f"Mismatch in points and labels count for {bin_file}, skipping...")
                continue

            distances = np.sqrt(np.sum(points[:, :3]**2, axis=1))

            unique_labels = np.unique(labels)
            for lbl in unique_labels:
                mask = (labels == lbl)
                dist_sum = distances[mask].sum()
                count = mask.sum()

                if lbl not in label_distance_sum:
                    label_distance_sum[lbl] = 0.0
                    label_count[lbl] = 0
                label_distance_sum[lbl] += dist_sum
                label_count[lbl] += count

    print("Average distance per label:")
    for lbl in sorted(label_distance_sum.keys()):
        avg_dist = label_distance_sum[lbl] / label_count[lbl]
        print(f"Label {lbl}: {avg_dist:.3f} m label count: {label_count[lbl]:0f}")

if __name__ == "__main__":
    main()
