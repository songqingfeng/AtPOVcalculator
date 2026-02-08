import open3d as o3d
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog


class DbsProcessor:
    def __init__(self, pcd):
        self.pcd = pcd

    def process(self, eps=0.02, min_points=10, limit=100):
        """
        Perform DBSCAN clustering on point cloud and intelligently filter valid clusters.
        :param eps: DBSCAN neighborhood radius
        :param min_points: Minimum points per cluster
        :param limit: Minimum point count threshold for cluster retention
        :return: Filtered point cloud (open3d.geometry.PointCloud)
        """
        print("Performing DBSCAN clustering...")
        labels = np.array(self.pcd.cluster_dbscan(eps=eps, min_points=min_points))

        min_label = labels.min()
        max_label = labels.max()

        if max_label == -1:
            print("Warning: All points are noise, no valid clusters detected.")
            return o3d.geometry.PointCloud()  # Return empty point cloud

        print(f"Detected {max_label + 1} valid clusters (excluding noise).")

        pcdjoint = o3d.geometry.PointCloud()
        total_points = 0
        max_cluster_size = 0
        largest_cluster_pcd = None

        # Iterate through all labels (including -1 noise)
        for label in range(min_label, max_label + 1):
            label_indices = np.where(labels == label)[0]
            label_pcd = self.pcd.select_by_index(label_indices)
            cluster_size = len(label_pcd.points)

            print(f'Label {label}: {cluster_size} points')

            # Record largest cluster (for fallback mechanism)
            if cluster_size > max_cluster_size:
                max_cluster_size = cluster_size
                largest_cluster_pcd = label_pcd

            # Retain clusters larger than limit
            if cluster_size > limit:
                print(f'  → Retained (>{limit})')
                pcdjoint += label_pcd
                total_points += cluster_size

        # Fallback mechanism: If total retained points < limit, keep the largest cluster
        if total_points < limit and largest_cluster_pcd is not None:
            print(f'Total retained points ({total_points}) < {limit}, falling back to largest cluster ({max_cluster_size} points)')
            pcdjoint = largest_cluster_pcd

        return pcdjoint


def main():
    # Open file selection dialog
    root = tk.Tk()
    root.withdraw()
    input_path = filedialog.askopenfilename(
        title="Select a .ply point cloud file",
        filetypes=[("PLY files", "*.ply"), ("All files", "*.*")]
    )

    if not input_path or not os.path.exists(input_path):
        print("No valid file selected. Exiting program.")
        return

    # Read point cloud
    pcd = o3d.io.read_point_cloud(input_path)
    if len(pcd.points) == 0:
        print("Error: Point cloud is empty.")
        return

    print(f"Successfully loaded point cloud with {len(pcd.points)} points.")

    # === Configuration parameters ===
    EPS = 1     # Clustering neighborhood radius (unit: cm)
    MIN_PTS = 20    # Minimum points per cluster
    LIMIT = 1000    # Minimum point count threshold for cluster retention

    # Execute processing
    processor = DbsProcessor(pcd)
    result_pcd = processor.process(eps=EPS, min_points=MIN_PTS, limit=LIMIT)

    # Save result
    if len(result_pcd.points) == 0:
        print("Warning: Processed point cloud is empty, skipping file save.")
        return

    base_name = os.path.splitext(input_path)[0]
    output_path = f"{base_name}_DBSCANFiltered.ply"
    o3d.io.write_point_cloud(output_path, result_pcd, write_ascii=False)
    print(f"Processing completed! Result saved to: {output_path}")


if __name__ == "__main__":
    main()
