import open3d as o3d
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog


def main():
    # Select point cloud file
    root = tk.Tk()
    root.withdraw()
    input_path = filedialog.askopenfilename(
        title="Select a .ply point cloud file (statistical denoising)",
        filetypes=[("PLY files", "*.ply"), ("All files", "*.*")]
    )
    if not input_path or not os.path.exists(input_path):
        print("No valid file selected.")
        return

    # Read point cloud
    pcd = o3d.io.read_point_cloud(input_path)
    if len(pcd.points) == 0:
        print("Point cloud is empty.")
        return
    print(f"Successfully loaded, {len(pcd.points)} points in total.")

    # Get user parameters
    try:
        nb_neighbors = int(input("Enter neighborhood point count (e.g., 10): "))
        std_ratio = float(input("Enter standard deviation multiplier (e.g., 2.0): "))
    except ValueError:
        print(" Invalid input, please enter numbers.")
        return

    # Perform statistical denoising
    filtered_pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)

    print(f"Denoising completed, {len(filtered_pcd.points)} points remaining.")

    # Save result
    base = os.path.splitext(input_path)[0]
    output_path = f"{base}_StatDenoised.ply"
    o3d.io.write_point_cloud(output_path, filtered_pcd, write_ascii=False)
    print(f"Result saved to: {output_path}")


if __name__ == "__main__":
    main()
