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
        title="Select a .ply point cloud file (voxel downsampling)",
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

    # Get user parameters (unit: cm)
    try:
        voxel_cm = float(input("Enter voxel size (unit: cm): "))
    except ValueError:
        print("Invalid input, please enter a number.")
        return

    # Convert to meters (Open3D uses SI units: meters)
    voxel_size = voxel_cm / 100.0  # cm → m

    # Execute voxel downsampling
    print(f" Performing voxel downsampling (voxel size = {voxel_cm} cm = {voxel_size:.4f} m)...")
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    print(f"Downsampling completed, {len(downsampled_pcd.points)} points remaining.")

    # Save result
    base = os.path.splitext(input_path)[0]
    output_path = f"{base}_Voxel{voxel_cm}cm.ply"
    o3d.io.write_point_cloud(output_path, downsampled_pcd, write_ascii=False)
    print(f"Result saved to: {output_path}")


if __name__ == "__main__":
    main()
