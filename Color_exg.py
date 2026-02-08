import plyfile
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog


def read_ply_as_array(ply_path):
    """Read .ply file and return (N, 9) NumPy array: [x, y, z, nx, ny, nz, red, green, blue]"""
    with open(ply_path, 'rb') as f:
        plydata = plyfile.PlyData.read(f)

    vertex = plydata['vertex'].data
    
    # Check if required fields exist
    required_fields = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'red', 'green', 'blue']
    for field in required_fields:
        if field not in vertex.dtype.names:
            raise ValueError(f"PLY file missing field: {field}")
    
    points = np.vstack([
        vertex['x'],
        vertex['y'],
        vertex['z'],
        vertex['nx'],
        vertex['ny'],
        vertex['nz'],
        vertex['red'].astype(np.float32),
        vertex['green'].astype(np.float32),
        vertex['blue'].astype(np.float32)
    ]).T  # shape: (N, 9)

    return points, vertex.dtype


def write_ply_from_array(output_path, points, original_dtype):
    """Write (N, 9) array back to .ply file, preserving original field types"""
    N = points.shape[0]
    if N == 0:
        print("Warning: No points after filtering, skipping save.")
        return
    
    vertex = np.empty(N, dtype=original_dtype)
    vertex['x'] = points[:, 0]
    vertex['y'] = points[:, 1]
    vertex['z'] = points[:, 2]
    vertex['nx'] = points[:, 3]
    vertex['ny'] = points[:, 4]
    vertex['nz'] = points[:, 5]
    vertex['red'] = points[:, 6].astype(original_dtype['red'])
    vertex['green'] = points[:, 7].astype(original_dtype['green'])
    vertex['blue'] = points[:, 8].astype(original_dtype['blue'])

    el = plyfile.PlyElement.describe(vertex, 'vertex')
    plyfile.PlyData([el], text=False).write(output_path)


class ColorPro:
    def __init__(self, processply):
        self.totalply = processply  # shape: (N, 9)

    def exgMethod(self, threshold):
        R = self.totalply[:, 6]
        G = self.totalply[:, 7]
        B = self.totalply[:, 8]
        exg = 2 * G - R - B
        mask = exg > threshold
        return self.totalply[mask]


def main():

    root = tk.Tk()
    root.withdraw()  # Hide main window

    input_path = filedialog.askopenfilename(
        title="Select a .ply point cloud file",
        filetypes=[("PLY files", "*.ply"), ("All files", "*.*")]
    )

    if not input_path:
        print("No file selected. Exiting program.")
        return

    if not os.path.exists(input_path):
        print(f"Error: File {input_path} does not exist.")
        return

    try:
        points, dtype = read_ply_as_array(input_path)
        print(f"Successfully loaded point cloud with {points.shape[0]} points.")
    except Exception as e:
        print(f"Failed to read PLY file: {e}")
        return

    # Execute ExG filtering (threshold fixed at 40, can be modified as needed)
    color_pro = ColorPro(points)
    
    filtered_points = color_pro.exgMethod(threshold=40)

    print(f"ExG filtering completed, {filtered_points.shape[0]} points retained.")

    # Generate output path
    base_name = os.path.splitext(input_path)[0]
    output_path = f"{base_name}_ExGFiltered.ply"

    # Save result
    write_ply_from_array(output_path, filtered_points, dtype)
    print(f"Result saved to: {output_path}")


if __name__ == "__main__":
    main()