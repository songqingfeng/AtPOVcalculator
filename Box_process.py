import open3d as o3d
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog


class BoxPro:
    def __init__(self, xl: float, xr: float, yl: float, yr: float, zu: float, zd: float):
        self.xl = xl
        self.xr = xr
        self.yl = yl
        self.yr = yr
        self.zu = zu
        self.zd = zd

    def create_cube_edges(self):
        """Create a cube wireframe for visualization"""
        vertices = np.array([[-self.xl, -self.yl, self.zd],
                             [self.xr, -self.yl, self.zd],
                             [self.xr, self.yr, self.zd],
                             [-self.xl, self.yr, self.zd],
                             [-self.xl, -self.yl, self.zu],
                             [self.xr, -self.yl, self.zu],
                             [self.xr, self.yr, self.zu],
                             [-self.xl, self.yr, self.zu]])

        edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0],
                          [4, 5], [5, 6], [6, 7], [7, 4],
                          [0, 4], [1, 5], [2, 6], [3, 7]])

        cube_edges = o3d.geometry.LineSet()
        cube_edges.points = o3d.utility.Vector3dVector(vertices)
        cube_edges.lines = o3d.utility.Vector2iVector(edges)
        return cube_edges

    def crop_point_cloud(self, point_cloud):
        """Crop point cloud based on bounding box range"""
        points = np.asarray(point_cloud.points)
        mask = (
            (points[:, 0] >= -self.xl) & (points[:, 0] <= self.xr) &
            (points[:, 1] >= -self.yl) & (points[:, 1] <= self.yr) &
            (points[:, 2] >= self.zd) & (points[:, 2] <= self.zu)
        )
        cropped_pcd = point_cloud.select_by_index(np.where(mask)[0])
        return cropped_pcd

    def check_box(self, point_cloud):
        """Visualize original point cloud and cropping box"""
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1200, height=800)
        vis.get_render_option().background_color = np.asarray([1.0, 1.0, 1.0])

        vis.add_geometry(point_cloud)
        vis.add_geometry(self.create_cube_edges())

        vis.run()
        vis.destroy_window()


def main():
    root = tk.Tk()
    root.withdraw()

    # Open file selection dialog
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

    # Read point cloud
    pcd = o3d.io.read_point_cloud(input_path)
    if len(pcd.points) == 0:
        print("Error: Point cloud is empty.")
        return

    # === Cropping parameters ===
    XL, XR = 15.0, 15.0   # x ∈ [-15, 15]
    YL, YR = 15.0, 15.0   # y ∈ [-15, 15]
    ZD, ZU = 20.0, 50.0   # z ∈ [20, 50]

    box_pro = BoxPro(xl=XL, xr=XR, yl=YL, yr=YR, zu=ZU, zd=ZD)

    # Enable visualization by default
    print("Displaying original point cloud and cropping box. Close window to continue...")
    box_pro.check_box(pcd)

    # Crop point cloud
    cropped_pcd = box_pro.crop_point_cloud(pcd)

    base_name = os.path.splitext(input_path)[0]
    output_path = f"{base_name}_Boxrocess.ply"

    # Save result
    o3d.io.write_point_cloud(output_path, cropped_pcd, write_ascii=False, compressed=False)
    print(f"Cropping completed! Result saved to: {output_path}")


if __name__ == "__main__":
    main()