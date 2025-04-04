import time
import numpy as np
import open3d as o3d


def create_partial_box(min_bound, max_bound):
    # Create and transform the box
    size = max_bound - min_bound
    box = o3d.geometry.TriangleMesh.create_box(*size).translate(min_bound)
    box.paint_uniform_color([0.93, 0.61, 0.42])
    box.compute_vertex_normals()
    box.compute_triangle_normals()
    triangles = np.asarray(box.triangles)
    triangles = triangles[:, ::-1]
    box.triangles = o3d.utility.Vector3iVector(triangles)
    return box


def get_joint_color_config():
    CLUSTERS = {
        0: [7, 9, 13, 14, 15, 20, 21, 25],  # Head
        1: [3, 12, 16, 17, 18, 19, 22, 23, 24],  # Right Arm
        2: [29, 38, 42, 43, 44, 45, 48, 49, 50],  # Left Leg
        3: [33, 35, 39, 40, 41, 46, 47, 51],  # Right Leg
        4: [6, 11, 26, 27, 28, 32, 37, 52, 53, 54, 1, 2, 5, 10, 31, 36],  # Torso
        5: [0, 4, 8, 30, 34],  # Left Arm
    }

    CLUSTER_COLORS = {
        0: [0.4627, 0.0510, 0.0078],
        1: [0.9725, 0.8745, 0.7804],
        2: [0.9333, 0.6157, 0.4196],
        3: [0.4627, 0.8863, 0.8314],
        4: [0.0, 0.4392, 0.5333],
        5: [0.5, 0.5, 0.5],
    }

    JOINT_COLORS = {}
    for cluster_id, joint_indices in CLUSTERS.items():
        for joint in joint_indices:
            JOINT_COLORS[joint] = CLUSTER_COLORS[cluster_id]

    return CLUSTERS, CLUSTER_COLORS, JOINT_COLORS


def create_room_from_sequence(sequence, margin=0.5):
    """
    Build a partial box around the entire pose sequence.
    """
    all_points = sequence.reshape(-1, 3)
    min_bound = np.min(all_points, axis=0) - margin
    max_bound = np.max(all_points, axis=0) + margin

    box_mesh = create_partial_box(min_bound, max_bound)
    box_mesh.translate((0, -min_bound[1] - 0.1, 0))
    return box_mesh


def create_sphere_at(radius=0.03, color=[0.5, 0.5, 0.5]):
    # Higher-resolution sphere for smoother look
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=6)
    sphere.compute_vertex_normals()
    sphere.paint_uniform_color(color)
    return sphere


def animate_dance(sequence, clusters, cluster_colors, delay=0.033):
    # Create window
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Dance Animation", width=800, height=600)

    # Create ballroom
    box_mesh = create_room_from_sequence(sequence)
    vis.add_geometry(box_mesh)

    _, num_joints, _ = sequence.shape

    # Create spheres
    joint_spheres = []
    for i in range(num_joints):
        color = cluster_colors.get(i, [0.5, 0.5, 0.5])
        sphere = create_sphere_at(color=color)
        # sphere = create_cube_at(color=color)
        joint_spheres.append(sphere)
        vis.add_geometry(sphere)

    vis.poll_events()
    vis.update_renderer()

    # Set camera
    ctr = vis.get_view_control()
    ctr.set_zoom(0.5)

    # Start Animation
    for frame in sequence:
        for sphere, pos in zip(joint_spheres, frame):
            sphere.translate(pos - sphere.get_center(), relative=True)
            vis.update_geometry(sphere)

        vis.poll_events()
        vis.update_renderer()
        time.sleep(delay)

    vis.destroy_window()


def rotate_data(data):
    R = np.array([[-1, 0, 0], [0, 0, -1], [0, 1, 0]])

    return np.einsum("ij,tkj->tki", R, data)


if __name__ == "__main__":
    DATA_PATH = "data/raw/mariel_knownbetter.npy"
    data = np.load(DATA_PATH)

    if data.shape[0] < data.shape[1]:
        data = data.transpose(1, 0, 2)
        data = rotate_data(data)

    CLUSTERS = {
        0: [7, 9, 13, 14, 15, 20, 21, 25],  # Head
        1: [3, 12, 16, 17, 18, 19, 22, 23, 24],  # Right Arm
        2: [29, 38, 42, 43, 44, 45, 48, 49, 50],  # Left Leg
        3: [33, 35, 39, 40, 41, 46, 47, 51],  # Right Leg
        4: [6, 11, 26, 27, 28, 32, 37, 52, 53, 54, 1, 2, 5, 10, 31, 36],  # Torso
        5: [0, 4, 8, 30, 34],  # Left Arm
    }

    CLUSTER_COLORS = {
        0: [0.4627, 0.0510, 0.0078],
        1: [0.9725, 0.8745, 0.7804],
        2: [0.9333, 0.6157, 0.4196],
        3: [0.4627, 0.8863, 0.8314],
        4: [0.0, 0.4392, 0.5333],
        5: [0.5, 0.5, 0.5],
    }

    num_joints = data.shape[1]
    JOINT_COLORS = {}
    for cluster_id, joint_indices in CLUSTERS.items():
        for joint in joint_indices:
            JOINT_COLORS[joint] = CLUSTER_COLORS[cluster_id]

    animate_dance(data, CLUSTERS, JOINT_COLORS)
