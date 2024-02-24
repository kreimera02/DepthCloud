import open3d as o3d
import rosbag
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import matplotlib.pyplot as plt

bag = rosbag.Bag('drive-autonomy.bag')

aggregate_pcd = o3d.geometry.PointCloud()
voxel_size = 0.1

for topic, msg, t in bag.read_messages(topics=['/rtabmap/cloud_map']):
    # Convert the ROS message to a list of points
    points_list = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
    # Convert the list of points to a numpy array
    points_np = np.array(points_list)
    # Convert the numpy array to an Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)

    # Voxel Downsampling
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    aggregate_pcd += pcd
    # Optional: Visualize the downsampled point cloud with normals
    #o3d.visualization.draw_geometries([downsampled_pcd], point_show_normal=True)
    

bag.close()
o3d.visualization.draw_geometries([aggregate_pcd], point_show_normal=True)
aggregate_pcd = aggregate_pcd.voxel_down_sample(voxel_size=voxel_size)
o3d.visualization.draw_geometries([aggregate_pcd], point_show_normal=True)
agg_norm  = aggregate_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
# Continue with your processing using `downsampled_pcd` instead of `pcd`
# For example, you can proceed with segmentation on the downsampled point cloud

plane_model, inliers = aggregate_pcd.segment_plane(distance_threshold=0.02, ransac_n=3, num_iterations=1000)
ground = aggregate_pcd.select_by_index(inliers)
obstacles = aggregate_pcd.select_by_index(inliers, invert=True)

# Visualize Ground and Obstacles Separately
o3d.visualization.draw_geometries([ground], window_name="Ground Plane")
o3d.visualization.draw_geometries([obstacles], window_name="Obstacles")

# Convert the obstacles to a point cloud with only XYZ for clustering
obstacles_xyz = np.asarray(obstacles.points)
obstacles_pcd_xyz = o3d.geometry.PointCloud()
obstacles_pcd_xyz.points = o3d.utility.Vector3dVector(obstacles_xyz)

# DBSCAN Clustering to segment obstacles
eps = 0.05
min_points = 10

print("Clustering with eps=%.3f and min_points=%d" % (eps, min_points))

labels = np.array(obstacles_pcd_xyz.cluster_dbscan(eps, min_points, print_progress=True))

# # Create a color map for visualizing clusters
max_label = labels.max()
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0  # Setting outliers to black
obstacles_pcd_xyz.colors = o3d.utility.Vector3dVector(colors[:, :3])

# Visualize the clustered obstacles
o3d.visualization.draw_geometries([obstacles_pcd_xyz], window_name="Clustered Obstacles")

def calculate_centroid(points):
    return np.mean(points, axis=0)

# Function to calculate distance from all points in a cluster to the centroid
def calculate_distances_to_centroid(points, centroid):
    distances = np.sqrt(np.sum((points - centroid)**2, axis=1))
    return distances

# Identify potential craters based on cluster characteristics
crater_candidates = []
for i in range(max_label + 1):
    cluster_points = obstacles_pcd_xyz.points[np.where(labels == i)]
    if len(cluster_points) == 0:
        continue

    centroid = calculate_centroid(cluster_points)
    distances = calculate_distances_to_centroid(cluster_points, centroid)
    mean_distance = np.mean(distances)
    # Heuristic criteria for identifying craters: adjust these thresholds as necessary
    if mean_distance > 0.5 and mean_distance < 2.0:  # Example thresholds for "crater size"
        crater_candidates.append(i)

# Visualize potential crater candidates
for crater_id in crater_candidates:
    print(f"Cluster {crater_id} might be a crater.")
    crater_points = obstacles_pcd_xyz.points[np.where(labels == crater_id)]
    crater_pcd = o3d.geometry.PointCloud()
    crater_pcd.points = o3d.utility.Vector3dVector(crater_points)
    o3d.visualization.draw_geometries([crater_pcd], window_name=f"Crater Candidate {crater_id}")