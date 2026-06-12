import numpy as np
import logging
import joblib
import laspy
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy.stats import binned_statistic_2d
from scipy.ndimage import binary_dilation, label, distance_transform_edt
from tqdm import tqdm
from scipy.stats import mode
import scipy.ndimage as ndimage

from point_cloud_classifier.helper import getSingleIDperGroup, get_bounds, get_data_summary, get_accuracy, get_f1, get_precision, get_recall

from point_cloud_classifier.constants import SELECTED_FEATURE_NAMES, SEED, CLASSIFICATION_MAP, TILE_SIZE

from scipy.spatial import cKDTree
import open3d as o3d
import pandas as pd

logger = logging.getLogger(__name__)

class PointCloudClassifier:
    def __init__(self, tile_size: float = TILE_SIZE, raster_resolution: float = 0.5, binary_ground_classifier = joblib.load("./trained_models/ground/RandomForestClassifier_97_94_97_91.pkl"), binary_vegetation_classifier = joblib.load("./trained_models/vegetation/vegetation_RandomForestClassifier_94_95_97_93.pkl"), binary_roof_classifier = joblib.load("./trained_models/roof_facade/Building roofs_RandomForestClassifier_98_97_95_99.pkl"), binary_facade_classifier = joblib.load("./trained_models/facade/Building facades_RandomForestClassifier_95_85_76_97.pkl"), binary_roof_structure_classifier: ClassifierMixin = joblib.load("./trained_models/roof_structure/Roof structures_RandomForestClassifier_100_62_50_80.pkl"), binary_car_classifier: ClassifierMixin = joblib.load("./trained_models/car/RandomForestClassifier_99_64_48_96.pkl")):
        self.binary_ground_classifier = binary_ground_classifier
        self.binary_vegetation_classifier = binary_vegetation_classifier
        self.binary_roof_classifier = binary_roof_classifier
        self.binary_facade_classifier = binary_facade_classifier
        self.raster_resolution = raster_resolution
        self.binary_roof_structure_classifier = binary_roof_structure_classifier
        self.binary_car_classifier = binary_car_classifier
        self.patch_size = tile_size

    def predict(self, data: np.ndarray, points: np.ndarray, nsquared_patches:int = 1) -> np.ndarray:
        labels = np.zeros(len(points), dtype=int)

        patches = self.__get_patches(points, nsquared_patches)
        self.patch_size /= nsquared_patches

        for patch in tqdm(patches, desc="Prediction over patches", unit="patch"):
            mask = np.zeros(len(data), dtype=bool)
            mask[patch] = True
            patch_mask = mask
            labels[patch[self.classify_ground_points(data[mask]).astype(bool)]] = 2

            mask = mask & (labels == 0)
            indicies = np.where(mask)[0]
            labels[indicies[self.classify_vegetation_points(points[mask], data[mask]).astype(bool)]] = 3

            mask = mask & (labels == 0)
            indicies = np.where(mask)[0]
            labels[indicies[self.classify_roof_points(points[mask], data[mask]).astype(bool)]] = 6

            mask = mask & (labels == 0)
            indicies = np.where(mask)[0]
            labels[indicies[self.classify_facade_points(points[mask], data[mask], points[patch_mask & (labels == 6)])]] = 22

            mask = mask & (labels == 0)
            indicies = np.where(mask)[0]
            labels[indicies[self.classify_roof_structure_points(points[mask], points[patch_mask & (labels == 6)])]] = 26

        #labels = self.__smooth_prediction(points, labels)
        return labels
            

    def classify_ground_points(self, data: np.ndarray) -> np.ndarray:
        if not self.binary_ground_classifier:
            raise ValueError("The ground classifier is not trained")
        return self.binary_ground_classifier.predict(data)
    
    def classify_vegetation_points(self, points: np.ndarray, data: np.ndarray, voxel_size: float = 0.625, probability_blurr_n_neighbours: int = 30, probability_blurr_sigma: float = 6, certainity_threshold: float = 0.86, max_dilation: int = 6, label_structure: np.ndarray = np.ones((3,3,3), dtype=int), max_tip_size: int = 40, min_cluster_size: int = 10, label_blurr_n_neighbours:int = 10, label_blurr_sigma: float = 4) -> np.ndarray:
        if not self.binary_vegetation_classifier:
            raise ValueError("The vegetation classifier is not trained")
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        probabilities = self.binary_vegetation_classifier.predict_proba(data)[:,1]

        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)

        origin = voxel_grid.get_min_bound()
        voxel_indices = np.floor((points - origin) / voxel_size).astype(int)

        df = pd.DataFrame(voxel_indices, columns=["vx", "vy", "vz"])
        df["probability"] = self.__featureBlurr(points, probabilities, K_NEIGHBORS=probability_blurr_n_neighbours, SIGMA=probability_blurr_sigma)
        df["predicted"] = np.round(probabilities)
        df["certainity"] = df["probability"] > certainity_threshold

        voxel_stats = (df.groupby(["vx", "vy", "vz"])["certainity"].agg(count="count", mean_predicted="mean").reset_index())

        df = df.merge(voxel_stats, on=["vx", "vy", "vz"], how="left")
        df['is_vegetation'] = np.round(df['mean_predicted']).astype(bool) & df['predicted'].astype(bool)
        

        min_vx, min_vy, min_vz = voxel_stats["vx"].min(), voxel_stats["vy"].min(), voxel_stats["vz"].min()
        max_vx, max_vy, max_vz = voxel_stats["vx"].max(), voxel_stats["vy"].max(), voxel_stats["vz"].max()
        grid_shape = (max_vx - min_vx + 1, max_vy - min_vy + 1, max_vz - min_vz + 1)

        vegetation_grid = np.zeros(grid_shape, dtype=bool)
        non_vegetation_grid = np.zeros(grid_shape, dtype=bool)

        veg_voxels = voxel_stats[np.round(voxel_stats["mean_predicted"]) == 1]
        non_veg_voxels = voxel_stats[np.round(voxel_stats["mean_predicted"]) == 0]

        grid_vx = veg_voxels["vx"].values - min_vx
        grid_vy = veg_voxels["vy"].values - min_vy
        grid_vz = veg_voxels["vz"].values - min_vz
        vegetation_grid[grid_vx, grid_vy, grid_vz] = True

        grid_vx = non_veg_voxels["vx"].values - min_vx
        grid_vy = non_veg_voxels["vy"].values - min_vy
        grid_vz = non_veg_voxels["vz"].values - min_vz
        non_vegetation_grid[grid_vx, grid_vy, grid_vz] = True

        point_vx = df["vx"].values - min_vx
        point_vy = df["vy"].values - min_vy
        point_vz = df["vz"].values - min_vz

        dilated_grid = binary_dilation(vegetation_grid, iterations=max_dilation)

        candidate_voxel =  np.bitwise_and(dilated_grid, non_vegetation_grid)
        labeled_grid, _ = label(non_vegetation_grid, structure=label_structure)
        cluster_sizes = np.bincount(labeled_grid.ravel())

        small_clusters_mask = cluster_sizes <= max_tip_size
        isolated_non_veg_grid = small_clusters_mask[labeled_grid] & (labeled_grid > 0)
        contaminated_voxels = isolated_non_veg_grid & candidate_voxel

        df["contaminated_voxels"] = contaminated_voxels[point_vx, point_vy, point_vz]
        df["is_vegetation_extended"] = df["is_vegetation"] | df['contaminated_voxels']

        extended_veg_grid = np.zeros(grid_shape, dtype=bool)
        veg_ext_points = df[df["is_vegetation_extended"] == True]

        ext_vx = veg_ext_points["vx"].values - min_vx
        ext_vy = veg_ext_points["vy"].values - min_vy
        ext_vz = veg_ext_points["vz"].values - min_vz
        extended_veg_grid[ext_vx, ext_vy, ext_vz] = True

        labeled_veg_grid, _ = label(extended_veg_grid, structure=label_structure)
        veg_cluster_sizes = np.bincount(labeled_veg_grid.ravel())
 
        large_clusters_mask = veg_cluster_sizes >= min_cluster_size
        cleaned_veg_grid = large_clusters_mask[labeled_veg_grid] & (labeled_veg_grid > 0)

        df["is_vegetation_cleaned"] = np.round(self.__featureBlurr(points, cleaned_veg_grid[point_vx, point_vy, point_vz], K_NEIGHBORS=label_blurr_n_neighbours, SIGMA=label_blurr_sigma))

        return np.array(df["is_vegetation_cleaned"], dtype=np.float32)

    def classify_roof_points(self, points: np.ndarray, data: np.ndarray, max_planes: int = 2000, radius_normal_determination: float = 1.5, max_nn_normal_determination: int = 30, min_cluster_size: int = 100, min_dbscan_cluster_size: int = 30, n_phi_bins: int = 6, n_theta_bins: int = 6, ransac_distance_threshold: float = 0.1, inlier_distance_threshold: float = 0.4, dbscan_distance_threshold: float = 0.5, ransac_n_iter: int = 2000, ransac_n: int = 3, fraction_correctly_classified: float = 0.5, normal_z_threshold: float = 90, add_mask: bool = False, mask: np.ndarray | None = None, initial_dbscan: bool = False, min_z: float = 2.25):
        if not self.binary_roof_classifier:
            raise ValueError("The roof classifier is not trained")

        pcd_full = o3d.geometry.PointCloud()
        pcd_full.points = o3d.utility.Vector3dVector(points)

        pcd_full.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal_determination, max_nn=max_nn_normal_determination))

        normals = np.asarray(pcd_full.normals)

        predictions = self.binary_roof_classifier.predict(data)
        true_prediction = (predictions == 1)
        remaining = np.ones(len(points), dtype=bool)
        classification_array = np.zeros(len(points), dtype=int)

        is_filtered_normal = (np.abs(normals[:,2]) <= np.sin(np.radians(normal_z_threshold)))

        for i in tqdm(range(max_planes), desc="Plane fitting", unit="plane", leave=False):
            
            used_points = true_prediction & remaining & is_filtered_normal

            if add_mask:
                used_points = used_points & mask

            if np.count_nonzero(used_points) < min_cluster_size:
                break

            normals_bins = self.__spherical_histogram(normals[used_points], n_cos_phi=n_phi_bins, n_theta=n_theta_bins)
            vals, cnts = np.unique(normals_bins, return_counts=True)

            best_bin = vals[cnts.argmax()]
            used_filtered = np.zeros_like(used_points, dtype=bool)

            used_filtered[used_points] = (np.isin(normals_bins,best_bin))

            if initial_dbscan:
                pcd_seeds_filtered = o3d.geometry.PointCloud()
                pcd_seeds_filtered.points = o3d.utility.Vector3dVector(points[used_filtered])
                
                seed_labels = np.array(pcd_seeds_filtered.cluster_dbscan(eps=dbscan_distance_threshold, min_points=min_dbscan_cluster_size, print_progress=False))
                
                if len(seed_labels) == 0 or np.all(seed_labels == -1):
                    remaining[used_filtered] = False
                    continue
                    
                unique_seed_labels, seed_counts = np.unique(seed_labels[seed_labels != -1], return_counts=True)
                best_seed_label = unique_seed_labels[seed_counts.argmax()]
                
                idx_in_seed_cluster = (seed_labels == best_seed_label)
                global_filtered_indices = np.where(used_filtered)[0]
                used_filtered = global_filtered_indices[idx_in_seed_cluster]
                
            pcd_seeds = o3d.geometry.PointCloud()
            pcd_seeds.points = o3d.utility.Vector3dVector(points[used_filtered])
            
            equation, _ = pcd_seeds.segment_plane(distance_threshold=ransac_distance_threshold, ransac_n=ransac_n, num_iterations=ransac_n_iter)

            a, b, c, d = equation

            distances = np.abs(a * (points[:, 0]) + b * points[:, 1] + c * points[:, 2] + d)
            inliers_plane_mask = (distances <= inlier_distance_threshold) & remaining
        
            xyz_plane = points[inliers_plane_mask]
            pcd_plane = o3d.geometry.PointCloud()
            pcd_plane.points = o3d.utility.Vector3dVector(xyz_plane)
            labels = np.array(pcd_plane.cluster_dbscan(eps=dbscan_distance_threshold, min_points=min_dbscan_cluster_size, print_progress=False))

            if len(labels) == 0 or np.all(labels == -1):
                remaining[used_filtered] = False
                continue
            
            unique_labels, label_counts = np.unique(labels[labels != -1], return_counts=True)
            best_label = unique_labels[label_counts.argmax()]
            
            idx_in_cluster = (labels == best_label)

            global_plane_indices = np.where(inliers_plane_mask)[0]
            isolated_roof_global = global_plane_indices[idx_in_cluster]

            if (predictions[isolated_roof_global].mean() > fraction_correctly_classified) & (len(isolated_roof_global) > min_cluster_size):
                classification_array[isolated_roof_global] = True

            remaining[isolated_roof_global] = False 
        

        classification_array[(data[:, SELECTED_FEATURE_NAMES.index("z_norm")] < min_z)] = False
        return classification_array

    def classify_facade_points(self, points: np.ndarray, data: np.ndarray, roof_points: np.ndarray, max_planes: int = 2000, radius_normal_determination: float = 1.5, max_nn_normal_determination: int = 30, min_cluster_size: int = 15, min_dbscan_cluster_size: int = 5, n_phi_bins: int = 6, n_theta_bins: int = 6, ransac_distance_threshold: float = 0.6, inlier_distance_threshold: float = 1.0, dbscan_distance_threshold: float = 1.3, ransac_n_iter: int = 2000, ransac_n: int = 3, fraction_correctly_classified: float = 0.1, normal_z_threshold: float = 4, min_z: float = 0.0, filter_height: bool = True) -> np.ndarray:


        col_indicies, row_indicies = self.__get_raster_indicies(points)

        raster_mask = self.__get_raster(roof_points)
        is_under_raster = raster_mask[row_indicies, col_indicies]

        classification_array = self.classify_roof_points(points = points, data = data, add_mask=True, mask=is_under_raster, initial_dbscan = True, max_planes = max_planes, radius_normal_determination = radius_normal_determination, max_nn_normal_determination=max_nn_normal_determination, min_cluster_size=min_cluster_size, min_dbscan_cluster_size=min_dbscan_cluster_size, n_phi_bins=n_phi_bins, n_theta_bins=n_theta_bins, ransac_distance_threshold=ransac_distance_threshold, inlier_distance_threshold=inlier_distance_threshold, dbscan_distance_threshold=dbscan_distance_threshold, ransac_n_iter=ransac_n_iter, ransac_n=ransac_n, fraction_correctly_classified=fraction_correctly_classified, normal_z_threshold=normal_z_threshold, min_z=min_z)

        classification_array = (classification_array & is_under_raster).astype(bool)

        if filter_height:
            dem = self.__get_dem(roof_points, statistic = 'max')

            alt_x_indices, alt_y_indices = self.__get_raster_indicies(points)
            nearest_roof_alt = dem[alt_y_indices, alt_x_indices]
            is_under_roof = points[:,2] < nearest_roof_alt
            classification_array = classification_array & is_under_roof

        return classification_array

    def classify_roof_structure_points(self, points: np.ndarray, roof_points: np.ndarray, closing_iterations: int = 3):
        col_indicies, row_indicies = self.__get_raster_indicies(points)

        structure = ndimage.generate_binary_structure(2, 2)
        raster_mask = self.__get_raster(roof_points) 
        raster_mask = ndimage.binary_closing(raster_mask, structure=structure, iterations=closing_iterations).astype(int)
        is_under_raster = raster_mask[row_indicies, col_indicies]

        dem = self.__get_dem(roof_points, statistic = 'min')

        nearest_roof_alt = dem[row_indicies, col_indicies]
        is_over_roof = points[:,2] > nearest_roof_alt
        classification_array = is_under_raster & is_over_roof

        return classification_array.astype(bool)

    def set_binary_ground_classifier(self, model: ClassifierMixin):
        self.binary_ground_classifier = model
    
    def predict_binary_classifier(self, data: np.ndarray, classifier_name: str = "binary_ground_classifier"):
        if not hasattr(self, classifier_name):
            raise AttributeError(f"'{self.__class__.__name__}' has not classifier named '{classifier_name}'")
        else:
            classifier = getattr(self, classifier_name)
        return classifier.predict(data)

    def set_binary_vegetation_classifier(self, model: ClassifierMixin):
        self.binary_vegetation_classifier = model

    def fit_classifier(self, data: np.ndarray, labels: np.ndarray, classifier_name: str = "binary_ground_classifier", model: ClassifierMixin = RandomForestClassifier(n_jobs=-1, class_weight="balanced")):
        if not hasattr(self, classifier_name):
            raise AttributeError(f"'{self.__class__.__name__}' has not classifier named '{classifier_name}'")
        
        logger.info("Starting model fitting with %s...", model.__class__.__name__)
        classifier = model.fit(data, labels)
        logger.info("Model successfully fitted.")
        setattr(self, classifier_name, classifier)

        return classifier
    
    def test_binary_classifier(self, test_data: np.ndarray, test_logits: np.ndarray, classifier: ClassifierMixin | str = "binary_ground_classifier", save_model: bool = False, save_dir: str = "ground", batchsize = 100000):
        
        if isinstance(classifier, str):
            if not hasattr(self, classifier):
                raise AttributeError(f"'{self.__class__.__name__}' has not classifier named '{classifier}'")
            classifier = getattr(self, classifier)


        tp_total, fp_total, fn_total, tn_total = 0, 0, 0, 0
        N = len(test_data)

        for i in np.arange(0, N, batchsize):
            Predicted = classifier.predict(test_data[i:i+batchsize])
            
            acc = get_accuracy(Predicted, test_logits[i:i+batchsize])

            logger.debug("Accuracy: %.2f %% | Progression: %.2f %%", acc * 100, (i / N) * 100)

            tp_total += np.sum((Predicted == 1) & (test_logits[i:i+batchsize] == 1))
            fp_total += np.sum((Predicted == 1) & (test_logits[i:i+batchsize] == 0))
            fn_total += np.sum((Predicted == 0) & (test_logits[i:i+batchsize] == 1))
            tn_total += np.sum((Predicted == 0) & (test_logits[i:i+batchsize] == 0))

        global_acc = (tp_total + tn_total) / N
        global_precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
        global_recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
        global_f1 = 2 * (global_precision * global_recall) / (global_precision + global_recall) if (global_precision + global_recall) > 0 else 0

        logger.info("="*50)
        logger.info(f"Résultats Globaux :")
        logger.info(f" -> Accuracy  : {global_acc * 100:.2f} %")
        logger.info(f" -> F1-Score  : {global_f1:.4f}")
        logger.info(f" -> Recall    : {global_recall:.4f}")
        logger.info(f" -> Precision : {global_precision:.4f}")
        logger.info("="*50)

        if save_model:
            model_path = f"./trained_models/{save_dir}/{classifier.__class__.__name__}_{round(global_acc * 100)}_{round(global_f1 * 100)}_{round(global_recall * 100)}_{round(global_precision * 100)}.pkl"
            joblib.dump(self.binary_ground_classifier, model_path)
            logger.info("Model saved to %s", model_path)

    def __featureBlurr(self, points: np.ndarray, feature: np.ndarray, tree: cKDTree = None, K_NEIGHBORS: int = 100, SIGMA: float = 1):

        if tree is None:
            tree = cKDTree(points)
        distances, neighbor_indices = tree.query(points, k=K_NEIGHBORS, workers=-1)

        weights = np.exp(-(distances ** 2) / (2 * (SIGMA ** 2)))

        weights_sum = np.sum(weights, axis=1, keepdims=True)
        weights_normalized = weights / (weights_sum + 1e-12)

        neighbor_values = feature[neighbor_indices]
        return np.sum(neighbor_values * weights_normalized, axis=1)

    def __spherical_histogram(self, normals: np.ndarray, n_cos_phi: int = 8, n_theta: int = 8) -> np.ndarray:
        x, y, z = normals[:, 0], normals[:, 1], normals[:, 2]
        theta = np.arctan2(y, x)
        cos_phi = z

        cos_phi_bins = np.linspace(-1.0, 1.0, n_cos_phi + 1)
        theta_bins = np.linspace(-np.pi, np.pi, n_theta + 1)
        
        cos_phi_idx = np.digitize(cos_phi, cos_phi_bins[:-1]) - 1
        theta_idx = np.digitize(theta, theta_bins[:-1]) - 1
        
        cos_phi_idx = np.clip(cos_phi_idx, 0, n_cos_phi - 1)
        theta_idx = np.clip(theta_idx, 0, n_theta - 1)
        
        bin_ids = cos_phi_idx * n_theta + theta_idx
        
        return bin_ids
    
    def __get_raster(self, points: np.ndarray) -> np.ndarray:
        
        grid_width = int(self.patch_size / self.raster_resolution)
        grid_height = int(self.patch_size / self.raster_resolution)
        grid_mask = np.zeros((grid_height, grid_width), dtype=bool)
        if len(points) == 0:
            return grid_mask
        
        x_indicies, y_indicies = self.__get_raster_indicies(points)
        grid_mask[y_indicies, x_indicies] = True
        
        return grid_mask

    def __get_patches(self, points: np.ndarray, nsquared_patches: int) -> list[np.ndarray]:
        x = points[:, 0]
        y = points[:, 1]

        x_edges = np.linspace(x.min(), x.max(), nsquared_patches + 1)
        y_edges = np.linspace(y.min(), y.max(), nsquared_patches + 1)

        patches = []

        for i in range(nsquared_patches):
            for j in range(nsquared_patches):
                mask = ((x >= x_edges[i]) & (x < x_edges[i + 1] if i < nsquared_patches - 1 else x <= x_edges[i + 1]) & (y >= y_edges[j]) & (y < y_edges[j + 1] if j < nsquared_patches - 1 else y <= y_edges[j + 1]))
                patches.append(np.where(mask)[0])

        return patches

    def __smooth_prediction(self, points: np.ndarray, labels: np.ndarray, k_neighbors: int = 20) -> np.ndarray:
        
        logger.info("Smoothing the predictions...")
        tree = cKDTree(points)
        _, indices = tree.query(points, k=k_neighbors)
        
        neighbor_labels = labels[indices]
        smoothed_labels, _ = mode(neighbor_labels, axis=1, keepdims=False)
        return smoothed_labels.astype(int)
    
    def __get_dem(self, points: np.ndarray, statistic: str = "min") -> np.ndarray:

        if len(points) == 0:
            grid_size = int(np.ceil(self.patch_size / self.raster_resolution))
            return np.zeros((grid_size, grid_size), dtype=np.float64)

        x = points[:, 0]
        y = points[:, 1]
        altitudes = points[:, 2]

        x_min = np.floor(round(x.min()) / self.patch_size) * self.patch_size
        y_min = np.floor(round(y.min()) / self.patch_size) * self.patch_size
        y_max = y_min + self.patch_size
        x_max = x_min + self.patch_size


        x_edges = np.arange(x_min, x_max + self.raster_resolution, self.raster_resolution)
        y_edges = np.arange(y_min, y_max + self.raster_resolution, self.raster_resolution)

        dem, _, _, _ = binned_statistic_2d(y, x, altitudes, statistic=f'{statistic}', bins=[y_edges, x_edges])

        nan_mask = np.isnan(dem)

        _, indices = distance_transform_edt(nan_mask, return_distances=True, return_indices=True)

        dem =  dem[tuple(indices)]
        return dem

    def __get_raster_indicies(self, points):

        if len(points) == 0:
            return np.array([]), np.array([])
        
        x = points[:,0]
        y = points[:, 1]

        x_min = np.floor(round(x.min()) / self.patch_size) * self.patch_size
        y_min = np.floor(round(y.min()) / self.patch_size) * self.patch_size

        grid_width = int(np.ceil(self.patch_size / self.raster_resolution))
        grid_height = int(np.ceil(self.patch_size / self.raster_resolution))

        x_indices = np.floor((x - x_min) / self.raster_resolution).astype(np.int32)
        y_indices = np.floor((y - y_min) / self.raster_resolution).astype(np.int32)

        x_indices = np.clip(x_indices, 0, grid_width - 1)
        y_indices = np.clip(y_indices, 0, grid_height - 1)
        return x_indices, y_indices

class DataClassifierFormat:
    def __init__(self):
        pass
    
    @staticmethod
    def load_data(point_cloud_paths: str | list[str], true_id: int, features: list[str] = SELECTED_FEATURE_NAMES, fraction_of_dataset: float = 0.001, data_overview: bool = False, is_random: bool = True):
        np.random.seed(SEED)
        data: np.ndarray = None
        logits: np.ndarray = None
        points: np.ndarray = None

        if isinstance(point_cloud_paths, str):
            point_cloud_paths = [point_cloud_paths]
        
        for point_cloud_path in tqdm(point_cloud_paths, desc="Loading dataset", unit="pointcloud"):
            if not point_cloud_path.endswith((".laz", ".las")):
                continue
            
            pc: laspy.LasData = getSingleIDperGroup(laspy.read(point_cloud_path))
            N: int = pc.header.point_count
            if is_random:
                pc = pc[np.random.choice(N, size=int(N * fraction_of_dataset), replace=False)]
            else:
                pc = pc[:int(N * fraction_of_dataset)]

            if logits is None:
                logits: np.ndarray = np.isin(pc.classification, true_id)
            else:
                logits = np.concat((logits, np.isin(pc.classification, true_id)))

            if points is None:
                points = pc.xyz
            else:
                points = np.vstack((points, pc.xyz), dtype=np.float32)

            feature_list = [(getattr(pc, f) - bounds[0]) / (bounds[1] - bounds[0]) if (bounds := get_bounds(f)) is not None else getattr(pc, f) for f in features]
            if data is None:
                data: np.ndarray = np.stack(feature_list, axis=1)
            else:
                data = np.vstack((data, np.stack(feature_list, axis=1)))

        if data_overview:
            get_data_summary(logits, {i: CLASSIFICATION_MAP[k] for i, k in enumerate(np.unique([0] + true_id)) if k in CLASSIFICATION_MAP})

        return points, data, logits

    @staticmethod
    def split_train_test(*arrays: np.ndarray, test_size: float = 0.2, random_state: int = SEED):
            return train_test_split(*arrays, test_size=test_size, random_state=random_state)