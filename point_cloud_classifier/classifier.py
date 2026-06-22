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
import os

from point_cloud_classifier.helper import getSingleIDperGroup, get_bounds, get_data_summary

from point_cloud_classifier.constants import SELECTED_FEATURE_NAMES, SEED, CLASSIFICATION_MAP, TILE_SIZE

from scipy.spatial import cKDTree
import open3d as o3d
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from point_cloud_classifier.car_cnn import Trainer, CarNet
from point_cloud_classifier.loss_function import BCEDiceLoss, BCEDiceWeightedLoss

from typing import Union

logger = logging.getLogger(__name__)

class PointCloudClassifier:
    def __init__(self, tile_size: float = TILE_SIZE, raster_resolution: float = 0.5, car_raster_resolution: float = 0.25, 
                 binary_ground_classifier = None, 
                 binary_vegetation_classifier = None,
                 binary_roof_classifier = None, 
                 binary_facade_classifier = None, 
                 car_model_path: str = "./trained_models/car/carNet_AdamW_lr1e-4_0.25m.pth"):
        
        self.binary_ground_classifier = binary_ground_classifier or joblib.load("./trained_models/ground/RandomForestClassifier_97_94_97_91.pkl")
        self.binary_vegetation_classifier = binary_vegetation_classifier or joblib.load("./trained_models/vegetation/vegetation_RandomForestClassifier_94_95_97_93.pkl")
        self.binary_roof_classifier = binary_roof_classifier or joblib.load("./trained_models/roof_facade/Building roofs_RandomForestClassifier_98_97_95_99.pkl")
        self.binary_facade_classifier = binary_facade_classifier or joblib.load("./trained_models/facade/Building facades_RandomForestClassifier_95_85_76_97.pkl")
        self.raster_resolution = raster_resolution
        self.car_raster_resolution = car_raster_resolution
        self.tile_size = tile_size
        self.patch_size = self.tile_size

        self.set_car_model(car_model_path)

    def predict(self, data: np.ndarray, points: np.ndarray, nsquared_patches:int = 1) -> np.ndarray:
        labels = np.zeros(len(points), dtype=int)

        patches = self.__get_patches(points, nsquared_patches)
        self.patch_size = self.tile_size/nsquared_patches

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

            mask = mask & (labels == 0)
            indicies = np.where(mask)[0]
            labels[indicies[self.classify_car_points(points[mask], data[mask])]] = 21

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


        col_indicies, row_indicies = self.__get_raster_indicies(points, self.raster_resolution)

        raster_mask = self.__get_raster(roof_points, self.raster_resolution)
        is_under_raster = raster_mask[row_indicies, col_indicies]

        classification_array = self.classify_roof_points(points = points, data = data, add_mask=True, mask=is_under_raster, initial_dbscan = True, max_planes = max_planes, radius_normal_determination = radius_normal_determination, max_nn_normal_determination=max_nn_normal_determination, min_cluster_size=min_cluster_size, min_dbscan_cluster_size=min_dbscan_cluster_size, n_phi_bins=n_phi_bins, n_theta_bins=n_theta_bins, ransac_distance_threshold=ransac_distance_threshold, inlier_distance_threshold=inlier_distance_threshold, dbscan_distance_threshold=dbscan_distance_threshold, ransac_n_iter=ransac_n_iter, ransac_n=ransac_n, fraction_correctly_classified=fraction_correctly_classified, normal_z_threshold=normal_z_threshold, min_z=min_z)

        classification_array = (classification_array & is_under_raster).astype(bool)

        if filter_height:
            dem = self.__get_dem(roof_points, statistic = 'max')

            alt_x_indices, alt_y_indices = self.__get_raster_indicies(points, self.raster_resolution)
            nearest_roof_alt = dem[alt_y_indices, alt_x_indices]
            is_under_roof = points[:,2] < nearest_roof_alt
            classification_array = classification_array & is_under_roof

        return classification_array

    def classify_roof_structure_points(self, points: np.ndarray, roof_points: np.ndarray, closing_iterations: int = 3):
        col_indicies, row_indicies = self.__get_raster_indicies(points, self.raster_resolution)

        structure = ndimage.generate_binary_structure(2, 2)
        raster_mask = self.__get_raster(roof_points, self.raster_resolution) 
        raster_mask = ndimage.binary_closing(raster_mask, structure=structure, iterations=closing_iterations).astype(int)
        is_under_raster = raster_mask[row_indicies, col_indicies]

        dem = self.__get_dem(roof_points, statistic = 'min')

        nearest_roof_alt = dem[row_indicies, col_indicies]
        is_over_roof = points[:,2] > nearest_roof_alt
        classification_array = is_under_raster & is_over_roof

        return classification_array.astype(bool)

    def classify_car_points(self, points: np.ndarray, data: np.ndarray, patch_size: int = 64, stride: int = 32, threshold: float = 0.5):
        raster_mask = self.predict_car_model(points, data, self.car_raster_resolution, patch_size, stride) >= threshold
        col_indicies, row_indicies = self.__get_raster_indicies(points, self.car_raster_resolution)
        is_under_raster = raster_mask[row_indicies, col_indicies]
        return is_under_raster

    def fit_classifier(self, data: np.ndarray, labels: np.ndarray, classifier_name: str = "binary_ground_classifier", model: ClassifierMixin = RandomForestClassifier(n_jobs=-1, class_weight="balanced")):
        if not hasattr(self, classifier_name):
            raise AttributeError(f"'{self.__class__.__name__}' has not classifier named '{classifier_name}'")
        
        logger.info("Starting model fitting with %s...", model.__class__.__name__)
        classifier = model.fit(data, labels)
        logger.info("Model successfully fitted.")
        setattr(self, classifier_name, classifier)

        return classifier
    
    def train_car_model(self, X: np.ndarray, Y: np.ndarray, X_val: np.ndarray, Y_val: np.ndarray, loss: Union[nn.Module, torch.nn.modules.loss._Loss] = BCEDiceLoss(), optim: str = "AdamW", batch_size: int = 128, epochs: int = 15, lr: float = 1e-4, output_name: str = "carNet.pth", use_board: bool = False, keep_best_iou: bool = False, lr_adapt: bool = False):
        mean = X.mean(axis=(0, 2, 3), keepdims=True)
        std  = X.std(axis=(0, 2, 3), keepdims=True) + 1e-6

        X = (X - mean) / std
        X_tensor = torch.from_numpy(X).float()
        Y_tensor = torch.from_numpy(Y).float()
        dataset = TensorDataset(X_tensor, Y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

        X_val = (X_val - mean) / std
        X_tensor_val = torch.from_numpy(X_val).float()
        Y_tensor_val = torch.from_numpy(Y_val).float()
        dataset_val = TensorDataset(X_tensor_val, Y_tensor_val)
        loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, pin_memory=True)

        model = CarNet()
        trainer = Trainer(model=model, lr=lr, epochs=epochs, optim=optim, batch_size=batch_size, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), use_board=use_board, loss=loss, lr_adapt=lr_adapt)
        output_path: str = f"./trained_models/car/{output_name}"
        save_callback = None
        if keep_best_iou:
            save_callback = lambda: self._save_best_checkpoint(trainer, mean, std, output_path)
        
        trainer.train_all(loader, loader_val, save_callback=save_callback)
        if keep_best_iou and os.path.exists(output_path):
            checkpoint = torch.load(output_path)
            trainer.model.load_state_dict(checkpoint['model_state_dict'])
        self.car_trainer = trainer
        self.car_mean = mean
        self.car_std = std
    
    def predict_car_model(self, points: np.ndarray, data: np.ndarray, resolution: float = 0.15, patch_size: int = 64, stride: int = 32):
        patches = CarConvolutionalNetworkData.generate_patches(points, data, resolution, patch_size, stride)
        patches = (patches - self.car_mean) / self.car_std
        patches_predicted = self.car_trainer.predict(patches)
        reconstructed_raster: np.ndarray = CarConvolutionalNetworkData.reconstruct_from_patches(patches_predicted, (int(np.ceil(TILE_SIZE/resolution)), int(np.ceil(TILE_SIZE/resolution))), patch_size, stride)
        return reconstructed_raster

    def set_car_model(self, car_model_path: str):
        car_model_dict: dict = torch.load(car_model_path, weights_only=False)
        model: CarNet = CarNet()
        model.load_state_dict(car_model_dict['model_state_dict'])
        self.car_trainer: Trainer = Trainer(model=model, lr=1e-4, epochs=15, batch_size=128, use_board=False)
        self.car_mean: np.ndarray = car_model_dict['mean']
        self.car_std: np.ndarray = car_model_dict['std']

    def test_binary_classifier(self, predicted_data: np.ndarray, test_labels: np.ndarray) -> None:
        preds = predicted_data.astype(bool)
        labels = test_labels.astype(bool)

        tp_total = np.sum(preds & labels)
        fp_total = np.sum(preds & ~labels)
        fn_total = np.sum(~preds & labels)
        tn_total = np.sum(~preds & ~labels)

        total_elements = predicted_data.size 
        global_acc = (tp_total + tn_total) / total_elements if total_elements > 0 else 0
        global_precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
        global_recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
        global_f1 = 2 * (global_precision * global_recall) / (global_precision + global_recall) if (global_precision + global_recall) > 0 else 0
        
        logger.info("=" * 50)
        logger.info("Résultats Globaux :")
        logger.info(f" -> Accuracy  : {global_acc * 100:.2f} %")
        logger.info(f" -> F1-Score  : {global_f1:.4f}")
        logger.info(f" -> Recall    : {global_recall:.4f}")
        logger.info(f" -> Precision : {global_precision:.4f}")
        logger.info("=" * 50)

    def save_classifier(self, classifier: ClassifierMixin, output_path: str = "./trained_models/ground/ground_classifier.pkl"):
            joblib.dump(classifier, output_path)
            logger.info("Model saved to %s", output_path)

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
    
    def __get_raster(self, points: np.ndarray, resolution: float) -> np.ndarray:
        
        grid_width = int(np.ceil(self.patch_size / resolution))
        grid_height = int(np.ceil(self.patch_size / resolution))
        grid_mask = np.zeros((grid_height, grid_width), dtype=bool)
        if len(points) == 0:
            return grid_mask
        
        x_indicies, y_indicies = self.__get_raster_indicies(points, resolution)
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

    def __get_raster_indicies(self, points: np.ndarray, resolution: float) -> tuple[np.ndarray, np.ndarray]:

        if len(points) == 0:
            return np.array([]), np.array([])
        
        x = points[:,0]
        y = points[:, 1]

        x_min = np.floor(round(x.min()) / self.patch_size) * self.patch_size
        y_min = np.floor(round(y.min()) / self.patch_size) * self.patch_size

        grid_width = int(np.ceil(self.patch_size / resolution))
        grid_height = int(np.ceil(self.patch_size / resolution))

        x_indices = np.floor((x - x_min) / resolution).astype(np.int32)
        y_indices = np.floor((y - y_min) / resolution).astype(np.int32)

        x_indices = np.clip(x_indices, 0, grid_width - 1)
        y_indices = np.clip(y_indices, 0, grid_height - 1)
        return x_indices, y_indices
    
    def _save_best_checkpoint(self, trainer: Trainer, mean: float, std: float, output_path: str):
        checkpoint = {'model_state_dict': trainer.model.state_dict(), 'mean': mean, 'std': std}
        torch.save(checkpoint, output_path)

class DataClassifierFormat:
    def __init__(self):
        pass
    
    @staticmethod
    def load_data(point_cloud_paths: str | list[str], classified_true_id: int | list[int] | None = None, features: list[str] = SELECTED_FEATURE_NAMES, return_classification: bool = False, fraction_of_dataset: float = 0.001, data_overview: bool = False, is_random: bool = True):
        np.random.seed(SEED)

        data: np.ndarray = None
        if classified_true_id or return_classification:
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

            if classified_true_id or return_classification:
                if logits is None:
                    if return_classification:
                        logits: np.ndarray = pc.classification
                    else:
                        logits: np.ndarray = np.isin(pc.classification, classified_true_id)
                else:
                    if return_classification:
                        logits = np.concat((logits,pc.classification))
                    else:
                        logits = np.concat((logits, np.isin(pc.classification, classified_true_id)))

            if points is None:
                points = pc.xyz
            else:
                points = np.vstack((points, pc.xyz), dtype=np.float32)

            feature_list = [(getattr(pc, f) - bounds[0]) / (bounds[1] - bounds[0]) if (bounds := get_bounds(f)) is not None else getattr(pc, f) for f in features]
            if data is None:
                data: np.ndarray = np.stack(feature_list, axis=1)
            else:
                data = np.vstack((data, np.stack(feature_list, axis=1)))

        if data_overview and classified_true_id:
            get_data_summary(logits, {i: CLASSIFICATION_MAP[k] for i, k in enumerate(np.unique([0] + classified_true_id)) if k in CLASSIFICATION_MAP})
        
        if classified_true_id or return_classification:
            return points, data, logits
        
        return points, data

    @staticmethod
    def split_train_test(*arrays: np.ndarray, test_size: float = 0.2, random_state: int = SEED):
            return train_test_split(*arrays, test_size=test_size, random_state=random_state)
    

class CarConvolutionalNetworkData:
    def __init__(self):
        pass

    @staticmethod
    def generate_patches(points: np.ndarray, data: np.ndarray, resolution: float, patch_size: int = 64, stride: int = 32):
        density_grid = CarConvolutionalNetworkData._generate_grid_count(points, resolution=resolution)

        height_grid = CarConvolutionalNetworkData._generate_grid_stats(points, data[:, SELECTED_FEATURE_NAMES.index("z_norm")], resolution=resolution, ufunc=np.add.at)

        height_grid[density_grid > 0] /= density_grid[density_grid > 0].astype(np.float64)

        height_max_grid = CarConvolutionalNetworkData._generate_grid_stats(points, data[:, SELECTED_FEATURE_NAMES.index("z_norm")], resolution=resolution, ufunc=np.maximum.at)

        intensity_grid = CarConvolutionalNetworkData._generate_grid_stats(points, data[:, SELECTED_FEATURE_NAMES.index("intensity")], resolution=resolution, ufunc=np.add.at)

        intensity_grid[density_grid > 0] /= density_grid[density_grid > 0].astype(np.float64)

        raster = np.stack((density_grid, height_grid, height_max_grid, intensity_grid), axis=-1)

        patches = CarConvolutionalNetworkData._extract_patches(raster, patch_size, stride)
        patches = np.transpose(patches, (0, 3, 1, 2))
        return patches
    
    @staticmethod
    def load_patches(files: list[str]):
        X_list = []
        Y_list = []
        for file in tqdm(files, desc="Loading patches", unit="patch"):
            if not file.endswith(".npz"):
                continue

            data = np.load(os.path.join(file))
            X_list.append(data['X'])
            Y_list.append(data['Y'])


        X = np.concatenate(X_list, axis=0)
        X = np.transpose(X, (0, 3, 1, 2))
        Y = np.concatenate(Y_list, axis=0)
        
        return X, Y

    
    @staticmethod
    def reconstruct_from_patches(patches, image_shape, patch_size, stride):

        H, W = image_shape

        ys = CarConvolutionalNetworkData._get_patch_positions(H, patch_size, stride)
        xs = CarConvolutionalNetworkData._get_patch_positions(W, patch_size, stride)

        reconstructed = np.zeros((H, W), dtype=np.float32)
        count = np.zeros((H, W), dtype=np.float32)

        idx = 0

        for y in ys:
            for x in xs:

                reconstructed[y:y+patch_size, x:x+patch_size] += patches[idx]
                count[y:y+patch_size, x:x+patch_size] += 1

                idx += 1

        reconstructed /= np.maximum(count, 1)

        return reconstructed

    @staticmethod
    def generate_car_training_dataset(input_pointcloud_directory: str, resolution: float = 0.15, patch_size: int = 64, stride: int = 32, car_class_idx: int = 21, with_classifier: bool = False) -> None:

        classifier = PointCloudClassifier()

        if "vehicle_determination" not in os.listdir('./data'):
            os.mkdir("./data/vehicle_determination")
            os.mkdir("./data/vehicle_determination/training_dataset")
            os.mkdir("./data/vehicle_determination/testing_dataset")
        
        i = 0
        for file in tqdm(os.listdir(input_pointcloud_directory)):
            if not file.endswith('.las') and not file.endswith('.laz'):
                continue

            points, data, ground_truth = DataClassifierFormat.load_data(os.path.join(input_pointcloud_directory, file), [car_class_idx], fraction_of_dataset=1, is_random=False)
            if with_classifier:
                predicted = (classifier.predict(data, points, nsquared_patches=1) == 0)
                points = points[predicted]
                data = data[predicted]
                ground_truth = ground_truth[predicted]

            density_grid = CarConvolutionalNetworkData._generate_grid_count(points, resolution = resolution)

            height_grid = CarConvolutionalNetworkData._generate_grid_stats(points, data[:, SELECTED_FEATURE_NAMES.index("z_norm")], resolution = resolution, ufunc=np.add.at)
            height_grid[density_grid==0] = 0 
            height_grid[density_grid>0] /= density_grid[density_grid>0].astype(np.float64)

            height_max_grid = CarConvolutionalNetworkData._generate_grid_stats(points, data[:, SELECTED_FEATURE_NAMES.index("z_norm")], resolution = resolution, ufunc=np.maximum.at)

            intensity_grid = CarConvolutionalNetworkData._generate_grid_stats(points, data[:, SELECTED_FEATURE_NAMES.index("intensity")], resolution = resolution, ufunc=np.add.at)
            intensity_grid[density_grid==0] = 0 
            intensity_grid[density_grid>0] /= density_grid[density_grid>0].astype(np.float64)

            training_grid = np.stack((density_grid, height_grid, height_max_grid, intensity_grid), axis = -1)

            car_points = points[ground_truth.astype(bool)]
            car_density_grid = CarConvolutionalNetworkData._generate_grid_mask(car_points, resolution)

            X, Y = CarConvolutionalNetworkData._extract_overlapping_patches(training_grid, car_density_grid, patch_size=patch_size, stride=stride)
            if i % 5 == 0:
                np.savez_compressed(f"./data/vehicle_determination/testing_dataset/car_dataset_{file.split('.')[0]}_{resolution}m.npz",X=X,Y=Y)
            else:
                np.savez_compressed(f"./data/vehicle_determination/training_dataset/car_dataset_{file.split('.')[0]}_{resolution}m.npz",X=X,Y=Y)
            
            i +=1

    @staticmethod
    def _extract_patches(image: np.ndarray, patch_size: int, stride: int):
        H, W, C = image.shape

        ys = CarConvolutionalNetworkData._get_patch_positions(H, patch_size, stride)
        xs = CarConvolutionalNetworkData._get_patch_positions(W, patch_size, stride)

        patches = []

        for y in ys:
            for x in xs:
                patches.append(image[y:y+patch_size, x:x+patch_size])

        return np.array(patches)

    @staticmethod
    def _get_raster_indices(points: np.ndarray, resolution: float):
        x = points[:, 0]
        y = points[:, 1]

        if len(points) == 0:
            return np.array([]), np.array([])

        x_min = np.floor(round(x.min()) / TILE_SIZE) * TILE_SIZE
        y_min = np.floor(round(y.min()) / TILE_SIZE) * TILE_SIZE
        
        grid_width = int(TILE_SIZE / resolution)
        grid_height = int(TILE_SIZE / resolution)
        
        x_indices = np.floor((x - x_min) / resolution).astype(np.int32)
        y_indices = np.floor((y - y_min) / resolution).astype(np.int32)
        
        x_indices = np.clip(x_indices, 0, grid_width - 1)
        y_indices = np.clip(y_indices, 0, grid_height - 1)

        return x_indices, y_indices
    
    @staticmethod
    def _get_empty_raster(resolution: float) -> np.ndarray:
        grid_width = int(np.ceil(TILE_SIZE / resolution))
        grid_height = int(np.ceil(TILE_SIZE / resolution))
        grid_mask = np.zeros((grid_height, grid_width), dtype=float)
        return grid_mask

    @staticmethod
    def _generate_grid_count(points: np.ndarray, resolution: float):

        x_indices, y_indices = CarConvolutionalNetworkData._get_raster_indices(points, resolution)
        
        grid_count = CarConvolutionalNetworkData._get_empty_raster(resolution).astype(int)
        np.add.at(grid_count, (y_indices, x_indices), 1)
        return grid_count

    @staticmethod
    def _generate_grid_stats(points: np.ndarray, values:np.ndarray, resolution: float, ufunc=np.maximum.at):

        x_indices, y_indices = CarConvolutionalNetworkData._get_raster_indices(points, resolution)
        
        grid_count = CarConvolutionalNetworkData._get_empty_raster(resolution).astype(np.float64)
        ufunc(grid_count, (y_indices, x_indices), values)
        return grid_count

    @staticmethod
    def _generate_grid_mask(points: np.ndarray, resolution: float):
        grid_mask = CarConvolutionalNetworkData._get_empty_raster(resolution).astype(bool)

        if len(points) == 0:
            return grid_mask

        x_indices, y_indices = CarConvolutionalNetworkData._get_raster_indices(points, resolution)
        grid_mask[y_indices, x_indices] = True
        
        return grid_mask

    @staticmethod
    def _extract_overlapping_patches(image: np.ndarray, mask: np.ndarray, patch_size: int = 64, stride: int = 32):

        H, W, C = image.shape

        ys = CarConvolutionalNetworkData._get_patch_positions(H, patch_size, stride)
        xs = CarConvolutionalNetworkData._get_patch_positions(W, patch_size, stride)

        image_patches = []
        mask_patches = []

        for y in ys:
            for x in xs:

                img_patch = image[y:y+patch_size, x:x+patch_size]
                msk_patch = mask[y:y+patch_size, x:x+patch_size]

                for k in range(4):
                    image_patches.append(np.rot90(img_patch, k=k))
                    mask_patches.append(np.rot90(msk_patch, k=k))

        return np.array(image_patches), np.array(mask_patches)
    
    @staticmethod
    def _get_patch_positions(size: int, patch_size: int, stride: int):
        positions = list(range(0, size - patch_size + 1, stride))

        if positions[-1] != size - patch_size:
            positions.append(size - patch_size)

        return positions