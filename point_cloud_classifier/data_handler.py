from collections import defaultdict
import laspy
import geopandas as gpd
from networkx import radius
import numpy as np
import re
import pandas as pd
from point_cloud_classifier.helper import getSingleIDperGroup
from point_cloud_classifier.constants import REQUIRED_FEATURES, SLOPE_PATH
from scipy.stats import binned_statistic_2d

import CSF
from scipy.ndimage import distance_transform_edt

from jakteristics import compute_features
from jakteristics.ckdtree.ckdtree import cKDTree

from tqdm import tqdm

class GeometricFeatureCalculator:
    def __init__(self, point_cloud_path: str, slope: float | None = None):
        self.point_cloud_path: str = point_cloud_path
        with laspy.open(point_cloud_path, laz_backend=laspy.LazBackend.LazrsParallel) as f:
            self.pc: laspy.LasData = getSingleIDperGroup(f.read())
        self.slope = slope

        if self.slope is None:
            self.slope = self.__get_mean_slope()
        
        self._xyz = np.ascontiguousarray(self.pc.xyz, dtype=np.float64)
    
    def __get_mean_slope(self) -> float:
        gdf: gpd.GeoDataFrame = gpd.read_file(SLOPE_PATH)
        return gdf[gdf['tileid'] == re.search(r"\d{7}_\d{7}", self.point_cloud_path).group(0)]['_Slopemean'].item()

    def compute_all_features(self, features: dict[str, list[float]] = REQUIRED_FEATURES) -> None:

        features_z, features_1d, features_2d, features_3d = [], {}, {}, {}
        for feat, radii in features.items():
            suffix = feat[-3:]
            if feat.startswith("z_") or feat.endswith("_z"): features_z.append(feat)
            elif suffix == "_1d": features_1d[feat] = radii
            elif suffix == "_2d": features_2d[feat] = radii
            else: features_3d[feat] = radii

        all_features_list: list[str] = []
        all_features_list.extend(features_z)
        for dict_feat in [features_1d, features_2d, features_3d]:
            for feat_name, radii_list in dict_feat.items():
                if not radii_list:
                    all_features_list.append(feat_name)
                else:
                    for r in radii_list:
                        all_features_list.append(f"{feat_name}_{r}")

        self.add_features_to_point_cloud(all_features_list) 
        self.compute_relative_z(features_z)
        self.compute_3d_features(features_3d)
        self.compute_2d_features(features_2d)
        self.compute_1d_features(features_1d)

    def compute_relative_z(self, feature_names: list[str] = ["z_norm"]) -> None:
        csf: CSF.CSF = CSF.CSF()
        csf.setPointCloud(self._xyz)

        csf.params.bSloopSmooth = True
        csf.params.cloth_resolution = max(np.round(-0.0226 * self.slope + 1.045, 1), 0.3)
        csf.params.rigidness = 3 if self.slope < 5 else (2 if self.slope < 20 else 1)
        csf.params.time_step = 0.65
        csf.params.class_threshold = 0.5

        ground: CSF.VecInt = CSF.VecInt()
        non_ground: CSF.VecInt = CSF.VecInt()

        csf.do_filtering(ground, non_ground)

        ground_mask: np.ndarray = np.zeros(len(self.pc), dtype=bool)
        ground_mask[np.array(ground)] = True

        ground_points = self._xyz[ground_mask]

        resolution: float = csf.params.cloth_resolution

        dem = self._get_dem(ground_points, self._xyz, resolution)

        pix, piy = self._get_raster_indices(self._xyz, resolution)

        ground_height: np.ndarray = dem[piy, pix]
        relative_z: np.ndarray =  self._xyz[:, 2] - ground_height

        self.add_features_to_point_cloud(feature_names, relative_z)

    def compute_3d_features(self, features: dict[str, list[float]]) -> None:
        feat_by_radius_3d: defaultdict[float, list[str]] = defaultdict(list)

        for feature, radii in features.items():
            for radius in radii:
                feat_by_radius_3d[radius].append(feature)
        
        tree_3d: cKDTree = cKDTree(self._xyz)

        for radius, feat_list in tqdm(feat_by_radius_3d.items(), desc="Computing 3D features", unit="radius"):
            compute_list: list[str] = [feat for feat in feat_list if feat if feat != "density_3d"]
            if radius in features.get("density_3d", []):
                compute_list.append("number_of_neighbors")

            features_computed: np.ndarray = np.nan_to_num(compute_features(self._xyz, kdtree=tree_3d, search_radius=radius, feature_names=compute_list, num_threads=-1).astype(np.float32), nan=0.0)
            if "number_of_neighbors" in compute_list:
                features_computed[:,-1] = features_computed[:,-1] / (4/3 * np.pi * radius**3)

            self.add_features_to_point_cloud([f"density_3d_{radius}" if feat == "number_of_neighbors" else f"{feat}_{radius}" for feat in compute_list], features_computed)

    def compute_2d_features(self, features: dict[str, list[float]]) -> None:
        feat_by_radius_2d: defaultdict[float, list[str]] = defaultdict(list)

        for feature, radii in features.items():
            for radius in radii:
                feat_by_radius_2d[radius].append(feature)
        
        xy: np.ndarray = np.empty((len(self.pc.x), 3), dtype=np.float64)
        xy[:, 0] = self.pc.x
        xy[:, 1] = self.pc.y
        xy[:, 2] = 0.0
        xy = np.ascontiguousarray(xy)
        tree_2d: cKDTree = cKDTree(xy)

        for radius, feat_list in tqdm(feat_by_radius_2d.items(), desc="Computing 2D features", unit="radius"):
            compute_list: list[str] = [feat for feat in feat_list if feat if feat not in ("density_2d", "ratio_eigvals_2d", "sum_eigvals_2d")]
            n = len(compute_list)

            compute_density = "density_2d" in feat_list
            compute_ratio = "ratio_eigvals_2d" in feat_list
            compute_sum = "sum_eigvals_2d" in feat_list

            if compute_ratio or compute_sum:
                compute_list.extend(["eigenvalue1", "eigenvalue2"])
            if compute_density:
                compute_list.append("number_of_neighbors")

            features_computed: np.ndarray = np.nan_to_num(compute_features(xy, kdtree=tree_2d, search_radius=radius, feature_names=compute_list, num_threads=-1).astype(np.float32), nan=0.0)

            output_features: np.ndarray = np.empty((len(self.pc), len(feat_list)), dtype=np.float32)

            eig_idx = - int(compute_density)
            col = n
            if compute_ratio:
                output_features[:, col] = np.divide(
                    features_computed[:, eig_idx - 1],
                    features_computed[:, eig_idx - 2],
                    out=np.zeros_like(features_computed[:, eig_idx - 2]),
                    where=features_computed[:, eig_idx - 2] > 0
                )
                col += 1

            if compute_sum:
                output_features[:, col] = features_computed[:, eig_idx - 2] + features_computed[:, eig_idx - 1]
                col += 1

            if compute_density:
                output_features[:, col] = features_computed[:,-1]/ (np.pi * radius**2)

            output_names = [
                f"ratio_eigvals_2d_{radius}" if compute_ratio else None,
                f"sum_eigvals_2d_{radius}" if compute_sum else None,
                f"density_2d_{radius}" if compute_density else None,
            ]
            self.add_features_to_point_cloud([f"{feat}_2d_{radius}" for feat in compute_list[:n]] + [name for name in output_names if name is not None], output_features)

    def compute_1d_features(self, features: dict[str, list[float]]) -> None:
        feat_by_radius_1d: defaultdict[float, list[str]] = defaultdict(list)

        for feature, radii in features.items():
            for radius in radii:
                feat_by_radius_1d[radius].append(feature)

        x = np.asarray(self.pc.x, dtype=np.float32)
        y = np.asarray(self.pc.y, dtype=np.float32)
        z = np.asarray(self.pc.z, dtype=np.float32)

        for radius, feat_list in tqdm(feat_by_radius_1d.items(), desc="Computing 1D features", unit="radius"):
            grid_x = (x / radius).astype(np.int32)
            grid_y = (y / radius).astype(np.int32)
            
            df = pd.DataFrame({'gx': grid_x, 'gy': grid_y, 'z': z})
            grouped = df.groupby(['gx', 'gy'])['z']
            stats = grouped.agg(['min', 'max', 'mean', 'std', 'count'])
            df_merged = df.join(stats, on=['gx', 'gy'])

            output_features: np.ndarray = np.empty((len(self.pc), len(feat_list)), dtype=np.float32)
            
            col = 0
            if "deltaZ_1d" in feat_list:
                output_features[:, col] = df_merged['max'].values - df_merged['min'].values
                col += 1

            if "sigmaZ_1d" in feat_list:
                output_features[:, col] = df_merged['std'].fillna(0).values
                col += 1

            if "posZ_1d" in feat_list:
                output_features[:, col] = (z - df_merged['min'].values).astype(np.float32)

            self.add_features_to_point_cloud([f"{feat}_{radius}" for feat in feat_list], output_features)

    def add_features_to_point_cloud(self, feature_names: list[str],  features: np.ndarray | None = None):
       
        additional_dims: set[str] = set(feature_names) - set(self.pc.point_format.extra_dimension_names)
        add_list = [(laspy.ExtraBytesParams(name=feature_name, type=np.float32, description=feature_name)) for feature_name in additional_dims]
        self.pc.add_extra_dims(add_list)
        if len(add_list) > 0:
            self.pc.update_header()

        if features is not None:
            self.pc[feature_names] = features.astype(np.float32)

    def write_point_cloud(self, output_path: str) -> None:
        new_header = laspy.LasHeader(version=self.pc.header.version, point_format=self.pc.header.point_format)

        new_pc = laspy.LasData(new_header)
        new_pc.points = self.pc.points
        new_pc.write(output_path)

    def _get_dem(self, points: np.ndarray, all_points: np.ndarray, resolution: float, statistic: str = "min") -> np.ndarray:

        if len(points) == 0:
            raise ValueError

        x = points[:, 0]
        y = points[:, 1]
        altitudes = points[:, 2]

        x_min = round(all_points[:,0].min())
        y_min = round(all_points[:,1].min())

        x_max = round(all_points[:,0].max())
        y_max = round(all_points[:,1].max())

        x_edges = np.arange(x_min, x_max + resolution, resolution)
        y_edges = np.arange(y_min, y_max + resolution, resolution)

        dem, _, _, _ = binned_statistic_2d(y, x, altitudes, statistic=f'{statistic}', bins=[y_edges, x_edges])

        nan_mask = np.isnan(dem)
        if np.any(nan_mask):
            nearest_indices = distance_transform_edt(nan_mask, return_distances=False, return_indices=True)
            
            dem[nan_mask] = dem[nearest_indices[0][nan_mask], nearest_indices[1][nan_mask]]

        return dem

    def _get_raster_indices(self, points: np.ndarray, resolution: float):
        x = points[:, 0]
        y = points[:, 1]

        if len(points) == 0:
            return np.array([]), np.array([])

        x_min = round(points[:, 0].min())
        y_min = round(points[:, 1].min())

        x_max = round(points[:, 0].max())
        y_max = round(points[:, 1].max())

        x_edges = np.arange(x_min, x_max + resolution, resolution)
        y_edges = np.arange(y_min, y_max + resolution, resolution)

        grid_width = len(x_edges) - 1
        grid_height = len(y_edges) - 1

        x_indices = np.floor((x - x_min) / resolution).astype(np.int32)
        y_indices = np.floor((y - y_min) / resolution).astype(np.int32)

        x_indices = np.clip(x_indices, 0, grid_width - 1)
        y_indices = np.clip(y_indices, 0, grid_height - 1)

        return x_indices, y_indices