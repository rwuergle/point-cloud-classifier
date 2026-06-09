from collections import defaultdict
import laspy
import geopandas as gpd
from networkx import radius
import numpy as np
import re
import pandas as pd
from point_cloud_classifier.helper import getSingleIDperGroup
from point_cloud_classifier.constants import REQUIRED_FEATURES

import CSF
from scipy.interpolate import griddata

from jakteristics import compute_features
from jakteristics.ckdtree.ckdtree import cKDTree

from tqdm import tqdm

SLOPE_PATH = "./data/slopes/tile_decision_slope.gpkg"

class GeometricFeatureCalculator:
    def __init__(self, point_cloud_path: str):
        self.point_cloud_path: str = point_cloud_path
        self.pc: laspy.LasData = getSingleIDperGroup(laspy.read(point_cloud_path))
        self.slope: float = self.__get_mean_slope()
    
    def __get_mean_slope(self):
        gdf: gpd.GeoDataFrame = gpd.read_file(SLOPE_PATH)
        return gdf[gdf['tileid'] == re.search(r"\d{7}_\d{7}", self.point_cloud_path).group(0)]['_Slopemean'].item()

    def compute_features(self, features: dict[str, list[float]] = REQUIRED_FEATURES) -> None:

        features_z, features_1d, features_2d, features_3d = [], {}, {}, {}
        for feat, radii in features.items():
            suffix = feat[-3:]
            if suffix == feat.startswith("z_") or feat.endswith("_z"): features_z.append(feat)
            elif suffix == "_1d": features_1d[feat] = radii
            elif suffix == "_2d": features_2d[feat] = radii
            else: features_3d[feat] = radii

        self.compute_relative_z(features_z)
        self.compute_3d_features(features_3d)
        self.compute_2d_features(features_2d)
        self.compute_1d_features(features_1d)

    def compute_relative_z(self, feature_names: list[str] = ["relative_z"]) -> None:
        csf: CSF.CSF = CSF.CSF()
        csf.setPointCloud(self.pc.xyz)

        csf.params.bSloopSmooth = True
        csf.params.cloth_resolution = max(np.round(-0.0226 * self.slope + 1.045), 0.2)
        csf.params.rigidness = 3 if self.slope < 5 else (2 if self.slope < 20 else 1)
        csf.params.time_step = 0.65
        csf.params.class_threshold = 0.5
        csf.params.interations = 200 

        ground: CSF.VecInt = CSF.VecInt()
        non_ground: CSF.VecInt = CSF.VecInt()

        csf.do_filtering(ground, non_ground)

        ground_mask: np.ndarray = np.zeros(len(self.pc), dtype=bool)
        ground_mask[np.array(ground)] = True

        ground_points = self.pc.xyz[ground_mask]

        resolution: float = csf.params.cloth_resolution
        xRange: np.ndarray = np.arange(self.pc.x.min(), self.pc.x.max() + resolution, resolution)
        yRange: np.ndarray = np.arange(self.pc.y.min(), self.pc.y.max() + resolution, resolution)

        gridX, gridY = np.meshgrid(xRange, yRange)
        dem: np.ndarray = griddata(ground_points[:, :2], ground_points[:, 2], (gridX, gridY), method='linear')

        if np.isnan(dem).any():
            dem_nn = griddata(
                ground_points[:, :2],
                ground_points[:, 2],
                (gridX, gridY),
                method="nearest"
            )
            dem = np.where(np.isnan(dem), dem_nn, dem)

        px: np.ndarray = self.pc.xyz[:, 0]
        py: np.ndarray = self.pc.xyz[:, 1]
        pz: np.ndarray = self.pc.xyz[:, 2]

        pix: np.ndarray = np.clip(((px - xRange[0]) / resolution).astype(int), 0, len(xRange) - 1)
        piy: np.ndarray = np.clip(((py - yRange[0]) / resolution).astype(int), 0, len(yRange) - 1)

        ground_height: np.ndarray = dem[piy, pix]
        relative_z: np.ndarray = pz - ground_height

        self.add_features_to_point_cloud(relative_z, feature_names)


    def compute_3d_features(self, features: dict[str, list[float]]) -> None:
        feat_by_radius_3d: defaultdict[float, list[str]] = defaultdict(list)

        for feature, radii in features.items():
            for radius in radii:
                feat_by_radius_3d[radius].append(feature)
        
        xyz: np.ndarray = np.ascontiguousarray(self.pc.xyz, dtype=np.float64)
        tree_3d: cKDTree = cKDTree(xyz)

        for radius, feat_list in tqdm(feat_by_radius_3d.items(), desc="Computing 3D features", unit="radius"):
            compute_list: list[str] = [feat for feat in feat_list if feat if feat != "density_3d"]
            if radius in features.get("density_3d", []):
                compute_list.append("number_of_neighbors")

            features_computed: np.ndarray = np.nan_to_num(compute_features(xyz, kdtree=tree_3d, search_radius=radius, feature_names=compute_list).astype(np.float32), nan=0.0)
            if "number_of_neighbors" in compute_list:
                features_computed[:,-1] = features_computed[:,-1] / (4/3 * np.pi * radius**3)

            self.add_features_to_point_cloud(features_computed,[f"density_3d_{radius}" if feat == "number_of_neighbors" else f"{feat}_{radius}" for feat in compute_list])

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

            features_computed: np.ndarray = np.nan_to_num(compute_features(xy, kdtree=tree_2d, search_radius=radius, feature_names=compute_list).astype(np.float32), nan=0.0)

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
            self.add_features_to_point_cloud(output_features, [f"{feat}_2d_{radius}" for feat in compute_list[:n]] + [name for name in output_names if name is not None])

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

            self.add_features_to_point_cloud(output_features, [f"{feat}_{radius}" for feat in feat_list])

    def add_features_to_point_cloud(self, features: np.ndarray, feature_names: list[str]):
       
        additional_dims: set[str] = set(feature_names) - set(self.pc.point_format.extra_dimension_names)
        self.pc.add_extra_dims([(laspy.ExtraBytesParams(name=feature_name, type=np.float32, description=feature_name)) for feature_name in additional_dims])

        self.pc[feature_names] = features.astype(np.float32)
        self.pc.update_header()

    
    def write_point_cloud(self, output_path: str) -> None:
        self.pc.write(output_path)