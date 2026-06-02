import laspy
import geopandas as gpd
import numpy as np
import re
import CSF
from scipy.interpolate import griddata

from point_cloud_classifier.helper import getSingleIDperGroup

SLOPE_PATH = "./data/slopes/tile_decision_slope.gpkg"

class GeometricFeatureCalculator:
    def __init__(self, point_cloud_path: str):
        self.point_cloud_path = point_cloud_path
        self.pc = getSingleIDperGroup(laspy.read(point_cloud_path))
        self.slope = self.__get_mean_slope()
    
    def __get_mean_slope(self):
        gdf = gpd.read_file(SLOPE_PATH)
        return gdf[gdf['tileid'] == re.search(r"\d{7}_\d{7}", self.point_cloud_path).group(0)]['_Slopemean'].item()

    def compute_features(self):

        pass

    def compute_relative_z(self):
        csf = CSF.CSF()
        csf.setPointCloud(self.pc.xyz)

        csf.params.bSloopSmooth = True
        csf.params.cloth_resolution = max(np.round(-0.0226 * self.slope + 1.045), 0.2)
        csf.params.rigidness = 3 if self.slope < 5 else (2 if self.slope < 20 else 1)
        csf.params.time_step = 0.65
        csf.params.class_threshold = 0.
        csf.params.interations = 200 

        ground = CSF.VecInt()
        non_ground = CSF.VecInt()

        csf.do_filtering(ground, non_ground)

        ground_mask = np.zeros(len(self.pc), dtype=bool)
        ground_mask[np.array(ground)] = True

        ground_points = self.pc.xyz[ground_mask]

        resolution = csf.params.cloth_resolution
        xRange = np.arange(self.pc.x.min(), self.pc.x.max() + resolution, resolution)
        yRange = np.arange(self.pc.y.min(), self.pc.y.max() + resolution, resolution)

        gridX, gridY = np.meshgrid(xRange, yRange)
        dem = griddata(ground_points[:, :2], ground_points[:, 2], (gridX, gridY), method='linear')

        px = self.pc.xyz[:, 0]
        py = self.pc.xyz[:, 1]
        pz = self.pc.xyz[:, 2]

        pix = np.clip(((px - xRange[0]) / resolution).astype(int), 0, len(xRange) - 1)
        piy = np.clip(((py - yRange[0]) / resolution).astype(int), 0, len(yRange) - 1)

        ground_height = dem[piy, pix]
        relative_z = pz - ground_height
        return relative_z

