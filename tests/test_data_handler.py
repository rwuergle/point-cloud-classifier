from time import time
import numpy as np

from point_cloud_classifier.data_handler import GeometricFeatureCalculator
from point_cloud_classifier.helper import time_it, visualize_point_cloud

from point_cloud_classifier.constants import REQUIRED_FEATURES

@time_it
def test_compute_relative_z():
    relative_z_name: str = "relative_z"
    calculator: GeometricFeatureCalculator = GeometricFeatureCalculator("./data/point_clouds/2536500_1195500_tester_pointcloud.laz")
    calculator.compute_relative_z([relative_z_name])

    assert relative_z_name in calculator.pc.point_format.extra_dimension_names
    assert len(calculator.pc[relative_z_name]) == len(calculator.pc)
    
    visualize_point_cloud(calculator.pc, outputName=f"./visualization/{relative_z_name}.laz")

@time_it
def test_compute_3d_features():
    calculator: GeometricFeatureCalculator = GeometricFeatureCalculator("./data/point_clouds/2536500_1195500_tester_pointcloud.laz")
    calculator.compute_3d_features({"omnivariance": [0.1, 0.2], "density_3d": [0.2], "eigenentropy": [0.1]})

    assert "omnivariance_0.1" in calculator.pc.point_format.extra_dimension_names
    assert "omnivariance_0.2" in calculator.pc.point_format.extra_dimension_names
    assert "density_3d_0.2" in calculator.pc.point_format.extra_dimension_names
    assert "eigenentropy_0.1" in calculator.pc.point_format.extra_dimension_names

    visualize_point_cloud(calculator.pc, outputName=f"./visualization/3d_features.laz")

@time_it
def test_compute_2d_features():
    calculator: GeometricFeatureCalculator = GeometricFeatureCalculator("./data/point_clouds/2536500_1195500_tester_pointcloud.laz")
    calculator.compute_2d_features({"ratio_eigvals_2d": [0.1, 0.2], "density_2d": [0.2], "sum_eigvals_2d": [0.1], "eigenentropy": [0.1]})

    assert "ratio_eigvals_2d_0.1" in calculator.pc.point_format.extra_dimension_names
    assert "ratio_eigvals_2d_0.2" in calculator.pc.point_format.extra_dimension_names
    assert "density_2d_0.2" in calculator.pc.point_format.extra_dimension_names
    assert "sum_eigvals_2d_0.1" in calculator.pc.point_format.extra_dimension_names
    assert "eigenentropy_2d_0.1" in calculator.pc.point_format.extra_dimension_names
    visualize_point_cloud(calculator.pc, outputName=f"./visualization/2d_features.laz")

def test_compute_1d_features():
    calculator: GeometricFeatureCalculator = GeometricFeatureCalculator("./data/point_clouds/2536500_1195500_tester_pointcloud.laz")
    calculator.compute_1d_features({"deltaZ_1d": [0.1, 0.2], "sigmaZ_1d": [0.2], "posZ_1d": [0.1]})

    assert "deltaZ_1d_0.1" in calculator.pc.point_format.extra_dimension_names
    assert "deltaZ_1d_0.2" in calculator.pc.point_format.extra_dimension_names
    assert "sigmaZ_1d_0.2" in calculator.pc.point_format.extra_dimension_names
    assert "posZ_1d_0.1" in calculator.pc.point_format.extra_dimension_names

    visualize_point_cloud(calculator.pc, outputName=f"./visualization/1d_features.laz")

def test_compute_all_features():
    calculator: GeometricFeatureCalculator = GeometricFeatureCalculator("./data/point_clouds/2536500_1195500_tester_pointcloud.laz")
    calculator.compute_features(REQUIRED_FEATURES)

    assert "relative_z" in calculator.pc.point_format.extra_dimension_names
    for feature, radii in REQUIRED_FEATURES.items():
        for radius in radii:
            if feature.endswith("_1d"):
                assert f"{feature}_{radius}" in calculator.pc.point_format.extra_dimension_names
            elif feature.endswith("_2d"):
                assert f"{feature}_{radius}" in calculator.pc.point_format.extra_dimension_names
            else:
                assert f"{feature}_{radius}" in calculator.pc.point_format.extra_dimension_names


if __name__ == "__main__":
    test_compute_1d_features()