from point_cloud_classifier.data_handler import GeometricFeatureCalculator
from helper import visualize_point_cloud

def test_compute_relative_z():
    calculator = GeometricFeatureCalculator("./data/point_clouds/2536500_1195500.copc.laz")
    relative_z = calculator.compute_relative_z()
    assert relative_z is not None
    assert len(relative_z) == len(calculator.pc)

    visualize_point_cloud(calculator.pc, outputName="visualisation/relative_z.las")


if __name__ == "__main__":
    test_compute_relative_z()