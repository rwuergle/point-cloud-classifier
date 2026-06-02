import laspy

class GeometricFeatureCalculator:
    def __init__(self, point_cloud_path):
        self.pc = laspy.read(point_cloud_path)

    