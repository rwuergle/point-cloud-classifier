import laspy
import copy
import numpy as np
from constants import GROUPS

def getSingleIDperGroup(pc: laspy.LasData):
    for group in GROUPS:
        target_id = group[0]
        for other_id in group[1:]:
            pc.classification[pc.classification == other_id] = target_id
    return pc


def visualize_point_cloud(pc: laspy.LasData, classification: np.ndarray = None, outputName: str = "visualisation/visualization.las", remove_extra_dims: bool = False):
        if classification is None:
            classification = pc.classification

        header_copy = copy.deepcopy(pc.header)
        pcVis = laspy.LasData(header_copy)
        pcVis.points = pc.points.copy()
        pcVis.update_header()

        if remove_extra_dims:
            pcVis.remove_extra_dims(pcVis.point_format.extra_dimension_names)
            
        pcVis.write(outputName)