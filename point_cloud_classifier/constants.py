GROUPS = [[1],[2,18,31],[3,4,5],[6],[7],[9, 41],[11],[14],[15],[17],[19],[21],[22],[25],[26],[29]]
CLASSIFICATION_MAP = {
    0: "Not of the class",
    1: "Unclassified",
    2:	"Ground",
    3:	"vegetation",
    6:	"Building roofs",
    7:	"Low Point (Noise)",
    9:	"Water",
    11:	"Piles, heaps (natural materials)",
    14:	"Cables",
    15:	"Masts, antenas",
    17:	"Bridges",
    19:	"Street lights",
    21:	"Cars",
    22:	"Building facades",
    25:	"Cranes, trains, temporary objects",
    26:	"Roof structures",
    29:	"Walls",
}

REQUIRED_FEATURES = {
    "eigenvalue_sum": [1.0, 2.0],
    "omnivariance": [1.0, 2.0],
    "eigenentropy": [0.4, 1.0, 2.0],
    "anisotropy": [0.4, 1.0, 2.0],
    "planarity": [0.4, 1.0, 2.0],
    "linearity": [0.4, 1.0, 2.0],
    "surface_variation": [0.4, 1.0, 2.0],
    "sphericity": [0.4, 1.0, 2.0],
    "verticality": [0.4, 1.0, 2.0],
    "density_3d": [0.4, 1.0, 2.0],
    "density_2d": [0.4, 1.0, 2.0],
    "ratio_eigvals_2d": [1.0, 2.0],
    "sum_eigvals_2d": [0.4, 1.0, 2.0],
    "deltaZ_1d": [0.4, 1.0],
    "sigmaZ_1d": [0.4, 1.0], 
    "posZ_1d": [1.0, 2.0],
    "relative_z": None,
}

FEATURE_RANGES = {
    "intensity": [0,65535],
    "red":[0,255],
    "blue":[0,255],
    "green":[0,255],
    "density":[0, 500],
    "sigmaZ": [0, 500],
    "deltaZ": [0, 60],
    "posZ":[0,60],
    "relative_z":[0,60],
    "return_number": [1,7],
    "number_of_returns":[1,7]
}


EXTRA_ATTRIBUTES = [f"{key}_{val}" for key, values in REQUIRED_FEATURES.items() for val in values]

SELECTED_FEATURE_NAMES = EXTRA_ATTRIBUTES.copy()
SELECTED_FEATURE_NAMES.extend(["return_number", "number_of_returns", "intensity", "z_norm"])

SEED = 42