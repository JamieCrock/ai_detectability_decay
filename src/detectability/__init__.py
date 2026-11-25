"""AI detectability: linguistic-feature detectors and decay analysis."""

from .features import FEATURE_NAMES, extract_features, feature_matrix
from .models import DETECTORS, ModelBundle, build_detector, fit_bundle
from .transforms import TRANSFORMS, TransformationSchedule

__all__ = [
    "DETECTORS",
    "FEATURE_NAMES",
    "ModelBundle",
    "TRANSFORMS",
    "TransformationSchedule",
    "build_detector",
    "extract_features",
    "feature_matrix",
    "fit_bundle",
]

__version__ = "1.0.0"
