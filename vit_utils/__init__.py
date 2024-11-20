from enum import Enum

class KittiLabels(Enum):
    """
    Enum class for KITTI dataset labels and their corresponding IDs.
    """
    CAR = 0
    VAN = 1
    TRUCK = 2
    PEDESTRIAN = 3
    PERSON_SITTING = 4
    CYCLIST = 5
    TRAM = 6
    MISC = 7
    DONTCARE = 8

def label_to_id(label: str) -> int:
    """Convert a KITTI label string to its numeric ID"""
    try:
        return KittiLabels[label.upper()].value
    except KeyError:
        raise ValueError(f"Unknown KITTI label: {label}")

def id_to_label(id: int) -> str:
    """Convert a numeric ID to its corresponding KITTI label string"""
    try:
        return KittiLabels(id).name.lower()
    except ValueError:
        raise ValueError(f"Invalid KITTI label ID: {id}")
