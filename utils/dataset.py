import numpy as np
import torch
import torch.nn.functional as F
from enum import Enum

def read_P2_from_calib(calib_file):
    with open(calib_file, 'r') as f:
        for line in f:
            if line.startswith('P2:'):
                # split the line and convert strings to float
                values = [float(x) for x in line.strip().split()[1:]]
                # reshape the list into a 3x4 matrix
                P2 = np.array(values).reshape(3, 4)
                return P2
    raise ValueError(f"P2 matrix not found in {calib_file}")

# class KittiLabels(Enum):
#     """
#     Enum class for KITTI dataset labels and their corresponding IDs.
#     """
#     CAR = 0
#     VAN = 1
#     TRUCK = 2
#     PEDESTRIAN = 3
#     PERSON_SITTING = 4
#     CYCLIST = 5
#     TRAM = 6
#     MISC = 7
#     DONTCARE = 7
class KittiLabels(Enum):
    """
    Enum class for KITTI dataset labels and their corresponding IDs.
    """
    CAR = 0
    VAN = 0
    TRUCK = 0
    PEDESTRIAN = 1
    PERSON_SITTING = 1
    CYCLIST = 2
    TRAM = 2
    MISC = 2
    DONTCARE = 2

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
    
class AugmentImage:
    
    def __init__(self, fog_intensity=0.65,
                 salt_prob=0.1, pepper_prob=0.1, pixel_size:int=8,
                 haze_prob=0.5, haze_intensity=0.5):
        """
        Author: Shiv Vignesh
        Atmospheric fog augmentation class using depth and reflectance maps.
        :param beta_range: Range of fog density (scattering coefficient).
        :param airlight_intensity: Range for airlight brightness (fog background light).
        """

        self.fog_intensity = fog_intensity
        self.alpha = 1 - self.fog_intensity
        self.beta = self.fog_intensity 
        self.gamma = 0.0
        
        self.salt_prob = salt_prob
        self.pepper_prob = pepper_prob
        self.pixel_size = pixel_size
                    
    def apply_salt_pepper(self, images:torch.tensor):

        orig_range = (images.min(), images.max())
        if orig_range[1] > 1:  # Assume range is [0, 255]
            images = images / 255.0                
            
        depth_map = images[:, -2, :, :] # RGB + depth + reflectance
        reflectance = images[:, -1, :, :]
        rgb = images[:, :3, :, :]

        # Create a tensor for salt noise
        salt_mask = (torch.rand_like(rgb) < self.salt_prob).float()

        # Create a tensor for pepper noise
        pepper_mask = (torch.rand_like(rgb) < self.pepper_prob).float()

        # Add salt noise (set pixels to 1)
        images_with_salt = rgb + salt_mask

        # Add pepper noise (set pixels to 0)
        images_with_pepper = images_with_salt - pepper_mask

        # Clamp values to ensure they stay within the [0, 1] range
        noisy_images = torch.clamp(images_with_pepper, 0.0, 1.0)    
        
        if orig_range[1] > 1:
            noisy_images = (noisy_images * 255).to(torch.float)  

            
        noisy_images = torch.concat([noisy_images, depth_map.unsqueeze(1), reflectance.unsqueeze(1)], dim=1)
            
        return noisy_images
    
    def pixelate(self, images:torch.tensor):
        
        orig_range = (images.min(), images.max())
        if orig_range[1] > 1:  # Assume range is [0, 255]
            images = images / 255.0                
            
        depth_map = images[:, -2, :, :] # RGB + depth + reflectance
        reflectance = images[:, -1, :, :]
        rgb = images[:, :3, :, :]        

        batch_size, channels, height, width = rgb.shape

        # Compute the new height and width after downsampling
        new_height = height // self.pixel_size
        new_width = width // self.pixel_size

        # Downsample the image to the new size using average pooling
        rgb_resized = F.avg_pool2d(rgb, kernel_size=self.pixel_size, stride=self.pixel_size)

        # Upscale the image back to the original size using nearest neighbor interpolation
        pixelated_images = F.interpolate(rgb_resized, size=(height, width), mode='nearest')

        if orig_range[1] > 1:
            pixelated_images = (pixelated_images * 255).to(torch.float)  
            
        pixelated_images = torch.concat([pixelated_images, depth_map.unsqueeze(1), reflectance.unsqueeze(1)], dim=1)

        return pixelated_images        
    
    def __call__(self, images:torch.tensor, augmentation:str):
        
        if 'transmittance' == augmentation:
            return self.apply_haze(images)

        elif 'SaltPapperNoise' == augmentation:
            return self.apply_salt_pepper(images)  
        
        elif 'pixelate' == augmentation:
            return self.pixelate(images)

