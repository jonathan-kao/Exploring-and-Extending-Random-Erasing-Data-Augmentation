'''
This code refers to the standard implementation of RE from:
https://pytorch.org/vision/main/_modules/torchvision/transforms/transforms.html#RandomErasing

The two key features I added are selecting different erasure shapes and blurring edges.
The default hyperparameters that are already in the standard implementation remain the same.

Jonathan Kao
'''

import torch
import random
import numpy as np

# Define the random erasing function
def random_erasing(
    img_tensor,
    probability=0.5,
    sl=0.02,
    sh=0.4,
    r1=0.3,
    value='random',
    shape='rectangle',
    blend_edges=False,
    blend_type="linear",
    blend_factor=2,
    retain_prob=0.0
):
    # Check if we apply random erasing
    if torch.rand(1) > probability:
        return img_tensor

    # Calculate area of the image
    C, H, W = img_tensor.shape
    img_area = H * W
    
    # Restrict blend_factor between 2 and 6
    if not isinstance(blend_factor, int) or not 2 <= blend_factor <= 6:
        blend_factor = random.randint(2, 6)
    
    # Try up to 100 times to get a valid erasing region
    # While loop is used in the pseudocode in the original paper
    # The adjustment I made here aims to increase computational efficiency.
    # However, it should be rare for the erasing region to remain invalid after 100 iterations.
    for _ in range(100):
        erase_area = torch.empty(1).uniform_(sl, sh) * img_area
        aspect_ratio = torch.empty(1).uniform_(r1, 1/r1)

        # Calculate erasing region dimensions
        erase_height = int(torch.sqrt(erase_area * aspect_ratio))
        erase_width = int(torch.sqrt(erase_area / aspect_ratio))

        # Ensure the region dimensions are within image boundaries
        if erase_height < H and erase_width < W:
            # Randomly select the top-left corner of the erasing region
            x1 = torch.randint(0, W - erase_width + 1, size=(1,)).item()
            y1 = torch.randint(0, H - erase_height + 1, size=(1,)).item()

            # Choose erasure shape
            if shape == 'random':
                shape = random.choice(['rectangle', 'circle'])

            # Compute shape mask
            if shape == 'rectangle':
                shape_mask = torch.ones(C, erase_height, erase_width).to(img_tensor.device)
            elif shape == 'circle':
                shape_mask = generate_circular_mask(erase_width, erase_height).to(img_tensor.device)
                
            # Generate retain mask
            # This is intended to implement pixel-wise random erasing, but it is not functional in the final paper
            retain_mask = (torch.rand(C, erase_height, erase_width) < retain_prob).float().to(img_tensor.device)

            # Blur the edges
            # I'm thinking of designing different implementations for each shape if I have time
            if blend_edges:
                if blend_type == 'random':
                    shape = random.choice(['linear', 'sigmoid', 'quadratic', 'cubic', 'gaussian', 'cosine'])
                
                if blend_type == 'sigmoid':
                    vertical_edge_blur = torch.sigmoid(10 * (torch.linspace(0, 1, erase_height//blend_factor) - 0.5)).to(img_tensor.device)
                    horizontal_edge_blur = torch.sigmoid(10 * (torch.linspace(0, 1, erase_width//blend_factor) - 0.5)).to(img_tensor.device)
                if blend_type == 'quadratic':
                    vertical_edge_blur = torch.pow(torch.linspace(0, 1, erase_height//blend_factor), 2).to(img_tensor.device)
                    horizontal_edge_blur = torch.pow(torch.linspace(0, 1, erase_width//blend_factor), 2).to(img_tensor.device)
                elif blend_type == 'cubic':
                    vertical_edge_blur = torch.pow(torch.linspace(0, 1, erase_height//blend_factor), 3).to(img_tensor.device)
                    horizontal_edge_blur = torch.pow(torch.linspace(0, 1, erase_width//blend_factor), 3).to(img_tensor.device)
                elif blend_type == 'gaussian':
                    vertical_edge_blur = gaussian(torch.linspace(0, 1, erase_height//blend_factor)).to(img_tensor.device)
                    horizontal_edge_blur = gaussian(torch.linspace(0, 1, erase_width//blend_factor)).to(img_tensor.device)
                elif blend_type == 'cosine':
                    vertical_edge_blur = (1 - torch.cos(torch.linspace(0, np.pi, erase_height//blend_factor))) / 2.0
                    horizontal_edge_blur = (1 - torch.cos(torch.linspace(0, np.pi, erase_width//blend_factor))) / 2.0
                else:
                    vertical_edge_blur = torch.linspace(0, 1, erase_height//blend_factor).to(img_tensor.device)
                    horizontal_edge_blur = torch.linspace(0, 1, erase_width//blend_factor).to(img_tensor.device)
                    
                
                vertical_blend = torch.cat([vertical_edge_blur, torch.ones(erase_height - 2*len(vertical_edge_blur)), vertical_edge_blur.flip(0)])
                horizontal_blend = torch.cat([horizontal_edge_blur, torch.ones(erase_width - 2*len(horizontal_edge_blur)), horizontal_edge_blur.flip(0)])
                shape_mask *= vertical_blend[:, None] * horizontal_blend[None, :]

            # Apply the masks
            if value == 'random':
                random_values = torch.rand(C, erase_height, erase_width).to(img_tensor.device)
                img_tensor[:, y1:y1+erase_height, x1:x1+erase_width] = \
                    img_tensor[:, y1:y1+erase_height, x1:x1+erase_width] * (1 - shape_mask) + random_values * shape_mask * (1-retain_mask)
            else:
                if isinstance(value, tuple) or isinstance(value, list):
                    if len(value) != C:
                        raise ValueError(f"The 'value' tuple must have the same number of elements as the number of channels in the image. Expected {C}, got {len(value)}")
                    fill_value = value
                else:
                    fill_value = (value,) * C
                    
                fill_tensor = torch.tensor(fill_value, dtype=img_tensor.dtype)[:, None, None]
                img_tensor[:, y1:y1+erase_height, x1:x1+erase_width] = \
                    img_tensor[:, y1:y1+erase_height, x1:x1+erase_width] * (1 - shape_mask) + fill_tensor * shape_mask * (1-retain_mask)

            return img_tensor
    
    return img_tensor

# This is for the shape masks
# I will design more shape masks if I have time
def generate_circular_mask(erase_width, erase_height):
    # Calculate center and radius
    center_x, center_y = erase_width // 2, erase_height // 2
    radius = min(erase_width, erase_height) // 2

    # Compute a distance map from the center
    yy, xx = torch.meshgrid(torch.arange(erase_height), torch.arange(erase_width))
    distance_map = ((xx - center_x)**2 + (yy - center_y)**2).float().sqrt()

    # Create a mask for the circular region
    mask = (distance_map < radius).float()
    
    return mask

# This is for the edges blurring
def gaussian(x, mu=0.5, sigma=0.15):
    return torch.exp(-0.5 * ((x - mu) / sigma) ** 2)


'''
The hyperparameters probability=0.5, sl=0.02, sh=0.4, r1=0.3, and value=0 are from
the standard implementation of RE transform. The standard implementation: 
https://pytorch.org/vision/main/_modules/torchvision/transforms/transforms.html#RandomErasing
'''
class RandomErasingTransform:
    def __init__(
        self,
        probability=0.5,
        sl=0.02,
        sh=0.4,
        r1=0.3,
        value=0,
        shape='rectangle',
        blend_edges=False,
        blend_type="linear",
        blend_factor=0.0,
        retain_prob=0.0
    ):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.value = value
        self.shape = shape
        self.blend_edges = blend_edges
        self.blend_type = blend_type
        self.blend_factor = blend_factor
        self.retain_prob = retain_prob

    def __call__(self, img_tensor):

        # Apply random erasing
        augmented_tensor = random_erasing(
            img_tensor,
            self.probability,
            self.sl,
            self.sh,
            self.r1,
            self.value,
            self.shape,
            self.blend_edges,
            self.blend_type,
            self.blend_factor,
            self.retain_prob
        )

        return augmented_tensor