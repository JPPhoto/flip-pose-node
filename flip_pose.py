# Copyright (c) 2025 Jonathan S. Pollack (https://github.com/JPPhoto)

import numpy as np
from PIL import Image

from invokeai.invocation_api import (
    BaseInvocation,
    ImageField,
    ImageOutput,
    InputField,
    InvocationContext,
    WithBoard,
    WithMetadata,
    invocation,
)


@invocation(
    "flip_pose_invocation", title="FlipPoseInvocation", tags=["flip_pose", "openpose", "image"], version="1.0.0"
)
class FlipPoseInvocation(BaseInvocation, WithMetadata, WithBoard):
    """Flips an openpose image, preserving left and right sidedness"""

    image: ImageField = InputField(description="The pose image to flip")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.images.get_pil(self.image.image_name).convert("RGB")

        # Unified RGB color swap map (bones + joints)
        color_swap_map = {
            # Bones (lines)
            (153, 0, 0): (153, 51, 0),
            (153, 51, 0): (153, 0, 0),
            (153, 102, 0): (102, 153, 0),
            (102, 153, 0): (153, 102, 0),
            (153, 153, 0): (51, 153, 0),
            (51, 153, 0): (153, 153, 0),
            (0, 153, 0): (0, 153, 153),
            (0, 153, 153): (0, 153, 0),
            (51, 0, 153): (153, 0, 153),
            (153, 0, 153): (51, 0, 153),
            (102, 0, 153): (153, 0, 102),
            (153, 0, 102): (102, 0, 153),
            (0, 102, 153): (0, 153, 51),
            (0, 153, 51): (0, 102, 153),
            (0, 51, 153): (0, 153, 102),
            (0, 153, 102): (0, 51, 153),
            # Points (joints)
            (255, 170, 0): (85, 255, 0),
            (85, 255, 0): (255, 170, 0),
            (255, 255, 0): (0, 255, 0),
            (0, 255, 0): (255, 255, 0),
            (170, 255, 0): (0, 255, 85),
            (0, 255, 85): (170, 255, 0),
            (0, 255, 170): (0, 85, 255),
            (0, 85, 255): (0, 255, 170),
            (0, 255, 255): (0, 0, 255),
            (0, 0, 255): (0, 255, 255),
            (0, 170, 255): (85, 0, 255),
            (85, 0, 255): (0, 170, 255),
            (170, 0, 255): (255, 0, 255),
            (255, 0, 255): (170, 0, 255),
            (255, 0, 170): (255, 0, 85),
            (255, 0, 85): (255, 0, 170),
            # Central parts (nose, neck, etc.)
            (255, 0, 0): (255, 0, 0),
            (255, 85, 0): (255, 85, 0),
            (0, 0, 153): (0, 0, 153),
            # Pure white (faces)
            (255, 255, 255): (255, 255, 255),
        }

        # Normalize to uint8 and precompute allowed colors
        color_swap_map = {tuple(np.uint8(k)): tuple(np.uint8(v)) for k, v in color_swap_map.items()}
        allowed_colors = set(color_swap_map.values()).union({(0, 0, 0), (255, 255, 255)})

        def replace_with_tolerance(source_image, target_image, src_rgb, dst_rgb, tolerance=10):
            diff = np.linalg.norm(source_image.astype(np.int16) - np.array(src_rgb, dtype=np.int16), axis=-1)
            mask = diff <= tolerance
            target_image[mask] = dst_rgb

        # Convert to numpy
        image_np = np.array(image)

        # Flip horizontally
        flipped = np.fliplr(image_np)

        # Initialize remapped image (instead of modifying in-place)
        remapped = np.zeros_like(flipped)

        # Apply recoloring using flipped as read source, remapped as write target
        for src_color, dst_color in color_swap_map.items():
            replace_with_tolerance(flipped, remapped, src_color, dst_color, tolerance=10)

        # Clean any leftover non-matching pixels to black
        flat = remapped.reshape(-1, 3)
        for i, px in enumerate(flat):
            if tuple(px) not in allowed_colors:
                flat[i] = (0, 0, 0)
        remapped = flat.reshape(remapped.shape)

        # Save output
        final = Image.fromarray(remapped, mode="RGB")
        dto = context.images.save(image=final)
        return ImageOutput.build(dto)
