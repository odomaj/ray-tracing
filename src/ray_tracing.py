from typing import Annotated, Literal
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt


Point_3D = Annotated[np.typing.NDArray[np.float32], Literal[3]]
Color_RGB = Annotated[np.typing.NDArray[np.float32], Literal[3]]
Raster_RGB_NxM = Annotated[
    np.typing.NDArray[np.float32],
    Literal["N", "M", 3],
]


@dataclass
class Sphere:
    center: Point_3D
    radius: np.float32
    color: Color_RGB
    ka: np.float32
    kd: np.float32
    ks: np.float32
    shininess: np.float32


def render_scene(pixels: Raster_RGB_NxM) -> Raster_RGB_NxM:
    """TODO:
    1. complete the function render_scene() to output the final image
    2. inside the render_scene() function, you need to implement:
        i)   Compute the ray direction
        ii)  Find the closest intersection
        iii) Shade each pixel using Blinn-Phong Shading
    """
    return pixels


# Main function
if __name__ == "__main__":
    # Define spheres
    spheres: list[Sphere] = [
        # Cyan
        Sphere(
            center=np.array([0.0, 0.0, -5.0], dtype=np.float32),
            radius=1.0,
            color=np.array([0.0, 1.0, 1.0], dtype=np.float32),
            ka=0.1,
            kd=0.7,
            ks=0.5,
            shininess=32.0,
        ),
        # Magenta
        Sphere(
            center=np.array([2.0, 0.0, -6.0], dtype=np.float32),
            radius=1.5,
            color=np.array([1.0, 0.0, 1.0], dtype=np.float32),
            ka=0.1,
            kd=0.7,
            ks=0.5,
            shininess=32.0,
        ),
    ]

    # Define light source
    light_position: Point_3D = np.array([5, 5, -10], dtype=np.float32)
    light_intensity: np.float32 = np.float32(1.0)
    # White light
    light_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    # Render the scene
    image_width: np.int32 = np.int32(800)
    image_height: np.int32 = np.int32(800)

    # Initial pixel colors of the scene (final output image)
    pixel_colors: Raster_RGB_NxM = np.zeros(
        (image_height, image_width, 3), dtype=np.float32
    )

    # Define the image plane
    image_plane_height: np.float32 = np.float32(2.0)
    aspect_ratio: np.float32 = image_width / image_height
    image_plane_width: np.float32 = aspect_ratio * image_plane_height

    # Define the camera
    camera_position: Point_3D = np.array([0, 0, 0])

    # Distance between the camera and the image plane
    image_plane_dist: np.float32 = np.float32(1.0)

    image: Raster_RGB_NxM = render_scene(pixel_colors)

    # Display the image
    plt.imshow(image)
    # plt.axis('off')
    plt.show()
