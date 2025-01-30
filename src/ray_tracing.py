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


@dataclass
class Light:
    position: Point_3D
    intensity: np.float32
    color: Color_RGB


@dataclass
class Scene:
    image_height: np.int32
    image_width: np.int32
    image_plane_height: np.float32
    image_plane_width: np.float32
    aspect_ratio: np.float32
    camera_position: Point_3D
    image_plane_dist: np.float32


def render_scene(
    scene: Scene,
    spheres: list[Sphere],
    lights: list[Light],
) -> Raster_RGB_NxM:
    """TODO:
    1. complete the function render_scene() to output the final image
    2. inside the render_scene() function, you need to implement:
        i)   Compute the ray direction
        ii)  Find the closest intersection
        iii) Shade each pixel using Blinn-Phong Shading
    """
    pixels: Raster_RGB_NxM = np.zeros(
        (scene.image_height, scene.image_width, 3), dtype=np.float32
    )
    return pixels


if __name__ == "__main__":
    spheres: list[Sphere] = [
        # Cyan
        Sphere(
            center=np.array([0.0, 0.0, -5.0], dtype=np.float32),
            radius=np.float32(1.0),
            color=np.array([0.0, 1.0, 1.0], dtype=np.float32),
            ka=np.float32(0.1),
            kd=np.float32(0.7),
            ks=np.float32(0.5),
            shininess=np.float32(32.0),
        ),
        # Magenta
        Sphere(
            center=np.array([2.0, 0.0, -6.0], dtype=np.float32),
            radius=np.float32(1.5),
            color=np.array([1.0, 0.0, 1.0], dtype=np.float32),
            ka=np.float32(0.1),
            kd=np.float32(0.7),
            ks=np.float32(0.5),
            shininess=np.float32(32.0),
        ),
    ]

    lights: list[Light] = [
        Light(
            position=np.array([5, 5, -10], dtype=np.float32),
            intensity=np.float32(1.0),
            color=np.ones(3, dtype=np.float32),
        )
    ]

    scene: Scene = Scene(
        image_height=np.int32(800),
        image_width=np.int32(800),
        image_plane_height=np.float32(2.0),
        image_plane_width=None,
        aspect_ratio=None,
        camera_position=np.zeros(3, dtype=np.float32),
        image_plane_dist=np.float32(1.0),
    )
    scene.aspect_ratio = np.float32(scene.image_width) / np.float32(
        scene.image_height
    )
    scene.image_plane_width = scene.aspect_ratio * scene.image_plane_height

    image: Raster_RGB_NxM = render_scene(scene, spheres, lights)

    # Display the image
    plt.imshow(image)
    # plt.axis('off')
    plt.show()
