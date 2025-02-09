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


def gen_ray(camera_position: Point_3D, pixel_position: Point_3D) -> Point_3D:
    """generates a point representing a vector from the camera to the pixel"""
    return -1 * np.array(
        [
            pixel_position[0] - camera_position[0],
            pixel_position[1] - camera_position[1],
            camera_position[2] - pixel_position[2],
        ],
        dtype=np.float32,
    )


def solve_quadratic(
    a: np.float32, b: np.float32, c: np.float32
) -> np.float32 | None:
    im: np.float32 = np.power(b, 2) - (4 * a * c)
    if im < 0:
        return None
    im = np.sqrt(im)

    sol_1 = ((-1 * b) + im) / (2 * a)
    sol_2 = ((-1 * b) - im) / (2 * a)

    return min(sol_1, sol_2)


def closest_intersection(
    camera_position: Point_3D,
    ray: Point_3D,
    spheres: list[Sphere],
) -> int | None:
    """returns the index of the first sphere to intersect the ray"""
    sphere_index: int | None = None
    sphere_intersection: int | None = None
    a: np.float32 = np.dot(ray, ray)
    for i in range(len(spheres)):
        start_to_center: Point_3D = camera_position - spheres[i].center
        b: np.float32 = np.dot(start_to_center, ray)
        b += b
        c: np.float32 = (
            np.dot(start_to_center, start_to_center) - spheres[i].radius
        )

        intersection: np.float32 = solve_quadratic(a, b, c)
        if intersection is not None:
            if sphere_index is not None:
                if intersection < sphere_intersection:
                    sphere_index = i
                    sphere_intersection = intersection
            else:
                sphere_index = i
                sphere_intersection = intersection

    return sphere_index


def shade(sphere: Sphere, lights: list[Light]) -> Color_RGB:
    return np.ones(3, dtype=np.float32)


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
        (scene.image_height, scene.image_width, 3),
        dtype=np.float32,
    )
    pixel_position: Point_3D = np.array(
        [0, 0, scene.image_plane_dist + scene.camera_position[2]],
        dtype=np.float32,
    )
    y_factor = scene.image_height / scene.image_plane_height
    x_factor = scene.image_width / scene.image_plane_width
    for y in range(len(pixels)):
        for x in range(len(pixels)):
            pixel_position[0] = x / x_factor - 1
            pixel_position[1] = y / y_factor - 1
            ray: Point_3D = gen_ray(scene.camera_position, pixel_position)
            close_sphere: int = closest_intersection(
                scene.camera_position, ray, spheres
            )
            if close_sphere is not None:
                pixels[y, x] = shade(spheres[close_sphere], lights)
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
