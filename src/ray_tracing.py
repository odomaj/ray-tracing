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


def normalize(vector: np.ndarray) -> np.ndarray:
    return vector / np.sqrt(np.dot(vector, vector))


def gen_ray(camera_position: Point_3D, pixel_position: Point_3D) -> Point_3D:
    """generates a point representing a vector from the camera to the pixel"""
    return np.array(
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
    """finds real solutions to ax^2+bx+c=0"""
    im: np.float32 = np.square(b) - (4 * a * c)
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
) -> tuple[int, Point_3D] | None:
    """returns the index of the first sphere to intersect the ray, along with
    the point of intersection"""
    sphere_index: int | None = None
    sphere_intersection: int | None = None

    # solves for the first point in time the ray intersects each sphere
    # and saves the sphere that is intersected first
    a: np.float32 = np.dot(ray, ray)
    for i in range(len(spheres)):
        start_to_center: Point_3D = camera_position - spheres[i].center
        b: np.float32 = np.dot(start_to_center, ray)
        b += b
        c: np.float32 = np.dot(start_to_center, start_to_center) - np.square(
            spheres[i].radius
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

    if sphere_index is not None:
        return (sphere_index, sphere_intersection * ray + camera_position)
    else:
        return None


def shade(
    intersection: Point_3D,
    ray: Point_3D,
    sphere: Sphere,
    lights: list[Light],
    ambiant_light: np.float32,
    phong_coefficient: np.float32,
) -> Color_RGB:
    """Implementation of the Blinn Phong shading alg"""
    normal_vector = normalize(sphere.center - intersection)
    total_light: Color_RGB = np.zeros(3, dtype=np.float32)
    # sum the light reflected off of the sphere in the pixel from each light
    # source, then add ambient light
    for light in lights:
        light_vector = normalize(intersection - light.position)

        color: Color_RGB = np.array(
            [
                sphere.color[0] * light.color[0],
                sphere.color[1] * light.color[1],
                sphere.color[2] * light.color[2],
            ],
            dtype=np.float32,
        )

        total_light += (
            sphere.kd
            * light.intensity
            * max(0, np.dot(normal_vector, light_vector))
            * color
        )

        total_light += (
            sphere.ks
            * light.intensity
            * np.power(
                max(
                    0,
                    np.dot(normal_vector, normalize(light_vector + ray)),
                ),
                phong_coefficient,
            )
            * color
        )

    total_light += sphere.ka * ambiant_light * sphere.color
    return total_light


def render_scene(
    scene: Scene,
    spheres: list[Sphere],
    lights: list[Light],
) -> Raster_RGB_NxM:
    """Output a grid of pixels generated from the spheres and lights from the
    inputs from the scene using ray tracing"""
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
    # for every pixel generate a ray, then find the first sphere that
    # intersects the ray and shade the frame
    for y in range(len(pixels)):
        for x in range(len(pixels)):
            # scale pixels to size of frame, then center the frame
            pixel_position[0] = x / x_factor - 1
            pixel_position[1] = y / y_factor - 1
            ray: Point_3D = gen_ray(scene.camera_position, pixel_position)
            close_sphere: tuple[int, Point_3D] | None = closest_intersection(
                scene.camera_position, ray, spheres
            )
            if close_sphere is not None:
                pixels[y, x] = shade(
                    close_sphere[1],
                    ray,
                    spheres[close_sphere[0]],
                    lights,
                    1,
                    1,
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
