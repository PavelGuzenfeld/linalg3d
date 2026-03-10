"""linalg3d - Fast linear algebra with zero-copy NumPy/SciPy interop."""

from linalg3d._core import (
    Vector2,
    Vector3,
    Vector4,
    Matrix2,
    Matrix3,
    Matrix4,
    Quaternion,
    Radians,
    Degrees,
    EulerAngles,
    quaternion_to_matrix,
    quaternion_to_euler_angles,
    euler_angles_to_quaternion,
    angle_between_vectors,
    angle_between_quaternions,
    slerp,
    mat_mul,
)

__version__ = "0.3.0"

__all__ = [
    "Vector2",
    "Vector3",
    "Vector4",
    "Matrix2",
    "Matrix3",
    "Matrix4",
    "Quaternion",
    "Radians",
    "Degrees",
    "EulerAngles",
    "quaternion_to_matrix",
    "quaternion_to_euler_angles",
    "euler_angles_to_quaternion",
    "angle_between_vectors",
    "angle_between_quaternions",
    "slerp",
    "mat_mul",
]
