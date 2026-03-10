import pytest
import numpy as np
from linalg3d import Vector2, Vector3, Vector4, Matrix2, Matrix3, Matrix4, Quaternion


class TestZeroCopyVectors:
    def test_vector3_numpy(self):
        v = Vector3(1.0, 2.0, 3.0)
        arr = np.asarray(v.numpy())
        assert arr.shape == (3,)
        assert arr.dtype == np.float64
        np.testing.assert_array_equal(arr, [1.0, 2.0, 3.0])

    def test_vector3_zero_copy(self):
        v = Vector3(1.0, 2.0, 3.0)
        arr = np.asarray(v.numpy())
        arr[0] = 99.0
        assert v.x == 99.0  # zero-copy: modification reflects

    def test_vector2_numpy(self):
        v = Vector2(3.0, 4.0)
        arr = np.asarray(v.numpy())
        assert arr.shape == (2,)
        np.testing.assert_array_equal(arr, [3.0, 4.0])

    def test_vector4_numpy(self):
        v = Vector4(1.0, 2.0, 3.0, 4.0)
        arr = np.asarray(v.numpy())
        assert arr.shape == (4,)
        np.testing.assert_array_equal(arr, [1.0, 2.0, 3.0, 4.0])

    def test_from_numpy(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        v = Vector3.from_numpy(arr)
        assert v.x == 1.0 and v.y == 2.0 and v.z == 3.0


class TestZeroCopyMatrices:
    def test_matrix3_numpy(self):
        m = Matrix3.identity()
        arr = np.asarray(m.numpy())
        assert arr.shape == (3, 3)
        assert arr.dtype == np.float64
        np.testing.assert_array_equal(arr, np.eye(3))

    def test_matrix3_zero_copy(self):
        m = Matrix3.identity()
        arr = np.asarray(m.numpy())
        arr[0, 1] = 42.0
        arr2 = np.asarray(m.numpy())
        assert arr2[0, 1] == 42.0  # zero-copy

    def test_matrix4_numpy(self):
        m = Matrix4.identity()
        arr = np.asarray(m.numpy())
        assert arr.shape == (4, 4)
        np.testing.assert_array_equal(arr, np.eye(4))

    def test_numpy_det_consistency(self):
        m = Matrix3(2.0, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 2.0)
        arr = np.asarray(m.numpy())
        np_det = np.linalg.det(arr)
        assert np_det == pytest.approx(m.determinant(), abs=1e-10)


class TestZeroCopyQuaternion:
    def test_quaternion_numpy(self):
        q = Quaternion(1.0, 2.0, 3.0, 4.0)
        arr = np.asarray(q.numpy())
        assert arr.shape == (4,)
        np.testing.assert_array_equal(arr, [1.0, 2.0, 3.0, 4.0])  # [w,x,y,z]

    def test_quaternion_zero_copy(self):
        q = Quaternion(1.0, 0.0, 0.0, 0.0)
        arr = np.asarray(q.numpy())
        arr[1] = 0.5  # modify x
        assert q.x == 0.5
