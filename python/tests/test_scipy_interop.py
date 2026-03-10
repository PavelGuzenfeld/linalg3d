import pytest
import numpy as np
from linalg3d import Quaternion, Vector3, slerp

scipy = pytest.importorskip("scipy")
from scipy.spatial.transform import Rotation, Slerp as ScipySlerp


class TestScipyQuaternionInterop:
    def test_to_scipy(self):
        q = Quaternion(1.0, 0.0, 0.0, 0.0)  # identity
        arr = np.asarray(q.to_scipy())
        # SciPy convention: [x, y, z, w]
        np.testing.assert_array_equal(arr, [0.0, 0.0, 0.0, 1.0])

    def test_from_scipy(self):
        # SciPy convention: [x, y, z, w]
        arr = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        q = Quaternion.from_scipy(arr)
        assert q.w == pytest.approx(1.0)
        assert q.x == pytest.approx(0.0)

    def test_roundtrip(self):
        q = Quaternion(0.5, 0.5, 0.5, 0.5)
        arr = np.asarray(q.to_scipy())
        q2 = Quaternion.from_scipy(arr)
        assert q2.w == pytest.approx(q.w, abs=1e-10)
        assert q2.x == pytest.approx(q.x, abs=1e-10)
        assert q2.y == pytest.approx(q.y, abs=1e-10)
        assert q2.z == pytest.approx(q.z, abs=1e-10)

    def test_rotation_consistency(self):
        # 90 degrees around Z axis
        import math
        angle = math.pi / 2
        q = Quaternion(math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2))
        scipy_arr = np.asarray(q.to_scipy())
        rot = Rotation.from_quat(scipy_arr)

        v = np.array([1.0, 0.0, 0.0])
        scipy_result = rot.apply(v)
        our_result = q.rotate(Vector3(1.0, 0.0, 0.0))

        assert our_result.x == pytest.approx(scipy_result[0], abs=1e-10)
        assert our_result.y == pytest.approx(scipy_result[1], abs=1e-10)
        assert our_result.z == pytest.approx(scipy_result[2], abs=1e-10)
