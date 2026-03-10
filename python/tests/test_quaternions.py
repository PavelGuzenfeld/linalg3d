import pytest
import math
from linalg3d import Quaternion, Vector3, slerp


class TestQuaternion:
    def test_identity(self):
        q = Quaternion()
        assert q.w == 1.0 and q.x == 0.0 and q.y == 0.0 and q.z == 0.0

    def test_multiply(self):
        q1 = Quaternion(1.0, 2.0, 3.0, 4.0)
        q2 = Quaternion(5.0, 6.0, 7.0, 8.0)
        r = q1 * q2
        assert r.w == pytest.approx(-60.0)
        assert r.x == pytest.approx(12.0)
        assert r.y == pytest.approx(30.0)
        assert r.z == pytest.approx(24.0)

    def test_rotation(self):
        angle = math.pi / 2.0
        q = Quaternion(math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2))
        v = Vector3(1.0, 0.0, 0.0)
        r = q.rotate(v)
        assert r.x == pytest.approx(0.0, abs=1e-10)
        assert r.y == pytest.approx(1.0, abs=1e-10)
        assert r.z == pytest.approx(0.0, abs=1e-10)

    def test_norm_preserving(self):
        q = Quaternion(0.70710678, 0.70710678, 0.0, 0.0)
        v = Vector3(1.0, 2.0, 3.0)
        r = q.rotate(v)
        assert r.norm() == pytest.approx(v.norm(), abs=1e-6)

    def test_inverse(self):
        q = Quaternion(1.0, 2.0, 3.0, 4.0)
        inv = q.inverse()
        product = q * inv
        assert product.w == pytest.approx(1.0, abs=1e-10)
        assert product.x == pytest.approx(0.0, abs=1e-10)

    def test_normalized(self):
        q = Quaternion(1.0, 2.0, 3.0, 4.0)
        n = q.normalized()
        assert n.norm() == pytest.approx(1.0, abs=1e-10)


class TestSlerp:
    def test_endpoints(self):
        a = Quaternion(1.0, 0.0, 0.0, 0.0)
        b = Quaternion(math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4))
        at0 = slerp(a, b, 0.0)
        assert at0.w == pytest.approx(a.w, abs=1e-10)
        at1 = slerp(a, b, 1.0)
        assert at1.w == pytest.approx(b.w, abs=1e-10)

    def test_midpoint_unit_norm(self):
        a = Quaternion(1.0, 0.0, 0.0, 0.0)
        b = Quaternion(math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4))
        mid = slerp(a, b, 0.5)
        assert mid.norm() == pytest.approx(1.0, abs=1e-10)
