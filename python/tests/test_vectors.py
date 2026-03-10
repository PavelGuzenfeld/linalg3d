import pytest
import numpy as np
from linalg3d import Vector2, Vector3, Vector4


class TestVector2:
    def test_construction(self):
        v = Vector2()
        assert v.x == 0.0 and v.y == 0.0
        v = Vector2(3.0, 4.0)
        assert v.x == 3.0 and v.y == 4.0

    def test_norm(self):
        assert Vector2(3.0, 4.0).norm() == pytest.approx(5.0)
        assert Vector2(3.0, 4.0).norm_sq() == pytest.approx(25.0)

    def test_normalized(self):
        n = Vector2(3.0, 4.0).normalized()
        assert n.x == pytest.approx(0.6)
        assert n.y == pytest.approx(0.8)

    def test_dot_cross(self):
        assert Vector2(1.0, 0.0).dot(Vector2(0.0, 1.0)) == pytest.approx(0.0)
        assert Vector2(1.0, 0.0).cross(Vector2(0.0, 1.0)) == pytest.approx(1.0)

    def test_arithmetic(self):
        a, b = Vector2(1.0, 2.0), Vector2(3.0, 4.0)
        r = a + b
        assert r.x == pytest.approx(4.0) and r.y == pytest.approx(6.0)
        r = a - b
        assert r.x == pytest.approx(-2.0) and r.y == pytest.approx(-2.0)
        r = a * 2.0
        assert r.x == pytest.approx(2.0) and r.y == pytest.approx(4.0)


class TestVector3:
    def test_construction(self):
        v = Vector3()
        assert v.x == 0.0 and v.y == 0.0 and v.z == 0.0

    def test_norm(self):
        assert Vector3(3.0, 4.0, 12.0).norm_sq() == pytest.approx(169.0)
        assert Vector3(3.0, 4.0, 0.0).norm() == pytest.approx(5.0)

    def test_dot(self):
        assert Vector3(1.0, 0.0, 0.0).dot(Vector3(0.0, 1.0, 0.0)) == pytest.approx(0.0)

    def test_cross(self):
        r = Vector3(1.0, 0.0, 0.0).cross(Vector3(0.0, 1.0, 0.0))
        assert r.x == pytest.approx(0.0)
        assert r.y == pytest.approx(0.0)
        assert r.z == pytest.approx(1.0)

    def test_arithmetic(self):
        a = Vector3(1.0, 2.0, 3.0)
        b = Vector3(4.0, 5.0, 6.0)
        r = a + b
        assert r.x == pytest.approx(5.0)
        assert r.y == pytest.approx(7.0)
        assert r.z == pytest.approx(9.0)

    def test_negation(self):
        v = -Vector3(1.0, -2.0, 3.0)
        assert v.x == pytest.approx(-1.0)
        assert v.y == pytest.approx(2.0)
        assert v.z == pytest.approx(-3.0)


class TestVector4:
    def test_construction(self):
        v = Vector4(1.0, 2.0, 3.0, 4.0)
        assert v.x == 1.0 and v.w == 4.0

    def test_dot(self):
        a = Vector4(1.0, 2.0, 3.0, 4.0)
        b = Vector4(5.0, 6.0, 7.0, 8.0)
        assert a.dot(b) == pytest.approx(70.0)

    def test_norm(self):
        assert Vector4(1.0, 2.0, 3.0, 4.0).norm_sq() == pytest.approx(30.0)
