import pytest
import numpy as np
from linalg3d import Matrix2, Matrix3, Matrix4, Vector2, Vector3, Vector4, mat_mul


class TestMatrix3:
    def test_identity(self):
        m = Matrix3.identity()
        assert m.determinant() == pytest.approx(1.0)

    def test_determinant(self):
        m = Matrix3(2.0, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 2.0)
        assert m.determinant() == pytest.approx(4.0)

    def test_inverse(self):
        m = Matrix3(2.0, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 2.0)
        inv = m.inverse()
        assert inv is not None
        result = m @ inv
        for i in range(3):
            for j in range(3):
                expected = 1.0 if i == j else 0.0
                assert np.asarray(result.numpy()).flat[i * 3 + j] == pytest.approx(expected, abs=1e-10)

    def test_singular(self):
        m = Matrix3(1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0)
        assert m.inverse() is None

    def test_transpose(self):
        m = Matrix3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
        t = m.transpose()
        arr = np.asarray(t.numpy())
        assert arr[0, 1] == pytest.approx(4.0)
        assert arr[1, 0] == pytest.approx(2.0)

    def test_mat_vec_mul(self):
        m = Matrix3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
        v = Vector3(1.0, 1.0, 1.0)
        r = mat_mul(m, v)
        assert r.x == pytest.approx(6.0)
        assert r.y == pytest.approx(15.0)
        assert r.z == pytest.approx(24.0)


class TestMatrix4:
    def test_identity(self):
        m = Matrix4.identity()
        assert m.determinant() == pytest.approx(1.0)

    def test_inverse(self):
        m = Matrix4(2.0, 0.0, 0.0, 1.0,
                     0.0, 3.0, 0.0, 0.0,
                     0.0, 0.0, 4.0, 0.0,
                     1.0, 0.0, 0.0, 2.0)
        inv = m.inverse()
        assert inv is not None
        result = m @ inv
        arr = np.asarray(result.numpy())
        np.testing.assert_allclose(arr, np.eye(4), atol=1e-10)

    def test_singular(self):
        m = Matrix4(1.0, 0.0, 0.0, 0.0,
                     0.0, 0.0, 0.0, 0.0,
                     0.0, 0.0, 1.0, 0.0,
                     0.0, 0.0, 0.0, 1.0)
        assert m.inverse() is None
