"""Property-based fuzz testing with Hypothesis."""
import pytest
import math
import numpy as np
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from linalg3d import (
    Vector2, Vector3, Vector4,
    Matrix3, Matrix4,
    Quaternion,
    Radians, Degrees,
    slerp, mat_mul,
    angle_between_vectors,
)

# Strategies
reasonable = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
small = st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)

vector3s = st.builds(Vector3, reasonable, reasonable, reasonable)
vector2s = st.builds(Vector2, reasonable, reasonable)
nonzero_vector3s = vector3s.filter(lambda v: v.norm_sq() > 1e-20)
quaternions = st.builds(Quaternion, small, small, small, small).filter(lambda q: q.norm_sq() > 1e-20)
unit_quaternions = quaternions.map(lambda q: q.normalized())
scalars = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)
nonzero_scalars = scalars.filter(lambda s: abs(s) > 1e-10)
t_values = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


# =============================================================================
# Vector properties
# =============================================================================

class TestVectorProperties:
    @given(v=vector3s)
    @settings(max_examples=200)
    def test_norm_non_negative(self, v):
        assert v.norm() >= 0.0

    @given(v=nonzero_vector3s)
    @settings(max_examples=200)
    def test_normalized_unit_length(self, v):
        n = v.normalized()
        assert n.norm() == pytest.approx(1.0, abs=1e-6)

    @given(v=vector3s)
    @settings(max_examples=200)
    def test_dot_self_equals_norm_sq(self, v):
        assert v.dot(v) == pytest.approx(v.norm_sq(), abs=1e-6)

    @given(a=vector3s, b=vector3s)
    @settings(max_examples=200)
    def test_cross_anti_commutative(self, a, b):
        ab = a.cross(b)
        ba = b.cross(a)
        # Tolerance scales with input magnitudes (FP error in cross product)
        tol = max(1e-6, a.norm() * b.norm() * 1e-10)
        assert ab.x == pytest.approx(-ba.x, abs=tol)
        assert ab.y == pytest.approx(-ba.y, abs=tol)
        assert ab.z == pytest.approx(-ba.z, abs=tol)

    @given(a=vector3s, b=vector3s)
    @settings(max_examples=200)
    def test_cross_perpendicular(self, a, b):
        c = a.cross(b)
        # Scale tolerance by magnitudes (floating-point error grows with large values)
        tol = max(1e-5, a.norm() * b.norm() * a.norm() * 1e-15)
        assert a.dot(c) == pytest.approx(0.0, abs=tol)

    @given(v=vector3s, s=nonzero_scalars)
    @settings(max_examples=200)
    def test_scalar_norm(self, v, s):
        scaled = v * s
        assert scaled.norm() == pytest.approx(abs(s) * v.norm(), rel=1e-6, abs=1e-10)


# =============================================================================
# Matrix properties
# =============================================================================

class TestMatrixProperties:
    @given(m00=small, m01=small, m02=small,
           m10=small, m11=small, m12=small,
           m20=small, m21=small, m22=small)
    @settings(max_examples=200)
    def test_transpose_involution(self, m00, m01, m02, m10, m11, m12, m20, m21, m22):
        m = Matrix3(m00, m01, m02, m10, m11, m12, m20, m21, m22)
        tt = m.transpose().transpose()
        arr_m = np.asarray(m.numpy())
        arr_tt = np.asarray(tt.numpy())
        np.testing.assert_allclose(arr_tt, arr_m, atol=1e-10)

    @given(m00=small, m01=small, m02=small,
           m10=small, m11=small, m12=small,
           m20=small, m21=small, m22=small)
    @settings(max_examples=200)
    def test_det_transpose(self, m00, m01, m02, m10, m11, m12, m20, m21, m22):
        m = Matrix3(m00, m01, m02, m10, m11, m12, m20, m21, m22)
        assert m.determinant() == pytest.approx(m.transpose().determinant(), abs=1e-6)

    @given(m00=small, m01=small, m02=small,
           m10=small, m11=small, m12=small,
           m20=small, m21=small, m22=small)
    @settings(max_examples=200)
    def test_inverse_identity(self, m00, m01, m02, m10, m11, m12, m20, m21, m22):
        m = Matrix3(m00, m01, m02, m10, m11, m12, m20, m21, m22)
        # Skip near-singular matrices where numerical inverse is unreliable
        assume(abs(m.determinant()) > 1e-6)
        inv = m.inverse()
        if inv is None:
            return  # singular
        result = m @ inv
        arr = np.asarray(result.numpy())
        np.testing.assert_allclose(arr, np.eye(3), atol=1e-4)


# =============================================================================
# Quaternion properties
# =============================================================================

class TestQuaternionProperties:
    @given(q=quaternions)
    @settings(max_examples=200)
    def test_normalized_unit(self, q):
        n = q.normalized()
        assert n.norm() == pytest.approx(1.0, abs=1e-10)

    @given(q=quaternions)
    @settings(max_examples=200)
    def test_inverse_identity(self, q):
        inv = q.inverse()
        product = q * inv
        assert product.w == pytest.approx(1.0, abs=1e-6)
        assert product.x == pytest.approx(0.0, abs=1e-6)
        assert product.y == pytest.approx(0.0, abs=1e-6)
        assert product.z == pytest.approx(0.0, abs=1e-6)

    @given(q1=quaternions, q2=quaternions, q3=quaternions)
    @settings(max_examples=100)
    def test_associativity(self, q1, q2, q3):
        a = (q1 * q2) * q3
        b = q1 * (q2 * q3)
        assert a.w == pytest.approx(b.w, abs=1e-3)
        assert a.x == pytest.approx(b.x, abs=1e-3)
        assert a.y == pytest.approx(b.y, abs=1e-3)
        assert a.z == pytest.approx(b.z, abs=1e-3)

    @given(q=unit_quaternions, v=vector3s)
    @settings(max_examples=200)
    def test_rotation_preserves_norm(self, q, v):
        r = q.rotate(v)
        assert r.norm() == pytest.approx(v.norm(), rel=1e-6, abs=1e-10)

    @given(a=unit_quaternions, b=unit_quaternions, t=t_values)
    @settings(max_examples=200)
    def test_slerp_unit_norm(self, a, b, t):
        result = slerp(a, b, t)
        assert result.norm() == pytest.approx(1.0, abs=1e-6)


# =============================================================================
# Angle properties
# =============================================================================

class TestAngleProperties:
    @given(x=st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False))
    @settings(max_examples=200)
    def test_radians_roundtrip(self, x):
        r = Radians(x)
        d = r.to_degrees()
        r2 = d.to_radians()
        assert r2.value() == pytest.approx(x, abs=1e-6)

    @given(x=st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False))
    @settings(max_examples=200)
    def test_sin_cos_identity(self, x):
        r = Radians(x)
        assert r.sin() ** 2 + r.cos() ** 2 == pytest.approx(1.0, abs=1e-10)


# =============================================================================
# NumPy consistency
# =============================================================================

class TestNumpyConsistency:
    @given(m00=small, m01=small, m02=small,
           m10=small, m11=small, m12=small,
           m20=small, m21=small, m22=small)
    @settings(max_examples=200)
    def test_det_matches_numpy(self, m00, m01, m02, m10, m11, m12, m20, m21, m22):
        m = Matrix3(m00, m01, m02, m10, m11, m12, m20, m21, m22)
        arr = np.asarray(m.numpy())
        np.testing.assert_allclose(np.linalg.det(arr), m.determinant(), atol=1e-4, rtol=1e-6)

    @given(v=vector3s)
    @settings(max_examples=200)
    def test_identity_matmul(self, v):
        m = Matrix3.identity()
        result = mat_mul(m, v)
        assert result.x == pytest.approx(v.x, abs=1e-10)
        assert result.y == pytest.approx(v.y, abs=1e-10)
        assert result.z == pytest.approx(v.z, abs=1e-10)
