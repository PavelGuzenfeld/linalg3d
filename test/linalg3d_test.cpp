#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "linalg3d/linalg.hpp"
#include "linalg3d/format.hpp"
#include <limits>

using namespace linalg3d;

// Mock strong type to test interop without depending on strong-types library
struct MockRadian
{
    double val;
    constexpr explicit MockRadian(double v) noexcept : val{v} {}
    constexpr MockRadian() noexcept = default;
    [[nodiscard]] constexpr double get() const noexcept { return val; }
    constexpr auto operator<=>(const MockRadian &) const = default;
};

constexpr double EPSILON = 1e-5;

constexpr bool nearly_equal(double a, double b, double tol = EPSILON)
{
    return linalg3d::fabs(a - b) < tol;
}

// =============================================================================
// Angle
// =============================================================================

TEST_CASE("Angle: radians <-> degrees conversion")
{
    constexpr Angle<AngleType::RADIANS> rad_angle(PI);
    constexpr auto deg_angle = rad_angle.to_degrees();
    static_assert(linalg3d::fabs(deg_angle.value() - 180.0) < EPSILON);

    constexpr Angle<AngleType::DEGREES> deg_angle2(180.0);
    constexpr auto rad_angle2 = deg_angle2.to_radians();
    static_assert(linalg3d::fabs(rad_angle2.value() - PI) < EPSILON);
}

TEST_CASE("Angle: identity conversion roundtrip")
{
    constexpr Angle<AngleType::RADIANS> orig(PI / 4.0);
    constexpr auto deg = orig.to_degrees();
    constexpr auto back = deg.to_radians();
    static_assert(linalg3d::fabs(orig.value() - back.value()) < EPSILON);
}

TEST_CASE("Angle: trig functions")
{
    constexpr Angle<AngleType::RADIANS> zero(0.0);
    static_assert(zero.sin() == 0.0);
    static_assert(zero.cos() == 1.0);
    static_assert(zero.tan() == 0.0);

    constexpr Angle<AngleType::RADIANS> half_pi(PI / 2.0);
    static_assert(half_pi.sin() == 1.0);
    static_assert(half_pi.cos() == 0.0);
}

TEST_CASE("Angle: operators")
{
    constexpr Angle<AngleType::RADIANS> a(PI / 3.0);
    constexpr Angle<AngleType::RADIANS> b(PI / 6.0);
    static_assert(linalg3d::fabs((a + b).value() - PI / 2.0) < EPSILON);
    static_assert(linalg3d::fabs((a - b).value() - PI / 6.0) < EPSILON);
    static_assert(linalg3d::fabs(a * 2.0 - (PI / 3.0) * 2.0) < EPSILON);
    static_assert(linalg3d::fabs(a / 2.0 - (PI / 3.0) / 2.0) < EPSILON);
}

TEST_CASE("Angle: comparisons")
{
    static_assert(Angle<AngleType::RADIANS>(PI / 2.0) == PI / 2.0);
    static_assert(Angle<AngleType::RADIANS>(PI / 2.0) != PI / 3.0);
    static_assert(Angle<AngleType::RADIANS>(PI / 3.0) < PI / 2.0);
}

TEST_CASE("Angle: static helpers")
{
    static_assert(Angle<AngleType::RADIANS>::from_radians(PI / 2.0) == PI / 2.0);
    static_assert(Angle<AngleType::DEGREES>::from_degrees(90.0) == 90.0);
}

TEST_CASE("Angle: negative conversion")
{
    constexpr Angle<AngleType::DEGREES> neg_deg(-90.0);
    constexpr auto neg_rad = neg_deg.to_radians();
    static_assert(linalg3d::fabs(neg_rad.value() + PI / 2.0) < EPSILON);
}

TEST_CASE("Angle: large value sanitization")
{
    Angle<AngleType::DEGREES> absurd(1e308);
    CHECK_FALSE(is_nan(absurd.to_radians().value()));
    CHECK_FALSE(is_inf(absurd.to_radians().value()));

    Angle<AngleType::RADIANS> absurd_rad(1e308);
    CHECK_FALSE(is_nan(absurd_rad.to_degrees().value()));
    CHECK_FALSE(is_inf(absurd_rad.to_degrees().value()));
}

TEST_CASE("Angle: NaN/Inf sanitization")
{
    constexpr Angle<AngleType::RADIANS> nan_angle(std::numeric_limits<double>::quiet_NaN());
    static_assert(nan_angle.value() == 0.0);

    constexpr Angle<AngleType::RADIANS> inf_angle(std::numeric_limits<double>::infinity());
    static_assert(inf_angle.value() == 0.0);
}

// =============================================================================
// Vector2
// =============================================================================

TEST_CASE("Vector2: construction and basic ops")
{
    static_assert(Vector2().x == 0.0 && Vector2().y == 0.0);
    static_assert(Vector2(3.0, 4.0).norm_sq() == 25.0);
    static_assert(Vector2(0.0, 0.0).norm() == 0.0);
}

TEST_CASE("Vector2: dot and cross")
{
    static_assert(Vector2(1.0, 0.0).dot(Vector2(0.0, 1.0)) == 0.0);
    static_assert(Vector2(1.0, 0.0).cross(Vector2(0.0, 1.0)) == 1.0);
    static_assert(Vector2(0.0, 1.0).cross(Vector2(1.0, 0.0)) == -1.0);
}

TEST_CASE("Vector2: arithmetic")
{
    static_assert((Vector2(1.0, 2.0) + Vector2(3.0, 4.0)) == Vector2(4.0, 6.0));
    static_assert((Vector2(5.0, 7.0) - Vector2(1.0, 2.0)) == Vector2(4.0, 5.0));
    static_assert((-Vector2(1.0, -2.0)) == Vector2(-1.0, 2.0));
    static_assert((Vector2(2.0, 3.0) * 2.0) == Vector2(4.0, 6.0));
    static_assert((Vector2(4.0, 6.0) / 2.0) == Vector2(2.0, 3.0));
}

TEST_CASE("Vector2: compound assignment")
{
    Vector2 v(1.0, 2.0);
    v += Vector2(3.0, 4.0);
    CHECK(v == Vector2(4.0, 6.0));
    v -= Vector2(1.0, 1.0);
    CHECK(v == Vector2(3.0, 5.0));
    v *= 2.0;
    CHECK(v == Vector2(6.0, 10.0));
    v /= 2.0;
    CHECK(v == Vector2(3.0, 5.0));
}

TEST_CASE("Vector2: normalized")
{
    constexpr Vector2 v(3.0, 4.0);
    constexpr auto n = v.normalized();
    static_assert(n.x == 3.0 / 5.0);
    static_assert(n.y == 4.0 / 5.0);
    static_assert(Vector2(0.0, 0.0).normalized() == Vector2(0.0, 0.0));
}

TEST_CASE("Vector2: comparison")
{
    static_assert(Vector2(1.0, 2.0) == Vector2(1.0, 2.0));
    static_assert(Vector2(1.0, 2.0) != Vector2(1.0, 3.0));
    static_assert(Vector2(1.0, 2.0) < Vector2(1.0, 3.0));
    static_assert(Vector2(1.0, 2.0) <= Vector2(1.0, 2.0));
}

// =============================================================================
// Vector3
// =============================================================================

TEST_CASE("Vector3: construction")
{
    static_assert(Vector3().x == 0.0 && Vector3().y == 0.0 && Vector3().z == 0.0);
    static_assert(Vector3(1.0, 2.0, 3.0).x == 1.0);
}

TEST_CASE("Vector3: norm")
{
    constexpr Vector3 v(3.0, 4.0, 12.0);
    static_assert(v.norm_sq() == 169.0);
    static_assert(Vector3(0.0, 0.0, 0.0).norm() == 0.0);
}

TEST_CASE("Vector3: normalized")
{
    constexpr Vector3 v(3.0, 4.0, 0.0);
    constexpr auto n = v.normalized();
    static_assert(n.x == 3.0 / 5.0 && n.y == 4.0 / 5.0 && n.z == 0.0);
    static_assert(Vector3(0.0, 0.0, 0.0).normalized() == Vector3(0.0, 0.0, 0.0));
}

TEST_CASE("Vector3: dot product")
{
    static_assert(Vector3(1.0, 0.0, 0.0).dot(Vector3(0.0, 1.0, 0.0)) == 0.0);
    constexpr Vector3 v1(2.0, 2.0, 1.0);
    static_assert(v1.dot(v1 * 3.0) == 27.0);
}

TEST_CASE("Vector3: cross product")
{
    constexpr Vector3 v1(1.0, 2.0, 3.0);
    static_assert(v1.cross(v1 * 2.0) == Vector3(0.0, 0.0, 0.0));
    static_assert(Vector3(1.0, 0.0, 0.0).cross(Vector3{0.0, 1.0, 0.0}) == Vector3(0.0, 0.0, 1.0));
}

TEST_CASE("Vector3: arithmetic")
{
    static_assert((Vector3(1.0, 2.0, 3.0) + Vector3(4.0, 5.0, 6.0)) == Vector3(5.0, 7.0, 9.0));
    static_assert((Vector3(5.0, 7.0, 9.0) - Vector3(1.0, 2.0, 3.0)) == Vector3(4.0, 5.0, 6.0));
    static_assert((-Vector3(1.0, -2.0, 3.0)) == Vector3(-1.0, 2.0, -3.0));
    static_assert((Vector3(1.0, -2.0, 3.0) * -1.5) == Vector3(-1.5, 3.0, -4.5));
    static_assert((Vector3(2.0, -4.0, 6.0) / 2.0) == Vector3(1.0, -2.0, 3.0));
}

TEST_CASE("Vector3: compound assignment")
{
    Vector3 v(1.0, 2.0, 3.0);
    v += Vector3(4.0, 5.0, 6.0);
    CHECK(v == Vector3(5.0, 7.0, 9.0));
    v -= Vector3(1.0, 1.0, 1.0);
    CHECK(v == Vector3(4.0, 6.0, 8.0));
    v *= 0.5;
    CHECK(v == Vector3(2.0, 3.0, 4.0));
    v /= 2.0;
    CHECK(v == Vector3(1.0, 1.5, 2.0));
}

TEST_CASE("Vector3: comparison operators")
{
    constexpr Vector3 a(1.0, 2.0, 3.0);
    constexpr Vector3 b(1.0, 2.0, 3.0);
    constexpr Vector3 c(1.0, 2.0, 4.0);

    static_assert(a == b);
    static_assert(a != c);
    static_assert(a < c);
    static_assert(a <= b);
    static_assert(a <= c);
    static_assert(!(a > b));
    static_assert(a >= b);
    static_assert(!(a >= c));
    static_assert((a <=> b) == 0);
    static_assert((a <=> c) < 0);
}

TEST_CASE("Vector3: normalization safety")
{
    constexpr Vector3 small_vec(1e-200, 1e-200, 1e-200);
    constexpr auto normed = small_vec.normalized();
    static_assert(!is_nan(normed.x) && !is_nan(normed.y) && !is_nan(normed.z));
}

// =============================================================================
// Vector4
// =============================================================================

TEST_CASE("Vector4: construction and basic ops")
{
    static_assert(Vector4().x == 0.0 && Vector4().y == 0.0 && Vector4().z == 0.0 && Vector4().w == 0.0);
    static_assert(Vector4(1.0, 2.0, 3.0, 4.0).norm_sq() == 30.0);
}

TEST_CASE("Vector4: dot product")
{
    constexpr Vector4 a(1.0, 2.0, 3.0, 4.0);
    constexpr Vector4 b(5.0, 6.0, 7.0, 8.0);
    static_assert(a.dot(b) == 70.0);
}

TEST_CASE("Vector4: arithmetic")
{
    static_assert((Vector4(1.0, 2.0, 3.0, 4.0) + Vector4(5.0, 6.0, 7.0, 8.0)) == Vector4(6.0, 8.0, 10.0, 12.0));
    static_assert((Vector4(5.0, 6.0, 7.0, 8.0) - Vector4(1.0, 2.0, 3.0, 4.0)) == Vector4(4.0, 4.0, 4.0, 4.0));
    static_assert((-Vector4(1.0, -2.0, 3.0, -4.0)) == Vector4(-1.0, 2.0, -3.0, 4.0));
    static_assert((Vector4(1.0, 2.0, 3.0, 4.0) * 2.0) == Vector4(2.0, 4.0, 6.0, 8.0));
    static_assert((Vector4(2.0, 4.0, 6.0, 8.0) / 2.0) == Vector4(1.0, 2.0, 3.0, 4.0));
}

TEST_CASE("Vector4: compound assignment")
{
    Vector4 v(1.0, 2.0, 3.0, 4.0);
    v += Vector4(1.0, 1.0, 1.0, 1.0);
    CHECK(v == Vector4(2.0, 3.0, 4.0, 5.0));
    v *= 2.0;
    CHECK(v == Vector4(4.0, 6.0, 8.0, 10.0));
}

TEST_CASE("Vector4: normalized")
{
    constexpr auto n = Vector4(0.0, 0.0, 0.0, 0.0).normalized();
    static_assert(n == Vector4(0.0, 0.0, 0.0, 0.0));
}

TEST_CASE("Vector4: comparison")
{
    static_assert(Vector4(1.0, 2.0, 3.0, 4.0) == Vector4(1.0, 2.0, 3.0, 4.0));
    static_assert(Vector4(1.0, 2.0, 3.0, 4.0) != Vector4(1.0, 2.0, 3.0, 5.0));
    static_assert(Vector4(1.0, 2.0, 3.0, 4.0) < Vector4(1.0, 2.0, 3.0, 5.0));
}

// =============================================================================
// Matrix2
// =============================================================================

TEST_CASE("Matrix2: construction")
{
    constexpr Matrix2 m;
    static_assert(m.m[0][0] == 0.0 && m.m[0][1] == 0.0 && m.m[1][0] == 0.0 && m.m[1][1] == 0.0);

    constexpr Matrix2 m2(1.0, 2.0, 3.0, 4.0);
    static_assert(m2.m[0][0] == 1.0 && m2.m[0][1] == 2.0 && m2.m[1][0] == 3.0 && m2.m[1][1] == 4.0);
}

TEST_CASE("Matrix2: identity")
{
    constexpr auto id = Matrix2::identity();
    static_assert(id.m[0][0] == 1.0 && id.m[0][1] == 0.0 && id.m[1][0] == 0.0 && id.m[1][1] == 1.0);
}

TEST_CASE("Matrix2: determinant")
{
    static_assert(Matrix2(1.0, 2.0, 3.0, 4.0).determinant() == -2.0);
    static_assert(Matrix2::identity().determinant() == 1.0);
    static_assert(Matrix2(1.0, 2.0, 2.0, 4.0).determinant() == 0.0);
}

TEST_CASE("Matrix2: transpose")
{
    static_assert(Matrix2(1.0, 2.0, 3.0, 4.0).transpose() == Matrix2(1.0, 3.0, 2.0, 4.0));
}

TEST_CASE("Matrix2: inverse")
{
    constexpr auto inv = Matrix2::identity().inverse();
    static_assert(inv.has_value());
    static_assert(inv.value() == Matrix2::identity());

    constexpr auto singular = Matrix2(1.0, 2.0, 2.0, 4.0).inverse();
    static_assert(!singular.has_value());
}

TEST_CASE("Matrix2: arithmetic")
{
    constexpr Matrix2 a(1.0, 2.0, 3.0, 4.0);
    constexpr Matrix2 b(5.0, 6.0, 7.0, 8.0);
    static_assert((a + b) == Matrix2(6.0, 8.0, 10.0, 12.0));
    static_assert((a * 2.0) == Matrix2(2.0, 4.0, 6.0, 8.0));
}

TEST_CASE("Matrix2: multiplication")
{
    constexpr Matrix2 a(1.0, 2.0, 3.0, 4.0);
    constexpr Matrix2 b(5.0, 6.0, 7.0, 8.0);
    constexpr auto c = a * b;
    static_assert(c == Matrix2(19.0, 22.0, 43.0, 50.0));
}

TEST_CASE("Matrix2: compound assignment")
{
    Matrix2 m(1.0, 2.0, 3.0, 4.0);
    m += Matrix2(1.0, 1.0, 1.0, 1.0);
    CHECK(m == Matrix2(2.0, 3.0, 4.0, 5.0));
    m *= 2.0;
    CHECK(m == Matrix2(4.0, 6.0, 8.0, 10.0));
}

TEST_CASE("Matrix2 * Vector2")
{
    constexpr auto id = Matrix2::identity();
    constexpr Vector2 v(3.0, 4.0);
    static_assert((id * v) == v);

    constexpr Matrix2 m(2.0, 0.0, 0.0, 3.0);
    static_assert((m * v) == Vector2(6.0, 12.0));
}

// =============================================================================
// Matrix3
// =============================================================================

TEST_CASE("Matrix3: construction")
{
    constexpr Matrix3 m;
    constexpr bool all_zero = [&m]()
    {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                if (m.m[i][j] != 0.0)
                    return false;
        return true;
    }();
    static_assert(all_zero);
}

TEST_CASE("Matrix3: identity")
{
    constexpr auto id = Matrix3::identity();
    static_assert(id.m[0][0] == 1.0 && id.m[1][1] == 1.0 && id.m[2][2] == 1.0);
    static_assert(id.m[0][1] == 0.0 && id.m[0][2] == 0.0);
}

TEST_CASE("Matrix3: transpose")
{
    constexpr Matrix3 m(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    static_assert(m.transpose() == Matrix3(1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0));
}

TEST_CASE("Matrix3: determinant")
{
    constexpr Matrix3 singular(1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0);
    static_assert(singular.determinant() == 0.0);
    static_assert(Matrix3::identity().determinant() == 1.0);
}

TEST_CASE("Matrix3: inverse with std::expected")
{
    constexpr auto id_inv = Matrix3::identity().inverse();
    static_assert(id_inv.has_value());
    static_assert(id_inv.value() == Matrix3::identity());

    constexpr Matrix3 singular(1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0);
    constexpr auto singular_inv = singular.inverse();
    static_assert(!singular_inv.has_value());
}

TEST_CASE("Matrix3: inverse correctness")
{
    constexpr Matrix3 m(2.0, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 2.0);
    constexpr auto inv = m.inverse();
    static_assert(inv.has_value());

    constexpr Matrix3 result = m * inv.value();
    static_assert(nearly_equal(result.m[0][0], 1.0));
    static_assert(nearly_equal(result.m[1][1], 1.0));
    static_assert(nearly_equal(result.m[2][2], 1.0));
    static_assert(nearly_equal(result.m[0][1], 0.0));
}

TEST_CASE("Matrix3: compound assignment")
{
    Matrix3 m(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    m *= 3.0;
    CHECK(m == Matrix3(3.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 3.0));
}

TEST_CASE("Matrix3 * Vector3")
{
    constexpr Matrix3 m(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    constexpr Vector3 v(1.0, 1.0, 1.0);
    constexpr auto result = m * v;
    static_assert(nearly_equal(result.x, 6.0));
    static_assert(nearly_equal(result.y, 15.0));
    static_assert(nearly_equal(result.z, 24.0));

    constexpr auto id = Matrix3::identity();
    constexpr Vector3 v2(3.14, -2.71, 0.0);
    static_assert((id * v2) == v2);
}

// =============================================================================
// Matrix4
// =============================================================================

TEST_CASE("Matrix4: construction and identity")
{
    constexpr Matrix4 m;
    static_assert(m.m[0][0] == 0.0 && m.m[3][3] == 0.0);

    constexpr auto id = Matrix4::identity();
    static_assert(id.m[0][0] == 1.0 && id.m[1][1] == 1.0 && id.m[2][2] == 1.0 && id.m[3][3] == 1.0);
    static_assert(id.m[0][1] == 0.0 && id.m[3][0] == 0.0);
}

TEST_CASE("Matrix4: determinant")
{
    static_assert(Matrix4::identity().determinant() == 1.0);

    constexpr Matrix4 m(
        2.0, 0.0, 0.0, 0.0,
        0.0, 3.0, 0.0, 0.0,
        0.0, 0.0, 4.0, 0.0,
        0.0, 0.0, 0.0, 5.0);
    static_assert(m.determinant() == 120.0);
}

TEST_CASE("Matrix4: transpose")
{
    constexpr Matrix4 m(
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0);
    constexpr auto t = m.transpose();
    static_assert(t.m[0][1] == 5.0 && t.m[1][0] == 2.0);
    static_assert(t.m[3][0] == 4.0 && t.m[0][3] == 13.0);
}

TEST_CASE("Matrix4: inverse")
{
    constexpr auto id_inv = Matrix4::identity().inverse();
    static_assert(id_inv.has_value());
    static_assert(id_inv.value() == Matrix4::identity());

    // Singular matrix
    constexpr Matrix4 singular(
        1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0);
    static_assert(!singular.inverse().has_value());
}

TEST_CASE("Matrix4: inverse correctness")
{
    constexpr Matrix4 m(
        2.0, 0.0, 0.0, 1.0,
        0.0, 3.0, 0.0, 0.0,
        0.0, 0.0, 4.0, 0.0,
        1.0, 0.0, 0.0, 2.0);
    constexpr auto inv = m.inverse();
    static_assert(inv.has_value());

    constexpr auto result = m * inv.value();
    CHECK(nearly_equal(result.m[0][0], 1.0));
    CHECK(nearly_equal(result.m[1][1], 1.0));
    CHECK(nearly_equal(result.m[2][2], 1.0));
    CHECK(nearly_equal(result.m[3][3], 1.0));
    CHECK(nearly_equal(result.m[0][1], 0.0));
    CHECK(nearly_equal(result.m[0][3], 0.0));
}

TEST_CASE("Matrix4: arithmetic")
{
    constexpr auto id = Matrix4::identity();
    static_assert((id + id) == (id * 2.0));
    static_assert((id * id) == id);
}

TEST_CASE("Matrix4 * Vector4")
{
    constexpr auto id = Matrix4::identity();
    constexpr Vector4 v(1.0, 2.0, 3.0, 4.0);
    static_assert((id * v) == v);

    constexpr Matrix4 scale(
        2.0, 0.0, 0.0, 0.0,
        0.0, 3.0, 0.0, 0.0,
        0.0, 0.0, 4.0, 0.0,
        0.0, 0.0, 0.0, 1.0);
    static_assert((scale * v) == Vector4(2.0, 6.0, 12.0, 4.0));
}

TEST_CASE("Matrix4: compound assignment")
{
    Matrix4 m = Matrix4::identity();
    m *= 5.0;
    CHECK(m.m[0][0] == 5.0);
    CHECK(m.m[1][1] == 5.0);
    CHECK(m.m[0][1] == 0.0);
}

// =============================================================================
// EulerAngles
// =============================================================================

TEST_CASE("EulerAngles: construction")
{
    constexpr EulerAngles<AngleType::RADIANS> e;
    static_assert(e.pitch == 0.0 && e.yaw == 0.0 && e.roll == 0.0);

    constexpr EulerAngles e2(1.0, 2.0, 3.0);
    static_assert(e2.pitch == 1.0 && e2.yaw == 2.0 && e2.roll == 3.0);
}

// =============================================================================
// Quaternion
// =============================================================================

TEST_CASE("Quaternion: default is identity")
{
    constexpr Quaternion q;
    static_assert(q.w == 1.0 && q.x == 0.0 && q.y == 0.0 && q.z == 0.0);
    static_assert(q == Quaternion::identity());
}

TEST_CASE("Quaternion: parameterized construction")
{
    constexpr Quaternion q(1.0, 2.0, 3.0, 4.0);
    static_assert(q.w == 1.0 && q.x == 2.0 && q.y == 3.0 && q.z == 4.0);
}

TEST_CASE("Quaternion: normalized")
{
    constexpr Quaternion q(1.0, 2.0, 3.0, 4.0);
    constexpr auto n = q.normalized();
    constexpr double expected_norm = gcem::sqrt(1.0 + 4.0 + 9.0 + 16.0);
    static_assert(n.w == 1.0 / expected_norm);
    static_assert(n.x == 2.0 / expected_norm);
}

TEST_CASE("Quaternion: inverse")
{
    constexpr Quaternion q(1.0, 2.0, 3.0, 4.0);
    constexpr auto inv = q.inverse();
    constexpr double n_sq = 30.0;
    static_assert(inv.w == 1.0 / n_sq);
    static_assert(inv.x == -2.0 / n_sq);

    constexpr auto product = q * inv;
    static_assert(nearly_equal(product.w, 1.0));
    static_assert(nearly_equal(product.x, 0.0));
    static_assert(nearly_equal(product.y, 0.0));
    static_assert(nearly_equal(product.z, 0.0));
}

TEST_CASE("Quaternion: dot product")
{
    constexpr Quaternion q1(1.0, 2.0, 3.0, 4.0);
    constexpr Quaternion q2(5.0, 6.0, 7.0, 8.0);
    static_assert(q1.dot(q2) == 70.0);
}

TEST_CASE("Quaternion: multiplication")
{
    constexpr Quaternion q1(1.0, 2.0, 3.0, 4.0);
    constexpr Quaternion q2(5.0, 6.0, 7.0, 8.0);
    constexpr auto result = q1 * q2;
    static_assert(result.w == -60.0 && result.x == 12.0 && result.y == 30.0 && result.z == 24.0);
}

TEST_CASE("Quaternion: identity multiplication")
{
    constexpr Quaternion q(1.0, 2.0, 3.0, 4.0);
    static_assert(q == q * Quaternion::identity());
}

TEST_CASE("Quaternion: associativity")
{
    constexpr Quaternion q1(1.0, 2.0, 3.0, 4.0);
    constexpr Quaternion q2(5.0, 6.0, 7.0, 8.0);
    constexpr Quaternion q3(9.0, 10.0, 11.0, 12.0);
    static_assert((q1 * q2) * q3 == q1 * (q2 * q3));
}

TEST_CASE("Quaternion: distributivity")
{
    constexpr Quaternion q1(1.0, 2.0, 3.0, 4.0);
    constexpr Quaternion q2(5.0, 6.0, 7.0, 8.0);
    constexpr Quaternion q3(9.0, 10.0, 11.0, 12.0);
    static_assert(q1 * (q2 + q3) == q1 * q2 + q1 * q3);
}

TEST_CASE("Quaternion: vector rotation")
{
    constexpr double angle = PI / 2.0;
    constexpr Quaternion q(gcem::cos(angle / 2.0), 0.0, 0.0, gcem::sin(angle / 2.0));
    constexpr Vector3 v(1.0, 0.0, 0.0);
    constexpr auto rotated = q * v;

    static_assert(rotated.x == 0.0);
    static_assert(rotated.y == 1.0);
    static_assert(rotated.z == 0.0);
}

TEST_CASE("Quaternion: rotation preserves norm")
{
    constexpr Quaternion q(0.70710678, 0.70710678, 0.0, 0.0);
    constexpr Vector3 v(0.0, 1.0, 0.0);
    constexpr auto rotated = q * v;
    static_assert(nearly_equal(v.norm(), rotated.norm()));
}

TEST_CASE("Quaternion: compound assignment")
{
    Quaternion q(1.0, 0.0, 0.0, 0.0);
    q *= 2.0;
    CHECK(q == Quaternion(2.0, 0.0, 0.0, 0.0));

    Quaternion q2(1.0, 0.0, 0.0, 0.0);
    q2 += Quaternion(0.0, 1.0, 0.0, 0.0);
    CHECK(q2 == Quaternion(1.0, 1.0, 0.0, 0.0));
}

// =============================================================================
// SLERP
// =============================================================================

TEST_CASE("SLERP: endpoints")
{
    constexpr Quaternion a(1.0, 0.0, 0.0, 0.0);
    constexpr double half_angle = PI / 4.0;
    constexpr Quaternion b(gcem::cos(half_angle), 0.0, 0.0, gcem::sin(half_angle));

    constexpr auto at_0 = slerp(a, b, 0.0);
    CHECK(nearly_equal(at_0.w, a.w));
    CHECK(nearly_equal(at_0.x, a.x));
    CHECK(nearly_equal(at_0.y, a.y));
    CHECK(nearly_equal(at_0.z, a.z));

    constexpr auto at_1 = slerp(a, b, 1.0);
    CHECK(nearly_equal(at_1.w, b.w));
    CHECK(nearly_equal(at_1.z, b.z));
}

TEST_CASE("SLERP: midpoint preserves unit norm")
{
    constexpr Quaternion a(1.0, 0.0, 0.0, 0.0);
    constexpr double half_angle = PI / 4.0;
    constexpr Quaternion b(gcem::cos(half_angle), 0.0, 0.0, gcem::sin(half_angle));

    constexpr auto mid = slerp(a, b, 0.5);
    CHECK(nearly_equal(mid.norm(), 1.0));
}

TEST_CASE("SLERP: nearly identical quaternions (linear fallback)")
{
    constexpr Quaternion a(1.0, 0.0, 0.0, 0.0);
    constexpr Quaternion b(0.99999, 0.00001, 0.0, 0.0);
    constexpr auto mid = slerp(a, b, 0.5);
    CHECK(nearly_equal(mid.norm(), 1.0));
}

TEST_CASE("SLERP: opposite quaternions take shorter path")
{
    constexpr Quaternion a(1.0, 0.0, 0.0, 0.0);
    constexpr Quaternion b(-1.0, 0.0, 0.0, 0.0);
    constexpr auto mid = slerp(a, b, 0.5);
    CHECK(nearly_equal(mid.norm(), 1.0));
}

// =============================================================================
// Cross-type operations (linalg.hpp)
// =============================================================================

TEST_CASE("quaternion_to_matrix")
{
    constexpr auto m = quaternion_to_matrix(Quaternion::identity());
    static_assert(m == Matrix3::identity());
}

TEST_CASE("quaternion_to_euler_angles roundtrip")
{
    constexpr EulerAngles e(0.1, 0.2, 0.3);
    constexpr auto q = euler_angles_to_quaternion(e);
    constexpr auto e2 = quaternion_to_euler_angles(q);

    CHECK(nearly_equal(e.pitch.value(), e2.pitch.value()));
    CHECK(nearly_equal(e.yaw.value(), e2.yaw.value()));
    CHECK(nearly_equal(e.roll.value(), e2.roll.value()));
}

TEST_CASE("euler_angles_to_vector3 and back")
{
    constexpr EulerAngles<AngleType::RADIANS> e(0.1, 0.2, 0.3);
    constexpr auto v = euler_angles_to_vector3(e);
    static_assert(v.x == 0.1);
    static_assert(v.z == 0.2);
    static_assert(v.y == 0.3);
}

TEST_CASE("quaternion_to_vector3")
{
    constexpr Quaternion q(0.5, 1.0, 2.0, 3.0);
    constexpr auto v = quaternion_to_vector3(q);
    static_assert(v.x == 1.0 && v.y == 2.0 && v.z == 3.0);
}

TEST_CASE("Vector3 * Quaternion rotation")
{
    constexpr Quaternion q(1.0, 0.0, 0.0, 0.0);
    constexpr Vector3 v(1.0, 2.0, 3.0);
    constexpr auto result = v * q;
    static_assert(nearly_equal(result.x, v.x));
    static_assert(nearly_equal(result.y, v.y));
    static_assert(nearly_equal(result.z, v.z));
}

// =============================================================================
// fmt formatters
// =============================================================================

TEST_CASE("fmt: Vector2")
{
    CHECK(fmt::format("{}", Vector2(1.0, 2.0)) == "(1, 2)");
}

TEST_CASE("fmt: Vector3")
{
    CHECK(fmt::format("{}", Vector3(1.0, 2.0, 3.0)) == "(1, 2, 3)");
}

TEST_CASE("fmt: Vector4")
{
    CHECK(fmt::format("{}", Vector4(1.0, 2.0, 3.0, 4.0)) == "(1, 2, 3, 4)");
}

TEST_CASE("fmt: Quaternion")
{
    auto s = fmt::format("{}", Quaternion::identity());
    CHECK(s == "Quaternion(w=1, x=0, y=0, z=0)");
}

TEST_CASE("fmt: Angle")
{
    CHECK(fmt::format("{}", Angle<AngleType::RADIANS>(1.0)).find("rad") != std::string::npos);
    CHECK(fmt::format("{}", Angle<AngleType::DEGREES>(90.0)).find("deg") != std::string::npos);
}

TEST_CASE("fmt: Matrix2")
{
    auto s = fmt::format("{}", Matrix2::identity());
    CHECK(s == "[[1, 0], [0, 1]]");
}

TEST_CASE("fmt: Matrix3")
{
    auto s = fmt::format("{}", Matrix3::identity());
    CHECK(s == "[[1, 0, 0], [0, 1, 0], [0, 0, 1]]");
}

// =============================================================================
// Strong type interop
// =============================================================================

TEST_CASE("Angle: get() returns native value")
{
    constexpr Angle<AngleType::RADIANS> rad(PI);
    static_assert(rad.get() == PI);

    constexpr Angle<AngleType::DEGREES> deg(180.0);
    static_assert(deg.get() == 180.0);
}

TEST_CASE("Angle: from_strong (radians)")
{
    constexpr MockRadian r{PI};
    constexpr auto a = Angle<AngleType::RADIANS>::from_strong(r);
    static_assert(linalg3d::fabs(a.value() - PI) < EPSILON);
}

TEST_CASE("Angle: from_strong (degrees)")
{
    constexpr MockRadian r{PI};
    constexpr auto a = Angle<AngleType::DEGREES>::from_strong(r);
    static_assert(linalg3d::fabs(a.value() - 180.0) < EPSILON);
}

TEST_CASE("Angle: to_strong roundtrip")
{
    constexpr Angle<AngleType::RADIANS> a(PI / 4.0);
    constexpr auto s = a.to_strong<MockRadian>();
    static_assert(linalg3d::fabs(s.get() - PI / 4.0) < EPSILON);

    constexpr Angle<AngleType::DEGREES> deg(90.0);
    constexpr auto s2 = deg.to_strong<MockRadian>();
    static_assert(linalg3d::fabs(s2.get() - PI / 2.0) < EPSILON);
}

TEST_CASE("EulerAngles: from strong types")
{
    constexpr MockRadian p{0.1};
    constexpr MockRadian y{0.2};
    constexpr MockRadian r{0.3};
    constexpr EulerAngles<AngleType::RADIANS> e(p, y, r);
    static_assert(linalg3d::fabs(e.pitch.value() - 0.1) < EPSILON);
    static_assert(linalg3d::fabs(e.yaw.value() - 0.2) < EPSILON);
    static_assert(linalg3d::fabs(e.roll.value() - 0.3) < EPSILON);
}

TEST_CASE("EulerAngles: from strong types to degrees")
{
    constexpr MockRadian p{PI / 2.0};
    constexpr EulerAngles<AngleType::DEGREES> e(p, p, p);
    CHECK(nearly_equal(e.pitch.value(), 90.0));
    CHECK(nearly_equal(e.yaw.value(), 90.0));
    CHECK(nearly_equal(e.roll.value(), 90.0));
}
