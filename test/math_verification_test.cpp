#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <numbers>

#include <linalg3d/linalg.hpp>

#include <Eigen/Dense>
#include <cmath>
#include <random>

using namespace linalg3d;

// ============================================================================
// Random generators
// ============================================================================

namespace
{

std::mt19937 rng{42};
std::uniform_real_distribution<double> dist{-1000.0, 1000.0};
std::uniform_real_distribution<double> small_dist{-1e-10, 1e-10};
std::uniform_real_distribution<double> unit_dist{-1.0, 1.0};

Vector3 random_vec3()
{
    return Vector3{dist(rng), dist(rng), dist(rng)};
}

Vector3 random_nonzero_vec3()
{
    Vector3 v;
    do
    {
        v = random_vec3();
    } while (v.norm() < 1e-12);
    return v;
}


Quaternion random_unit_quaternion()
{
    // Generate random unit quaternion via normalized random 4-vector
    double w = unit_dist(rng);
    double x = unit_dist(rng);
    double y = unit_dist(rng);
    double z = unit_dist(rng);
    double n = std::sqrt(w * w + x * x + y * y + z * z);
    if (n < 1e-12)
    {
        return Quaternion{1.0, 0.0, 0.0, 0.0};
    }
    return Quaternion{w / n, x / n, y / n, z / n};
}

Matrix3 random_invertible_matrix3()
{
    // Random matrix with determinant check
    for (int attempt = 0; attempt < 100; ++attempt)
    {
        Matrix3 m{dist(rng), dist(rng), dist(rng), dist(rng), dist(rng), dist(rng), dist(rng), dist(rng), dist(rng)};
        auto det = m.determinant();
        if (std::abs(det) > 1e-6)
        {
            return m;
        }
    }
    // Fallback: identity
    return Matrix3{1, 0, 0, 0, 1, 0, 0, 0, 1};
}

constexpr double EPSILON = 1e-9;
constexpr int N_TRIALS = 10000;

} // namespace

// ============================================================================
// Vector3 property-based tests
// ============================================================================

TEST_CASE("Vector3 — normalize produces unit length")
{
    for (int i = 0; i < N_TRIALS; ++i)
    {
        auto v = random_nonzero_vec3();
        auto n = v.normalized();
        CHECK(n.norm() == doctest::Approx(1.0).epsilon(EPSILON));
    }
}

TEST_CASE("Vector3 — dot product is commutative")
{
    for (int i = 0; i < N_TRIALS; ++i)
    {
        auto a = random_vec3();
        auto b = random_vec3();
        CHECK(a.dot(b) == doctest::Approx(b.dot(a)).epsilon(EPSILON));
    }
}

TEST_CASE("Vector3 — cross product is anti-commutative")
{
    for (int i = 0; i < N_TRIALS; ++i)
    {
        auto a = random_vec3();
        auto b = random_vec3();
        auto axb = a.cross(b);
        auto bxa = b.cross(a);
        CHECK(axb.x == doctest::Approx(-bxa.x).epsilon(EPSILON));
        CHECK(axb.y == doctest::Approx(-bxa.y).epsilon(EPSILON));
        CHECK(axb.z == doctest::Approx(-bxa.z).epsilon(EPSILON));
    }
}

TEST_CASE("Vector3 — cross product is perpendicular to both inputs")
{
    for (int i = 0; i < N_TRIALS; ++i)
    {
        auto a = random_nonzero_vec3();
        auto b = random_nonzero_vec3();
        auto c = a.cross(b);
        // dot(a, cross(a,b)) == 0
        double dot_a = a.dot(c);
        double dot_b = b.dot(c);
        CHECK(dot_a == doctest::Approx(0.0).epsilon(1e-6));
        CHECK(dot_b == doctest::Approx(0.0).epsilon(1e-6));
    }
}

TEST_CASE("Vector3 — norm squared equals dot with self")
{
    for (int i = 0; i < N_TRIALS; ++i)
    {
        auto v = random_vec3();
        double norm_sq = v.norm() * v.norm();
        double dot_self = v.dot(v);
        CHECK(norm_sq == doctest::Approx(dot_self).epsilon(EPSILON));
    }
}

TEST_CASE("Vector3 — triangle inequality")
{
    for (int i = 0; i < N_TRIALS; ++i)
    {
        auto a = random_vec3();
        auto b = random_vec3();
        auto sum = a + b;
        CHECK(sum.norm() <= a.norm() + b.norm() + EPSILON);
    }
}

TEST_CASE("Vector3 — scalar homogeneity: norm(s*v) == |s| * norm(v)")
{
    for (int i = 0; i < N_TRIALS; ++i)
    {
        auto v = random_vec3();
        double s = dist(rng);
        auto sv = v * s;
        CHECK(sv.norm() == doctest::Approx(std::abs(s) * v.norm()).epsilon(1e-6));
    }
}

TEST_CASE("Vector3 — addition is commutative")
{
    for (int i = 0; i < N_TRIALS; ++i)
    {
        auto a = random_vec3();
        auto b = random_vec3();
        auto ab = a + b;
        auto ba = b + a;
        CHECK(ab.x == doctest::Approx(ba.x).epsilon(EPSILON));
        CHECK(ab.y == doctest::Approx(ba.y).epsilon(EPSILON));
        CHECK(ab.z == doctest::Approx(ba.z).epsilon(EPSILON));
    }
}

TEST_CASE("Vector3 — subtraction: a - a == zero")
{
    for (int i = 0; i < N_TRIALS; ++i)
    {
        auto a = random_vec3();
        auto zero = a - a;
        CHECK(zero.x == doctest::Approx(0.0).epsilon(EPSILON));
        CHECK(zero.y == doctest::Approx(0.0).epsilon(EPSILON));
        CHECK(zero.z == doctest::Approx(0.0).epsilon(EPSILON));
    }
}

// ============================================================================
// Matrix3 property-based tests
// ============================================================================

TEST_CASE("Matrix3 — M * inverse(M) ≈ identity")
{
    int verified = 0;
    for (int i = 0; i < N_TRIALS; ++i)
    {
        auto m = random_invertible_matrix3();
        auto inv = m.inverse();
        if (!inv.has_value())
        {
            continue;
        }
        auto product = m * inv.value();
        // Check diagonal ≈ 1, off-diagonal ≈ 0
        CHECK(product.m[0][0] == doctest::Approx(1.0).epsilon(1e-6));
        CHECK(product.m[1][1] == doctest::Approx(1.0).epsilon(1e-6));
        CHECK(product.m[2][2] == doctest::Approx(1.0).epsilon(1e-6));
        CHECK(product.m[0][1] == doctest::Approx(0.0).epsilon(1e-6));
        CHECK(product.m[0][2] == doctest::Approx(0.0).epsilon(1e-6));
        CHECK(product.m[1][0] == doctest::Approx(0.0).epsilon(1e-6));
        CHECK(product.m[1][2] == doctest::Approx(0.0).epsilon(1e-6));
        CHECK(product.m[2][0] == doctest::Approx(0.0).epsilon(1e-6));
        CHECK(product.m[2][1] == doctest::Approx(0.0).epsilon(1e-6));
        ++verified;
    }
    CHECK(verified > N_TRIALS / 2); // most random matrices should be invertible
}

TEST_CASE("Matrix3 — det(M*N) ≈ det(M) * det(N)")
{
    for (int i = 0; i < N_TRIALS; ++i)
    {
        auto m = random_invertible_matrix3();
        auto n = random_invertible_matrix3();
        auto mn = m * n;
        double det_mn = mn.determinant();
        double det_m_det_n = m.determinant() * n.determinant();
        CHECK(det_mn == doctest::Approx(det_m_det_n).epsilon(1e-3));
    }
}

// ============================================================================
// Quaternion property-based tests
// ============================================================================

TEST_CASE("Quaternion — unit quaternion has norm 1")
{
    for (int i = 0; i < N_TRIALS; ++i)
    {
        auto q = random_unit_quaternion();
        double norm = std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
        CHECK(norm == doctest::Approx(1.0).epsilon(EPSILON));
    }
}

TEST_CASE("Quaternion — rotation preserves vector length")
{
    for (int i = 0; i < N_TRIALS; ++i)
    {
        auto q = random_unit_quaternion();
        auto v = random_vec3();
        auto rotated = q * v;
        CHECK(rotated.norm() == doctest::Approx(v.norm()).epsilon(1e-6));
    }
}

TEST_CASE("Quaternion — slerp endpoints: slerp(a,b,0)≈a, slerp(a,b,1)≈b")
{
    for (int i = 0; i < 1000; ++i)
    {
        auto a = random_unit_quaternion();
        auto b = random_unit_quaternion();

        auto s0 = slerp(a, b, 0.0);
        // slerp may flip sign (equivalent quaternion)
        double dot_a = std::abs(s0.w * a.w + s0.x * a.x + s0.y * a.y + s0.z * a.z);
        CHECK(dot_a == doctest::Approx(1.0).epsilon(1e-4));

        auto s1 = slerp(a, b, 1.0);
        double dot_b = std::abs(s1.w * b.w + s1.x * b.x + s1.y * b.y + s1.z * b.z);
        CHECK(dot_b == doctest::Approx(1.0).epsilon(1e-4));
    }
}

TEST_CASE("Quaternion — double rotation equals rotation by double angle")
{
    for (int i = 0; i < N_TRIALS; ++i)
    {
        auto q = random_unit_quaternion();
        auto v = random_vec3();
        auto once = q * v;
        auto twice = q * once;
        auto qq = q * q;
        auto direct = qq.normalized() * v;
        CHECK(twice.x == doctest::Approx(direct.x).epsilon(1e-6));
        CHECK(twice.y == doctest::Approx(direct.y).epsilon(1e-6));
        CHECK(twice.z == doctest::Approx(direct.z).epsilon(1e-6));
    }
}

// ============================================================================
// Cross-validation against Eigen
// ============================================================================

TEST_CASE("Cross-validate Vector3 — dot product matches Eigen")
{
    for (int i = 0; i < N_TRIALS; ++i)
    {
        auto a = random_vec3();
        auto b = random_vec3();

        double our_dot = a.dot(b);

        Eigen::Vector3d ea{a.x, a.y, a.z};
        Eigen::Vector3d eb{b.x, b.y, b.z};
        double eigen_dot = ea.dot(eb);

        CHECK(our_dot == doctest::Approx(eigen_dot).epsilon(EPSILON));
    }
}

TEST_CASE("Cross-validate Vector3 — cross product matches Eigen")
{
    for (int i = 0; i < N_TRIALS; ++i)
    {
        auto a = random_vec3();
        auto b = random_vec3();

        auto our_cross = a.cross(b);

        Eigen::Vector3d ea{a.x, a.y, a.z};
        Eigen::Vector3d eb{b.x, b.y, b.z};
        Eigen::Vector3d eigen_cross = ea.cross(eb);

        CHECK(our_cross.x == doctest::Approx(eigen_cross.x()).epsilon(EPSILON));
        CHECK(our_cross.y == doctest::Approx(eigen_cross.y()).epsilon(EPSILON));
        CHECK(our_cross.z == doctest::Approx(eigen_cross.z()).epsilon(EPSILON));
    }
}

TEST_CASE("Cross-validate Vector3 — norm matches Eigen")
{
    for (int i = 0; i < N_TRIALS; ++i)
    {
        auto v = random_vec3();
        double our_norm = v.norm();

        Eigen::Vector3d ev{v.x, v.y, v.z};
        double eigen_norm = ev.norm();

        CHECK(our_norm == doctest::Approx(eigen_norm).epsilon(EPSILON));
    }
}

TEST_CASE("Cross-validate Matrix3 — determinant matches Eigen")
{
    for (int i = 0; i < N_TRIALS; ++i)
    {
        auto m = random_invertible_matrix3();
        double our_det = m.determinant();

        Eigen::Matrix3d em;
        em << m.m[0][0], m.m[0][1], m.m[0][2], m.m[1][0], m.m[1][1], m.m[1][2], m.m[2][0], m.m[2][1], m.m[2][2];
        double eigen_det = em.determinant();

        CHECK(our_det == doctest::Approx(eigen_det).epsilon(1e-3));
    }
}

TEST_CASE("Cross-validate Matrix3 — inverse matches Eigen")
{
    int verified = 0;
    for (int i = 0; i < N_TRIALS; ++i)
    {
        auto m = random_invertible_matrix3();
        auto our_inv = m.inverse();
        if (!our_inv.has_value())
        {
            continue;
        }

        Eigen::Matrix3d em;
        em << m.m[0][0], m.m[0][1], m.m[0][2], m.m[1][0], m.m[1][1], m.m[1][2], m.m[2][0], m.m[2][1], m.m[2][2];
        Eigen::Matrix3d eigen_inv = em.inverse();

        auto &oi = our_inv.value();
        CHECK(oi.m[0][0] == doctest::Approx(eigen_inv(0, 0)).epsilon(1e-4));
        CHECK(oi.m[0][1] == doctest::Approx(eigen_inv(0, 1)).epsilon(1e-4));
        CHECK(oi.m[0][2] == doctest::Approx(eigen_inv(0, 2)).epsilon(1e-4));
        CHECK(oi.m[1][0] == doctest::Approx(eigen_inv(1, 0)).epsilon(1e-4));
        CHECK(oi.m[1][1] == doctest::Approx(eigen_inv(1, 1)).epsilon(1e-4));
        CHECK(oi.m[1][2] == doctest::Approx(eigen_inv(1, 2)).epsilon(1e-4));
        CHECK(oi.m[2][0] == doctest::Approx(eigen_inv(2, 0)).epsilon(1e-4));
        CHECK(oi.m[2][1] == doctest::Approx(eigen_inv(2, 1)).epsilon(1e-4));
        CHECK(oi.m[2][2] == doctest::Approx(eigen_inv(2, 2)).epsilon(1e-4));
        ++verified;
    }
    CHECK(verified > N_TRIALS / 2);
}

// ============================================================================
// Edge cases
// ============================================================================

TEST_CASE("Edge — zero vector norm is 0")
{
    Vector3 v{0, 0, 0};
    CHECK(v.norm() == 0.0);
}

TEST_CASE("Edge — axis-aligned unit vectors")
{
    CHECK(Vector3{1, 0, 0}.norm() == doctest::Approx(1.0));
    CHECK(Vector3{0, 1, 0}.norm() == doctest::Approx(1.0));
    CHECK(Vector3{0, 0, 1}.norm() == doctest::Approx(1.0));

    // Cross products of basis vectors
    auto xy = Vector3{1, 0, 0}.cross(Vector3{0, 1, 0});
    CHECK(xy.z == doctest::Approx(1.0));
}

TEST_CASE("Edge — very large values")
{
    Vector3 big{1e15, 1e15, 1e15};
    auto n = big.normalized();
    CHECK(n.norm() == doctest::Approx(1.0).epsilon(1e-6));
}

TEST_CASE("Edge — very small values")
{
    Vector3 tiny{1e-15, 1e-15, 1e-15};
    CHECK(tiny.norm() > 0.0);
    auto n = tiny.normalized();
    CHECK(n.norm() == doctest::Approx(1.0).epsilon(1e-4));
}

TEST_CASE("Edge — parallel vectors cross product is zero")
{
    Vector3 a{3.0, 0.0, 0.0};
    Vector3 b{7.0, 0.0, 0.0};
    auto c = a.cross(b);
    CHECK(c.norm() == doctest::Approx(0.0).epsilon(EPSILON));
}

TEST_CASE("Edge — identity quaternion preserves vector")
{
    Quaternion id{1, 0, 0, 0};
    for (int i = 0; i < 100; ++i)
    {
        auto v = random_vec3();
        auto rotated = id * v;
        CHECK(rotated.x == doctest::Approx(v.x).epsilon(EPSILON));
        CHECK(rotated.y == doctest::Approx(v.y).epsilon(EPSILON));
        CHECK(rotated.z == doctest::Approx(v.z).epsilon(EPSILON));
    }
}

TEST_CASE("Edge — 90 degree rotation around Z axis")
{
    // q = cos(45°) + sin(45°) * k
    double s = std::sin(std::numbers::pi / 4.0);
    double c = std::cos(std::numbers::pi / 4.0);
    Quaternion q{c, 0, 0, s};
    Vector3 v{1, 0, 0};
    auto rotated = q * v;
    CHECK(rotated.x == doctest::Approx(0.0).epsilon(1e-9));
    CHECK(rotated.y == doctest::Approx(1.0).epsilon(1e-9));
    CHECK(rotated.z == doctest::Approx(0.0).epsilon(1e-9));
}

// ============================================================================
// Angle tests
// ============================================================================

TEST_CASE("Angle — radian/degree conversion round-trip")
{
    for (int i = 0; i < N_TRIALS; ++i)
    {
        double val = dist(rng);
        Angle<RADIANS> rad{val};
        auto deg = rad.to_degrees();
        auto back = deg.to_radians();
        CHECK(back.value() == doctest::Approx(val).epsilon(1e-9));
    }
}

TEST_CASE("Angle — 180 degrees == pi radians")
{
    Angle<DEGREES> deg{180.0};
    auto rad = deg.to_radians();
    CHECK(rad.value() == doctest::Approx(std::numbers::pi).epsilon(1e-12));
}
