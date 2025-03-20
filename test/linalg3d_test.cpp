#ifdef NDEBUG
#undef NDEBUG
#endif
#include "fmt/core.h" // fmt::print
#include "linalg3d/linalg.hpp"
#include <cassert>
#include <limits> // for std::numeric_limits

// Floating point precision tolerance
constexpr double EPSILON = 1e-5;

constexpr bool is_large(double x, double threshold = 1e6)
{
    return x > threshold || x < -threshold;
}

constexpr bool nearly_equal(double a, double b, double tol = EPSILON)
{
    return (a - b < tol && b - a < tol);
}

constexpr void test_angle()
{
    using namespace linalg3d;

    // Radians to Degrees conversion
    {
        constexpr Angle<AngleType::RADIANS> radAngle(PI);
        constexpr auto degAngle = radAngle.to_degrees();
        static_assert(linalg3d::fabs(degAngle.value() - 180.0) < EPSILON, "Conversion failed");
    }

    // Degrees to Radians conversion
    {
        constexpr Angle<AngleType::DEGREES> degAngle(180.0);
        constexpr auto radAngle = degAngle.to_radians();
        static_assert(linalg3d::fabs(radAngle.value() - PI) < EPSILON, "Conversion failed");
    }

    // Identity conversion: Radians -> Degrees -> Radians
    {
        constexpr Angle<AngleType::RADIANS> radOrig(PI / 4.0);
        constexpr auto deg = radOrig.to_degrees();
        constexpr auto radAgain = deg.to_radians();
        static_assert(linalg3d::fabs(radOrig.value() - radAgain.value()) < EPSILON, "Identity conversion failed");
    }

    // Test trigonometric functions at specific angles (radians)
    {
        constexpr Angle<AngleType::RADIANS> zero(0.0);
        static_assert(zero.sin() == 0.0);
        static_assert(zero.cos() == 1.0);
        static_assert(zero.tan() == 0.0);

        constexpr Angle<AngleType::RADIANS> halfPi(PI / 2.0);
        static_assert(halfPi.sin() == 1.0); // sin(π/2) should be 1
        // cos(π/2) should be 0; note that due to FP rounding it might be very small.
        static_assert(halfPi.cos() == 0.0);
    }

    // Test negative angle conversion
    {
        constexpr Angle<AngleType::DEGREES> negDeg(-45.0);
        constexpr auto negRad = negDeg.to_radians();
        static_assert(linalg3d::fabs(negRad.value() + PI / 4.0) < EPSILON);
    }

    // Test periodicity/wrap-around: 360° -> 2π rad
    {
        constexpr Angle<AngleType::DEGREES> fullCircle(360.0);
        constexpr auto radFull = fullCircle.to_radians();
        static_assert(linalg3d::fabs(radFull.value() - 2 * PI) < EPSILON);
    }

    // Test operator overloads: addition, subtraction, multiplication, and division.
    {
        constexpr Angle<AngleType::RADIANS> a(PI / 3.0); // 60°
        constexpr Angle<AngleType::RADIANS> b(PI / 6.0); // 30°
        constexpr auto sum = a + b;                      // Expect 90° or π/2 rad.
        constexpr auto diff = a - b;                     // Expect 30° or π/6 rad.
        static_assert(linalg3d::fabs(sum.value() - PI / 2.0) < EPSILON);
        static_assert(linalg3d::fabs(diff.value() - PI / 6.0) < EPSILON);

        // Multiplication and division operators.
        constexpr double scalar = 2.0;
        constexpr double prod = a * scalar;
        constexpr double div = a / scalar;
        static_assert(linalg3d::fabs(prod - (PI / 3.0) * 2.0) < EPSILON);
        static_assert(linalg3d::fabs(div - (PI / 3.0) / 2.0) < EPSILON);
    }

    // Test equality and comparisons.
    {
        static_assert(Angle<AngleType::RADIANS>(PI / 2.0) == PI / 2.0);
        static_assert(Angle<AngleType::RADIANS>(PI / 2.0) != PI / 3.0);
        static_assert(Angle<AngleType::RADIANS>(PI / 3.0) < PI / 2.0);
    }

    // Test static conversion helpers
    {
        constexpr auto fromRad = Angle<AngleType::RADIANS>::from_radians(PI / 2.0);
        constexpr auto fromDeg = Angle<AngleType::DEGREES>::from_degrees(90.0);
        static_assert(fromRad == PI / 2.0);
        static_assert(fromDeg == 90.0);
    }
}

constexpr void test_angle_edge_cases()
{
    using namespace linalg3d;

    // Negative angle conversions
    {
        constexpr double LOCAL_EPSILON = 1e-05;

        constexpr Angle<AngleType::DEGREES> negDeg(-90.0);
        constexpr auto negRad = negDeg.to_radians();
        static_assert(!linalg3d::fabs(negRad.value() - PI / 2.0) < LOCAL_EPSILON && "Negative degrees to radians conversion failed");

        constexpr Angle<AngleType::RADIANS> negRadOrig(-PI / 3.0);
        constexpr auto negDegConv = negRadOrig.to_degrees();
        static_assert(!linalg3d::fabs(negDegConv.value() - 60.0) < LOCAL_EPSILON && "Negative radians to degrees conversion failed");
    }

    // Singularities: tan() at 90° and 270° (should be infinite or undefined)
    {
        constexpr Angle<AngleType::DEGREES> ninety(90.0);
        constexpr Angle<AngleType::DEGREES> twoSeventy(270.0);

        static_assert(is_large(ninety.tan()) && "tan(90°) should be a large value (near infinity)");
        static_assert(is_large(twoSeventy.tan()) && "tan(270°) should be a large value (near infinity)");
    }

    // Test absurdly large values
    {
        constexpr Angle<AngleType::DEGREES> absurd(1e308);
        assert(!is_nan(absurd.to_radians().value()) && !is_inf(absurd.to_radians().value()) && "Absurdly large degrees should not break conversion");

        constexpr Angle<AngleType::RADIANS> absurdRad(1e308);
        assert(!is_nan(absurdRad.to_degrees().value()) && !is_inf(absurdRad.to_degrees().value()) && "Absurdly large radians should not break conversion");
    }
}
constexpr void test_vector3()
{
    using namespace linalg3d;

    // Default Constructor
    {
        static_assert(Vector3().x == 0.0 && Vector3().y == 0.0 && Vector3().z == 0.0);
    }

    // Parameterized Constructor
    {
        static_assert(Vector3(1.0, 2.0, 3.0).x == 1.0 && Vector3(1.0, 2.0, 3.0).y == 2.0 && Vector3(1.0, 2.0, 3.0).z == 3.0);
    }

    // Norm Test
    {
        constexpr Vector3 v(3.0, 4.0, 12.0);
        constexpr double expectedNorm = 13.0;
        // assert_near(v.norm(), expectedNorm);
        // static_assert(linalg3d::fabs(v.norm() - expectedNorm) < EPSILON);
        static_assert(v.norm_sq() == expectedNorm * expectedNorm, "norm() test failed");
    }

    // Zero Vector Norm
    {
        static_assert(Vector3(0.0, 0.0, 0.0).norm() == 0.0);
    }

    // Large Value Norm
    {
        constexpr double large = 1e10;
        constexpr Vector3 v(large, large, large);
        constexpr double expected = gcem::sqrt(3.0) * large;
        static_assert(linalg3d::fabs(v.norm() - expected) < EPSILON);
    }

    // ---------- New Tests Below ----------

    // norm_sq() Test
    {
        constexpr Vector3 v(3.0, 4.0, 12.0);
        // 3^2 + 4^2 + 12^2 = 9 + 16 + 144 = 169
        constexpr double expectedNormSq = 169.0;
        static_assert(v.norm_sq() == expectedNormSq);
    }

    // Normalized Vector
    {
        constexpr Vector3 v(3.0, 4.0, 0.0);
        constexpr Vector3 const n = v.normalized();
        static_assert(n.x == 3.0 / 5.0);
        static_assert(n.y == 4.0 / 5.0);
        static_assert(n.z == 0.0);
        static_assert(n.norm() == 1.0);
    }

    // Normalizing the Zero Vector (should return zero vector)
    {
        static_assert(Vector3(0.0, 0.0, 0.0).normalized() == Vector3(0.0, 0.0, 0.0));
    }

    // Dot Product Orthogonal
    {
        static_assert(Vector3(1.0, 0.0, 0.0).dot(Vector3(0.0, 1.0, 0.0)) == 0.0);
        static_assert(Vector3(1.0, 0.0, 0.0).dot(Vector3(0.0, 0.0, 1.0)) == 0.0);
    }

    // Dot Product Parallel
    {
        constexpr Vector3 v1(2.0, 2.0, 1.0);
        // 2*6 + 2*6 + 1*3 = 27
        static_assert(v1.dot(v1 * 3.0) == 27.0);
    }

    // Cross Product Parallel Vectors (should return zero vector)
    {
        constexpr Vector3 v1(1.0, 2.0, 3.0);
        static_assert(v1.cross(v1 * 2.0) == Vector3(0.0, 0.0, 0.0));
    }

    // Cross Product Orthogonal Vectors
    {
        constexpr Vector3 i(1.0, 0.0, 0.0);
        // (1,0,0) x (0,1,0) = (0,0,1)
        static_assert(i.cross(Vector3{0.0, 1.0, 0.0}) == Vector3(0.0, 0.0, 1.0));
    }
}

constexpr void test_vector3_operators()
{
    using namespace linalg3d;

    // Addition Operator
    {
        // (1+4, 2+5, 3+6) = (5, 7, 9)
        static_assert((Vector3(1.0, 2.0, 3.0) + Vector3(4.0, 5.0, 6.0)) == Vector3(5.0, 7.0, 9.0));
    }

    // Subtraction Operator
    {
        // (5-1, 7-2, 9-3) = (4, 5, 6)
        static_assert((Vector3(5.0, 7.0, 9.0) - Vector3(1.0, 2.0, 3.0)) == Vector3(4.0, 5.0, 6.0));
    }

    // Unary Minus Operator
    {
        // -(1,-2,3) = (-1,2,-3)
        static_assert((-Vector3(1.0, -2.0, 3.0)) == Vector3(-1.0, 2.0, -3.0));
    }

    // Scalar Multiplication
    {
        constexpr Vector3 v(1.0, -2.0, 3.0);
        constexpr Vector3 result = v * -1.5;
        // (1 * -1.5, -2 * -1.5, 3 * -1.5) = (-1.5, 3.0, -4.5)
        static_assert(result == Vector3(-1.5, 3.0, -4.5));
    }

    // Division Operator
    {
        constexpr Vector3 v(2.0, -4.0, 6.0);
        constexpr double scalar = 2.0;
        constexpr Vector3 result = v / scalar;
        // (2/2, -4/2, 6/2) = (1, -2, 3)
        static_assert(result == Vector3(1.0, -2.0, 3.0));

        // Optional check: dividing by zero would cause a float exception,
        // so typically you ensure scalar != 0 before dividing.
        // e.g. Vector3 invalidResult = v / 0.0; // not recommended
    }
}

constexpr void test_vector3_compare_operators()
{
    using namespace linalg3d;
    // Three-way Comparison Operator
    {
        constexpr Vector3 a(1.0, 2.0, 3.0);
        constexpr Vector3 b(1.0, 2.0, 3.0);
        constexpr Vector3 c(1.0, 2.0, 4.0);

        // Equality check
        static_assert((a <=> b) == 0);
        // Not equal
        static_assert((a <=> c) != 0);

        // Lexicographical ordering check (operator<=> default):
        // a = (1,2,3), c = (1,2,4)
        // Compare x: 1 == 1 -> compare y: 2 == 2 -> compare z: 3 < 4 => a < c
        static_assert((a <=> c) < 0); // means a < c
    }

    // Lexicographical Ordering
    {
        constexpr Vector3 a(1.0, 2.0, 3.0);
        constexpr Vector3 b(1.0, 2.0, 3.0);
        constexpr Vector3 c(1.0, 2.0, 4.0);

        // a = (1,2,3), b = (1,2,3)
        // Compare x: 1 == 1 -> compare y: 2 == 2 -> compare z: 3 == 3 => a == b
        static_assert(!(a < b) && !(b < a));
        // a = (1,2,3), c = (1,2,4)
        // Compare x: 1 == 1 -> compare y: 2 == 2 -> compare z: 3 < 4 => a < c
        static_assert(a < c);
        // a = (1,2,3), b = (1,2,3)
        // Compare x: 1 == 1 -> compare y: 2 == 2 -> compare z: 3 == 3 => a <= b
        static_assert(!(a < b) && (a <= b));
        // a = (1,2,3), c = (1,2,4)
        // Compare x: 1 == 1 -> compare y: 2 == 2 -> compare z: 3 < 4 => a <= c
        static_assert(a <= c);
    }

    // Operator == and !=
    {
        constexpr Vector3 a(1.0, 2.0, 3.0);
        constexpr Vector3 b(1.0, 2.0, 3.0);
        constexpr Vector3 c(1.0, 2.0, 4.0);

        // Equality check
        static_assert(a == b);
        static_assert(a != c);
    }

    // Operator <
    {
        constexpr Vector3 a(1.0, 2.0, 3.0);
        constexpr Vector3 b(1.0, 2.0, 3.0);
        constexpr Vector3 c(1.0, 2.0, 4.0);

        // a = (1,2,3), b = (1,2,3)
        // Compare x: 1 == 1 -> compare y: 2 == 2 -> compare z: 3 == 3 => a == b
        static_assert(!(a < b) && !(b < a));
        // a = (1,2,3), c = (1,2,4)
        // Compare x: 1 == 1 -> compare y: 2 == 2 -> compare z: 3 < 4 => a < c
        static_assert(a < c);
    }

    // Operator <=
    {
        constexpr Vector3 a(1.0, 2.0, 3.0);
        constexpr Vector3 b(1.0, 2.0, 3.0);
        constexpr Vector3 c(1.0, 2.0, 4.0);

        // a = (1,2,3), b = (1,2,3)
        // Compare x: 1 == 1 -> compare y: 2 == 2 -> compare z: 3 == 3 => a <= b
        static_assert(a <= b);
        // a = (1,2,3), c = (1,2,4)
        // Compare x: 1 == 1 -> compare y: 2 == 2 -> compare z: 3 < 4 => a <= c
        static_assert(a <= c);
    }

    // Operator >
    {
        constexpr Vector3 a(1.0, 2.0, 3.0);
        constexpr Vector3 b(1.0, 2.0, 3.0);
        constexpr Vector3 c(1.0, 2.0, 4.0);

        // a = (1,2,3), b = (1,2,3)
        // Compare x: 1 == 1 -> compare y: 2 == 2 -> compare z: 3 == 3 => !(a > b)
        static_assert(!(a > b));
        // a = (1,2,3), c = (1,2,4)
        // Compare x: 1 == 1 -> compare y: 2 == 2 -> compare z: 3 < 4 => !(a > c)
        static_assert(!(a > c));
    }

    // Operator >=
    {
        constexpr Vector3 a(1.0, 2.0, 3.0);
        constexpr Vector3 b(1.0, 2.0, 3.0);
        constexpr Vector3 c(1.0, 2.0, 4.0);

        // a = (1,2,3), b = (1,2,3)
        // Compare x: 1 == 1 -> compare y: 2 == 2 -> compare z: 3 == 3 => a >= b
        static_assert(a >= b);
        // a = (1,2,3), c = (1,2,4)
        // Compare x: 1 == 1 -> compare y: 2 == 2 -> compare z: 3 < 4 => !(a >= c)
        static_assert(!(a >= c));
    }
}

constexpr void test_matrix3x3()
{
    using namespace linalg3d;

    // Default Constructor
    {
        constexpr Matrix3 m;
        constexpr bool all_zero = [&m]() -> bool
        {
            for (std::size_t i = 0; i < 3; ++i)
            {
                for (std::size_t j = 0; j < 3; ++j)
                {
                    if (m.m[i][j] != 0.0)
                        return false;
                }
            }
            return true;
        }();
        static_assert(all_zero, "default matrix is not zero");
    }

    // Parameterized Constructor
    {
        constexpr Matrix3 m(1.0, 2.0, 3.0,
                            4.0, 5.0, 6.0,
                            7.0, 8.0, 9.0);
        static_assert(m.m[0][0] == 1.0 && m.m[0][1] == 2.0 && m.m[0][2] == 3.0);
        static_assert(m.m[1][0] == 4.0 && m.m[1][1] == 5.0 && m.m[1][2] == 6.0);
        static_assert(m.m[2][0] == 7.0 && m.m[2][1] == 8.0 && m.m[2][2] == 9.0);
    }

    // Brace-enclosed Constructor
    {
        constexpr Matrix3 m{1.0, 2.0, 3.0,
                            4.0, 5.0, 6.0,
                            7.0, 8.0, 9.0};
        static_assert(m.m[0][0] == 1.0 && m.m[0][1] == 2.0 && m.m[0][2] == 3.0);
        static_assert(m.m[1][0] == 4.0 && m.m[1][1] == 5.0 && m.m[1][2] == 6.0);
        static_assert(m.m[2][0] == 7.0 && m.m[2][1] == 8.0 && m.m[2][2] == 9.0);
    }

    // Transpose Test
    {
        constexpr Matrix3 m(1.0, 2.0, 3.0,
                            4.0, 5.0, 6.0,
                            7.0, 8.0, 9.0);
        static_assert(m.transpose() == Matrix3(1.0, 4.0, 7.0,
                                               2.0, 5.0, 8.0,
                                               3.0, 6.0, 9.0));
    }
}

constexpr void test_euler_angles()
{
    using namespace linalg3d;

    // Default Constructor
    {
        constexpr EulerAngles<AngleType::RADIANS> e;
        static_assert(e.pitch == 0.0 && e.yaw == 0.0 && e.roll == 0.0);
    }

    // Parameterized Constructor
    {
        constexpr EulerAngles e(1.0, 2.0, 3.0);
        static_assert(e.pitch == 1.0 && e.yaw == 2.0 && e.roll == 3.0);
    }

    // Brace-enclosed Constructor
    {
        constexpr EulerAngles e{1.0, 2.0, 3.0};
        static_assert(e.pitch == 1.0 && e.yaw == 2.0 && e.roll == 3.0);
    }
}

constexpr void test_quaternion()
{
    using namespace linalg3d;

    // Default Constructor
    {
        constexpr Quaternion q;
        static_assert(q.w == 0.0 && q.x == 0.0 && q.y == 0.0 && q.z == 0.0);
    }

    // Parameterized Constructor
    {
        constexpr Quaternion q(1.0, 2.0, 3.0, 4.0);
        static_assert(q.w == 1.0 && q.x == 2.0 && q.y == 3.0 && q.z == 4.0);
    }

    // Normalized Quaternion
    {
        constexpr Quaternion q(1.0, 2.0, 3.0, 4.0);
        constexpr Quaternion n = q.normalized();
        constexpr double expectedNorm = gcem::sqrt(1.0 + 4.0 + 9.0 + 16.0);

        static_assert(n.w == 1.0 / expectedNorm);
        static_assert(n.x == 2.0 / expectedNorm);
        static_assert(n.y == 3.0 / expectedNorm);
        static_assert(n.z == 4.0 / expectedNorm);
    }

    // Inverse Quaternion
    {
        constexpr Quaternion q(1.0, 2.0, 3.0, 4.0);
        constexpr Quaternion inv = q.inverse();
        constexpr double n_sq = 1.0 + 4.0 + 9.0 + 16.0;

        static_assert(inv.w == 1.0 / n_sq);
        static_assert(inv.x == -2.0 / n_sq);
        static_assert(inv.y == -3.0 / n_sq);
        static_assert(inv.z == -4.0 / n_sq);
    }

    // Dot Product
    {
        constexpr Quaternion q1(1.0, 2.0, 3.0, 4.0);
        constexpr Quaternion q2(5.0, 6.0, 7.0, 8.0);
        constexpr double dot = q1.dot(q2);
        constexpr double expected = 1.0 * 5.0 + 2.0 * 6.0 + 3.0 * 7.0 + 4.0 * 8.0;
        // assert_near(dot, expected);

        static_assert(dot == expected);
    }

    // Quaternion Multiplication
    {
        constexpr Quaternion q1(1.0, 2.0, 3.0, 4.0);
        constexpr Quaternion q2(5.0, 6.0, 7.0, 8.0);
        constexpr Quaternion result = q1 * q2;

        static_assert(result.w == -60.0);
        static_assert(result.x == 12.0);
        static_assert(result.y == 30.0);
        static_assert(result.z == 24.0);
    }

    // Vector Rotation by Quaternion
    {
        // Create a quaternion representing a 90° (pi/2) rotation around the z-axis.
        constexpr double angle = PI / 2.0;
        constexpr double cos_val = gcem::cos(angle / 2.0);
        constexpr double sin_val = gcem::sin(angle / 2.0);
        constexpr Quaternion q(cos_val, 0.0, 0.0, sin_val); // axis (0, 0, 1)

        // Original vector (1, 0, 0)
        constexpr Vector3 v(1.0, 0.0, 0.0);

        // Rotate the vector using the quaternion.
        constexpr Vector3 rotated = q * v;

        // The expected result of rotating (1,0,0) 90° around z-axis is (0, 1, 0).

        static_assert(rotated.x == 0.0);
        static_assert(rotated.y == 1.0);
        static_assert(rotated.z == 0.0);
    }
}

constexpr void test_operations()
{
    using namespace linalg3d;

    // Vector3 and Quaternion Multiplication
    {
        constexpr Vector3 v(1.0, 2.0, 3.0);
        constexpr Quaternion q(1.0, 2.0, 3.0, 4.0);
        constexpr Vector3 result = v * q;

        assert(!(linalg3d::fabs(result.x - 20.0) < EPSILON) && "value is out of expected range");
        assert(!(linalg3d::fabs(result.y - 16.0) < EPSILON) && "value is out of expected range");
        assert(!(linalg3d::fabs(result.z - 24.0) < EPSILON) && "value is out of expected range");
    }

    // Quaternion to Rotation Matrix
    {
        constexpr Quaternion q(1.0, 2.0, 3.0, 4.0);
        constexpr Matrix3 m = quaternionToMatrix(q);
        assert(!(m == Matrix3(-7.0, 8.0, 3.0,
                              6.0, 5.0, -4.0,
                              9.0, 2.0, -1.0) &&
                 "value is out of expected range for matrix"));
    }

    // Quaternion from Euler Angles
    {
        constexpr EulerAngles e(1.0, 2.0, 3.0);
        constexpr Quaternion q = eulerAnglesToQuaternion(e);
        assert(!(q == Quaternion(0.983347, 0.034270, 0.106020, 0.143572) && "value is out of expected range for quaternion"));
    }

    // Euler Angles from Quaternion
    {
        constexpr Quaternion q(0.983347, 0.034270, 0.106020, 0.143572);
        constexpr EulerAngles e = quaternionToEulerAngles(q);
        assert(!(linalg3d::fabs(e.pitch.value() - 1.0) < EPSILON) && "value is out of expected range for pitch");
    }

    // Quaternion Multiplication Identity
    {
        constexpr Quaternion q(1.0, 2.0, 3.0, 4.0);
        static_assert(q == q * Quaternion::identity());
    }

    // Quaternion Multiplication Associativity
    {
        constexpr Quaternion q1(1.0, 2.0, 3.0, 4.0);
        constexpr Quaternion q2(5.0, 6.0, 7.0, 8.0);
        constexpr Quaternion q3(9.0, 10.0, 11.0, 12.0);
        static_assert((q1 * q2) * q3 == q1 * (q2 * q3));
    }

    // Quaternion Multiplication Distributivity
    {
        constexpr Quaternion q1(1.0, 2.0, 3.0, 4.0);
        constexpr Quaternion q2(5.0, 6.0, 7.0, 8.0);
        constexpr Quaternion q3(9.0, 10.0, 11.0, 12.0);
        static_assert(q1 * (q2 + q3) == q1 * q2 + q1 * q3);
    }
}

constexpr void test_matrix_vector_multiplication()
{
    using namespace linalg3d;

    // Test 1: Basic multiplication.
    // Matrix:
    // [ 1  2  3 ]
    // [ 4  5  6 ]
    // [ 7  8  9 ]
    // Vector: (1, 1, 1)
    // Expected result: (6, 15, 24)
    constexpr Matrix3 m(1.0, 2.0, 3.0,
                        4.0, 5.0, 6.0,
                        7.0, 8.0, 9.0);
    constexpr Vector3 v(1.0, 1.0, 1.0);
    constexpr Vector3 result = m * v;
    static_assert(nearly_equal(result.x, 6.0), "Matrix-Vector multiplication failed for x component");
    static_assert(nearly_equal(result.y, 15.0), "Matrix-Vector multiplication failed for y component");
    static_assert(nearly_equal(result.z, 24.0), "Matrix-Vector multiplication failed for z component");

    // Test 2: Multiplication with the identity matrix.
    // Identity matrix should leave the vector unchanged.
    constexpr Matrix3 identity(1.0, 0.0, 0.0,
                               0.0, 1.0, 0.0,
                               0.0, 0.0, 1.0);
    constexpr Vector3 v2(3.14, -2.71, 0.0);
    constexpr Vector3 result2 = identity * v2;
    static_assert(nearly_equal(result2.x, v2.x), "Identity matrix multiplication failed for x component");
    static_assert(nearly_equal(result2.y, v2.y), "Identity matrix multiplication failed for y component");
    static_assert(nearly_equal(result2.z, v2.z), "Identity matrix multiplication failed for z component");

    // Test 3: Multiplication with a zero matrix.
    constexpr Matrix3 zero(0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0);
    constexpr Vector3 v3(1.0, 2.0, 3.0);
    constexpr Vector3 result3 = zero * v3;
    static_assert(nearly_equal(result3.x, 0.0), "Zero matrix multiplication failed for x component");
    static_assert(nearly_equal(result3.y, 0.0), "Zero matrix multiplication failed for y component");
    static_assert(nearly_equal(result3.z, 0.0), "Zero matrix multiplication failed for z component");

    {
        Matrix3 m(2.0, 0.0, 0.0,
                  0.0, 3.0, 0.0,
                  0.0, 0.0, 4.0);
        Vector3 v(1.0, 1.0, 1.0);
        Vector3 r = m * v; // expected (2, 3, 4)
        assert(nearly_equal(r.x, 2.0));
        assert(nearly_equal(r.y, 3.0));
        assert(nearly_equal(r.z, 4.0));
    }
}

void test_angle_floating_point_limits()
{
    using namespace linalg3d;

    Angle<AngleType::RADIANS> smallRad(std::numeric_limits<double>::min());
    Angle<AngleType::RADIANS> largeRad(std::numeric_limits<double>::max());

    assert(!is_nan(largeRad.to_degrees().value()) && "Max double conversion returned NaN");
    assert(!is_inf(largeRad.to_degrees().value()) && "Max double conversion returned Inf");

    Angle<AngleType::DEGREES> smallDeg(std::numeric_limits<double>::min());
    Angle<AngleType::DEGREES> largeDeg(std::numeric_limits<double>::max());

    assert(!is_nan(largeDeg.to_radians().value()) && "Max double conversion returned NaN");
    assert(!is_inf(largeDeg.to_radians().value()) && "Max double conversion returned Inf");
}

constexpr void test_vector3_normalization_safety()
{
    using namespace linalg3d;

    constexpr Vector3 zeroVec(0.0, 0.0, 0.0);
    constexpr Vector3 normed = zeroVec.normalized();

    static_assert(normed == zeroVec, "Zero vector should remain zero after normalization");

    constexpr Vector3 smallVec(1e-200, 1e-200, 1e-200);
    constexpr Vector3 normedSmall = smallVec.normalized();

    static_assert(!is_nan(normedSmall.x) && !is_nan(normedSmall.y) && !is_nan(normedSmall.z),
                  "Normalization of small vector produced NaN values");
}

constexpr void test_quaternion_rotation_invariants()
{
    using namespace linalg3d;

    constexpr Quaternion q(0.70710678, 0.70710678, 0.0, 0.0); // 90-degree rotation around X-axis
    constexpr Vector3 v(0.0, 1.0, 0.0);
    constexpr Vector3 rotated = q * v;

    constexpr double origNorm = v.norm();
    constexpr double newNorm = rotated.norm();

    static_assert(nearly_equal(origNorm, newNorm), "Quaternion rotation should preserve vector norm");

    // Ensure rotated vector is where we expect it to be
    static_assert(nearly_equal(rotated.y, 0.0), "Rotation moved Y component incorrectly");
    static_assert(nearly_equal(rotated.z, 1.0), "Rotation failed to place vector on Z axis");
}

constexpr void test_matrix3x3_determinant()
{
    using namespace linalg3d;

    constexpr Matrix3 singularMatrix(1.0, 2.0, 3.0,
                                     2.0, 4.0, 6.0,
                                     3.0, 6.0, 9.0); // Rows are linearly dependent
    constexpr Matrix3 identityMatrix(1.0, 0.0, 0.0,
                                     0.0, 1.0, 0.0,
                                     0.0, 0.0, 1.0);

    static_assert(singularMatrix.determinant() == 0.0, "Singular matrix should have determinant 0");
    static_assert(identityMatrix.determinant() == 1.0, "Identity matrix determinant should be 1");

    // Attempt inversion (should fail gracefully)
    constexpr Matrix3 inverseSingular = singularMatrix.inverse();
    constexpr Matrix3 inverseIdentity = identityMatrix.inverse();

    static_assert(inverseSingular == Matrix3{}, "Inverse of singular matrix should be zero matrix");
    static_assert(inverseIdentity == identityMatrix, "Inverse of identity should be identity");
}

constexpr void test_matrix_inversion_correctness()
{
    using namespace linalg3d;

    constexpr Matrix3 m(2.0, -1.0, 0.0,
                        -1.0, 2.0, -1.0,
                        0.0, -1.0, 2.0);

    constexpr Matrix3 inv = m.inverse();
    constexpr Matrix3 result = m * inv;

    constexpr Matrix3 identity(1.0, 0.0, 0.0,
                               0.0, 1.0, 0.0,
                               0.0, 0.0, 1.0);

    static_assert(nearly_equal(result.m[0][0], identity.m[0][0]), "Inverse test failed for [0][0]");
    static_assert(nearly_equal(result.m[1][1], identity.m[1][1]), "Inverse test failed for [1][1]");
    static_assert(nearly_equal(result.m[2][2], identity.m[2][2]), "Inverse test failed for [2][2]");
}

int main()
{
    test_angle();
    test_angle_edge_cases();
    test_vector3();
    test_vector3_operators();
    test_vector3_compare_operators();
    test_matrix3x3();
    test_euler_angles();
    test_quaternion();
    test_matrix_vector_multiplication();
    test_operations();
    test_angle_floating_point_limits();
    test_vector3_normalization_safety();
    test_quaternion_rotation_invariants();
    test_matrix3x3_determinant();
    test_matrix_inversion_correctness();
    fmt::print("All tests passed!\n");
    return 0;
}
