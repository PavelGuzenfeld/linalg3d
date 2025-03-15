#include "linalg3d/linalg.hpp"
#include <cassert>    // For assert
#include <cstdlib>    // For std::exit
#include <fmt/core.h> // For formatting

// Floating point precision tolerance
constexpr double EPSILON = 1e-5;

inline void assert_near(double actual, double expected, std::string message = "", double tolerance = EPSILON)
{
    if (std::fabs(actual - expected) >= tolerance)
    {
        fmt::print("❌ Assertion failed: {} (actual: {}, expected: {})\n", message, actual, expected);
        std::exit(1);
    }
}

void test_angle()
{
    using namespace linalg3d;
    fmt::print("Running Angle tests...\n");

    // Radians to Degrees conversion
    {
        Angle<AngleType::RADIANS> radAngle(M_PI);
        auto degAngle = radAngle.to_degrees();
        assert_near(degAngle.value(), 180.0);
        fmt::print("✅ Radians to degrees: {} rad -> {}°\n", radAngle.value(), degAngle.value());
    }

    // Degrees to Radians conversion
    {
        Angle<AngleType::DEGREES> degAngle(180.0);
        auto radAngle = degAngle.to_radians();
        assert_near(radAngle.value(), M_PI);
        fmt::print("✅ Degrees to radians: {}° -> {} rad\n", degAngle.value(), radAngle.value());
    }

    // Identity conversion: Radians -> Degrees -> Radians
    {
        Angle<AngleType::RADIANS> radOrig(M_PI / 4.0);
        auto deg = radOrig.to_degrees();
        auto radAgain = deg.to_radians();
        assert_near(radOrig.value(), radAgain.value());
        fmt::print("✅ Identity conversion (rad->deg->rad): {} -> {} -> {}\n", radOrig.value(), deg.value(), radAgain.value());
    }

    // Test trigonometric functions at specific angles (radians)
    {
        Angle<AngleType::RADIANS> zero(0.0);
        assert_near(zero.sin(), 0.0);
        assert_near(zero.cos(), 1.0);
        assert_near(zero.tan(), 0.0);
        fmt::print("✅ Trigonometric functions at 0 radians\n");

        Angle<AngleType::RADIANS> halfPi(M_PI / 2.0);
        assert_near(halfPi.sin(), 1.0);
        // cos(π/2) should be 0; note that due to FP rounding it might be very small.
        assert_near(halfPi.cos(), 0.0);
        fmt::print("✅ Trigonometric functions at π/2 radians (sin, cos)\n");
    }

    // Test negative angle conversion
    {
        Angle<AngleType::DEGREES> negDeg(-45.0);
        auto negRad = negDeg.to_radians();
        assert_near(negRad.value(), -M_PI / 4.0);
        fmt::print("✅ Negative angle conversion: {}° -> {} rad\n", negDeg.value(), negRad.value());
    }

    // Test periodicity/wrap-around: 360° -> 2π rad
    {
        Angle<AngleType::DEGREES> fullCircle(360.0);
        auto radFull = fullCircle.to_radians();
        assert_near(radFull.value(), 2 * M_PI);
        fmt::print("✅ 360° conversion: {}° -> {} rad\n", fullCircle.value(), radFull.value());
    }

    // Test operator overloads: addition, subtraction, multiplication, and division.
    {
        Angle<AngleType::RADIANS> a(M_PI / 3.0); // 60°
        Angle<AngleType::RADIANS> b(M_PI / 6.0); // 30°
        auto sum = a + b;                        // Expect 90° or π/2 rad.
        auto diff = a - b;                       // Expect 30° or π/6 rad.
        assert_near(sum.value(), M_PI / 2.0);
        assert_near(diff.value(), M_PI / 6.0);
        fmt::print("✅ Operator overloads (addition/subtraction): {} + {} = {}, {} - {} = {}\n",
                   a.value(), b.value(), sum.value(), a.value(), b.value(), diff.value());

        // Multiplication and division operators.
        double scalar = 2.0;
        double prod = a * scalar;
        double div = a / scalar;
        assert_near(prod, (M_PI / 3.0) * 2.0);
        assert_near(div, (M_PI / 3.0) / 2.0);
        fmt::print("✅ Operator overloads (multiplication/division): {} * {} = {}, {} / {} = {}\n",
                   a.value(), scalar, prod, a.value(), scalar, div);
    }

    // Test equality and comparisons.
    {
        Angle<AngleType::RADIANS> a(M_PI / 2.0);
        Angle<AngleType::RADIANS> b(M_PI / 2.0);
        Angle<AngleType::RADIANS> c(M_PI / 3.0);
        assert(a == M_PI / 2.0);
        assert(a == b.value());
        assert(a != c.value());
        assert(c < a.value());
        fmt::print("✅ Equality and comparison operators\n");
    }

    // Test static conversion helpers
    {
        auto fromRad = Angle<AngleType::RADIANS>::from_radians(M_PI / 2.0);
        auto fromDeg = Angle<AngleType::DEGREES>::from_degrees(90.0);
        assert_near(fromRad.value(), M_PI / 2.0);
        assert_near(fromDeg.value(), 90.0);
        fmt::print("✅ Static conversion helpers: from_radians, from_degrees\n");
    }
}

void test_vector3()
{
    using namespace linalg3d;

    fmt::print("Running Vector3 tests...\n");

    // Default Constructor
    {
        Vector3 v;
        assert(v.x == 0.0 && v.y == 0.0 && v.z == 0.0);
        fmt::print("✅ Default constructor\n");
    }

    // Parameterized Constructor
    {
        Vector3 v(1.5, -2.5, 3.0);
        assert(v.x == 1.5 && v.y == -2.5 && v.z == 3.0);
        fmt::print("✅ Parameterized constructor\n");
    }

    // Norm Test
    {
        Vector3 v(3.0, 4.0, 12.0);
        double expectedNorm = 13.0;
        assert_near(v.norm(), expectedNorm);
        fmt::print("✅ Norm calculation\n");
    }

    // Zero Vector Norm
    {
        Vector3 v(0.0, 0.0, 0.0);
        assert(v.norm() == 0.0);
        fmt::print("✅ Zero vector norm\n");
    }

    // Large Value Norm
    {
        double large = 1e10;
        Vector3 v(large, large, large);
        double expected = std::sqrt(3.0) * large;
        assert_near(v.norm(), expected);
        fmt::print("✅ Large value norm\n");
    }

    // ---------- New Tests Below ----------

    // norm_sq() Test
    {
        Vector3 v(3.0, 4.0, 12.0);
        // 3^2 + 4^2 + 12^2 = 9 + 16 + 144 = 169
        double expectedNormSq = 169.0;
        assert_near(v.norm_sq(), expectedNormSq);
        fmt::print("✅ norm_sq calculation\n");
    }

    // Normalized Vector
    {
        Vector3 v(3.0, 4.0, 0.0);
        Vector3 n = v.normalized();
        assert_near(n.x, 3.0 / 5.0);
        assert_near(n.y, 4.0 / 5.0);
        assert_near(n.z, 0.0);
        assert_near(n.norm(), 1.0);
        fmt::print("✅ Normalized vector\n");
    }

    // Normalizing the Zero Vector (should return zero vector)
    {
        Vector3 v(0.0, 0.0, 0.0);
        Vector3 n = v.normalized();
        assert(n.x == 0.0 && n.y == 0.0 && n.z == 0.0);
        fmt::print("✅ Normalized zero vector\n");
    }

    // Dot Product Orthogonal
    {
        Vector3 v1(1.0, 0.0, 0.0);
        Vector3 v2(0.0, 5.0, 0.0);
        assert(v1.dot(v2) == 0.0);
        fmt::print("✅ Dot product of orthogonal vectors\n");
    }

    // Dot Product Parallel
    {
        Vector3 v1(2.0, 2.0, 1.0);
        Vector3 v2 = v1 * 3.0;
        // 2*6 + 2*6 + 1*3 = 27
        assert(v1.dot(v2) == 27.0);
        fmt::print("✅ Dot product of parallel vectors\n");
    }

    // Cross Product Parallel Vectors (should return zero vector)
    {
        Vector3 v1(1.0, 2.0, 3.0);
        Vector3 v2 = v1 * 2.0;
        Vector3 cross = v1.cross(v2);
        assert(cross.x == 0.0 && cross.y == 0.0 && cross.z == 0.0);
        fmt::print("✅ Cross product of parallel vectors\n");
    }

    // Cross Product Orthogonal Vectors
    {
        Vector3 i(1.0, 0.0, 0.0);
        Vector3 j(0.0, 1.0, 0.0);
        Vector3 k = i.cross(j);
        // (1,0,0) x (0,1,0) = (0,0,1)
        assert(k.x == 0.0 && k.y == 0.0 && k.z == 1.0);
        fmt::print("✅ Cross product of orthogonal vectors\n");
    }
}

void test_vector3_operators()
{
    using namespace linalg3d;

    // Addition Operator
    {
        Vector3 v1(1.0, 2.0, 3.0);
        Vector3 v2(4.0, 5.0, 6.0);
        Vector3 result = v1 + v2;
        // (1+4, 2+5, 3+6) = (5, 7, 9)
        assert(result.x == 5.0 && result.y == 7.0 && result.z == 9.0);
        fmt::print("✅ Vector addition operator\n");
    }

    // Subtraction Operator
    {
        Vector3 v1(5.0, 7.0, 9.0);
        Vector3 v2(1.0, 2.0, 3.0);
        Vector3 result = v1 - v2;
        // (5-1, 7-2, 9-3) = (4, 5, 6)
        assert(result.x == 4.0 && result.y == 5.0 && result.z == 6.0);
        fmt::print("✅ Vector subtraction operator\n");
    }

    // Unary Minus Operator
    {
        Vector3 v(1.0, -2.0, 3.0);
        Vector3 neg = -v;
        // -(1,-2,3) = (-1,2,-3)
        assert(neg.x == -1.0 && neg.y == 2.0 && neg.z == -3.0);
        fmt::print("✅ Unary minus operator\n");
    }

    // Scalar Multiplication
    {
        Vector3 v(1.0, -2.0, 3.0);
        Vector3 result = v * -1.5;
        // (1 * -1.5, -2 * -1.5, 3 * -1.5) = (-1.5, 3.0, -4.5)
        assert_near(result.x, -1.5);
        assert_near(result.y, 3.0);
        assert_near(result.z, -4.5);
        fmt::print("✅ Scalar multiplication operator\n");
    }

    // Division Operator
    {
        Vector3 v(2.0, -4.0, 6.0);
        double scalar = 2.0;
        Vector3 result = v / scalar;
        // (2/2, -4/2, 6/2) = (1, -2, 3)
        assert_near(result.x, 1.0);
        assert_near(result.y, -2.0);
        assert_near(result.z, 3.0);
        fmt::print("✅ Vector division operator\n");

        // Optional check: dividing by zero would cause a float exception,
        // so typically you ensure scalar != 0 before dividing.
        // e.g. Vector3 invalidResult = v / 0.0; // not recommended
    }
}

void test_vector3_compare_operators()
{
    using namespace linalg3d;
    // Three-way Comparison Operator
    {
        Vector3 a(1.0, 2.0, 3.0);
        Vector3 b(1.0, 2.0, 3.0);
        Vector3 c(1.0, 2.0, 4.0);

        // Equality check
        assert((a <=> b) == 0);
        // Not equal
        assert((a <=> c) != 0);

        // Lexicographical ordering check (operator<=> default):
        // a = (1,2,3), c = (1,2,4)
        // Compare x: 1 == 1 -> compare y: 2 == 2 -> compare z: 3 < 4 => a < c
        assert((a <=> c) < 0); // means a < c
        fmt::print("✅ Three-way comparison operator\n");
    }

    // Lexicographical Ordering
    {
        Vector3 a(1.0, 2.0, 3.0);
        Vector3 b(1.0, 2.0, 3.0);
        Vector3 c(1.0, 2.0, 4.0);

        // a = (1,2,3), b = (1,2,3)
        // Compare x: 1 == 1 -> compare y: 2 == 2 -> compare z: 3 == 3 => a == b
        assert(!(a < b) && !(b < a));
        // a = (1,2,3), c = (1,2,4)
        // Compare x: 1 == 1 -> compare y: 2 == 2 -> compare z: 3 < 4 => a < c
        assert(a < c);
        // a = (1,2,3), b = (1,2,3)
        // Compare x: 1 == 1 -> compare y: 2 == 2 -> compare z: 3 == 3 => a <= b
        assert(!(a < b) && (a <= b));
        // a = (1,2,3), c = (1,2,4)
        // Compare x: 1 == 1 -> compare y: 2 == 2 -> compare z: 3 < 4 => a <= c
        assert(a <= c);
        fmt::print("✅ Lexicographical ordering\n");
    }

    // Operator == and !=
    {
        Vector3 a(1.0, 2.0, 3.0);
        Vector3 b(1.0, 2.0, 3.0);
        Vector3 c(1.0, 2.0, 4.0);

        // Equality check
        assert(a == b);
        assert(a != c);
        fmt::print("✅ Equality and inequality operators\n");
    }

    // Operator <
    {
        Vector3 a(1.0, 2.0, 3.0);
        Vector3 b(1.0, 2.0, 3.0);
        Vector3 c(1.0, 2.0, 4.0);

        // a = (1,2,3), b = (1,2,3)
        // Compare x: 1 == 1 -> compare y: 2 == 2 -> compare z: 3 == 3 => a == b
        assert(!(a < b) && !(b < a));
        // a = (1,2,3), c = (1,2,4)
        // Compare x: 1 == 1 -> compare y: 2 == 2 -> compare z: 3 < 4 => a < c
        assert(a < c);
        fmt::print("✅ Less than operator\n");
    }

    // Operator <=
    {
        Vector3 a(1.0, 2.0, 3.0);
        Vector3 b(1.0, 2.0, 3.0);
        Vector3 c(1.0, 2.0, 4.0);

        // a = (1,2,3), b = (1,2,3)
        // Compare x: 1 == 1 -> compare y: 2 == 2 -> compare z: 3 == 3 => a <= b
        assert(a <= b);
        // a = (1,2,3), c = (1,2,4)
        // Compare x: 1 == 1 -> compare y: 2 == 2 -> compare z: 3 < 4 => a <= c
        assert(a <= c);
        fmt::print("✅ Less than or equal operator\n");
    }

    // Operator >
    {
        Vector3 a(1.0, 2.0, 3.0);
        Vector3 b(1.0, 2.0, 3.0);
        Vector3 c(1.0, 2.0, 4.0);

        // a = (1,2,3), b = (1,2,3)
        // Compare x: 1 == 1 -> compare y: 2 == 2 -> compare z: 3 == 3 => !(a > b)
        assert(!(a > b));
        // a = (1,2,3), c = (1,2,4)
        // Compare x: 1 == 1 -> compare y: 2 == 2 -> compare z: 3 < 4 => !(a > c)
        assert(!(a > c));
        fmt::print("✅ Greater than operator\n");
    }

    // Operator >=
    {
        Vector3 a(1.0, 2.0, 3.0);
        Vector3 b(1.0, 2.0, 3.0);
        Vector3 c(1.0, 2.0, 4.0);

        // a = (1,2,3), b = (1,2,3)
        // Compare x: 1 == 1 -> compare y: 2 == 2 -> compare z: 3 == 3 => a >= b
        assert(a >= b);
        // a = (1,2,3), c = (1,2,4)
        // Compare x: 1 == 1 -> compare y: 2 == 2 -> compare z: 3 < 4 => !(a >= c)
        assert(!(a >= c));
        fmt::print("✅ Greater than or equal operator\n");
    }
}

void test_matrix3x3()
{
    using namespace linalg3d;

    fmt::print("Running Matrix3x3 tests...\n");

    // Default Constructor
    {
        Matrix3x3 m;
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                assert(m.m[i][j] == 0.0);
            }
        }
        fmt::print("✅ Default constructor\n");
    }

    // Parameterized Constructor
    {
        Matrix3x3 m(1.0, 2.0, 3.0,
                    4.0, 5.0, 6.0,
                    7.0, 8.0, 9.0);
        assert(m.m[0][0] == 1.0 && m.m[0][1] == 2.0 && m.m[0][2] == 3.0);
        assert(m.m[1][0] == 4.0 && m.m[1][1] == 5.0 && m.m[1][2] == 6.0);
        assert(m.m[2][0] == 7.0 && m.m[2][1] == 8.0 && m.m[2][2] == 9.0);
        fmt::print("✅ Parameterized constructor\n");
    }

    // Brace-enclosed Constructor
    {
        Matrix3x3 m{1.0, 2.0, 3.0,
                    4.0, 5.0, 6.0,
                    7.0, 8.0, 9.0};
        assert(m.m[0][0] == 1.0 && m.m[0][1] == 2.0 && m.m[0][2] == 3.0);
        assert(m.m[1][0] == 4.0 && m.m[1][1] == 5.0 && m.m[1][2] == 6.0);
        assert(m.m[2][0] == 7.0 && m.m[2][1] == 8.0 && m.m[2][2] == 9.0);
        fmt::print("✅ Brace-enclosed constructor\n");
    }

    // Transpose Test
    {
        Matrix3x3 m(1.0, 2.0, 3.0,
                    4.0, 5.0, 6.0,
                    7.0, 8.0, 9.0);
        Matrix3x3 t = m.transpose();
        assert(t.m[0][0] == 1.0 && t.m[0][1] == 4.0 && t.m[0][2] == 7.0);
        assert(t.m[1][0] == 2.0 && t.m[1][1] == 5.0 && t.m[1][2] == 8.0);
        assert(t.m[2][0] == 3.0 && t.m[2][1] == 6.0 && t.m[2][2] == 9.0);
        fmt::print("✅ Transpose calculation\n");
    }
}

void test_euler_angles()
{
    using namespace linalg3d;

    fmt::print("Running Euler Angles tests...\n");

    // Default Constructor
    {
        EulerAngles<AngleType::RADIANS> e;
        assert(e.pitch == 0.0 && e.yaw == 0.0 && e.roll == 0.0);
        fmt::print("✅ Default constructor\n");
    }

    // Parameterized Constructor
    {
        EulerAngles e(1.0, 2.0, 3.0);
        assert(e.pitch == 1.0 && e.yaw == 2.0 && e.roll == 3.0);
        fmt::print("✅ Parameterized constructor\n");
    }

    // Brace-enclosed Constructor
    {
        EulerAngles e{1.0, 2.0, 3.0};
        assert(e.pitch == 1.0 && e.yaw == 2.0 && e.roll == 3.0);
        fmt::print("✅ Brace-enclosed constructor\n");
    }
}

void test_quaternion()
{
    using namespace linalg3d;

    fmt::print("Running Quaternion tests...\n");

    // Default Constructor
    {
        Quaternion q;
        assert(q.w == 0.0 && q.x == 0.0 && q.y == 0.0 && q.z == 0.0);
        fmt::print("✅ Default constructor\n");
    }

    // Parameterized Constructor
    {
        Quaternion q(1.0, 2.0, 3.0, 4.0);
        assert(q.w == 1.0 && q.x == 2.0 && q.y == 3.0 && q.z == 4.0);
        fmt::print("✅ Parameterized constructor\n");
    }

    // Normalized Quaternion
    {
        Quaternion q(1.0, 2.0, 3.0, 4.0);
        Quaternion n = q.normalized();
        double expectedNorm = std::sqrt(1.0 + 4.0 + 9.0 + 16.0);
        assert_near(n.w, 1.0 / expectedNorm);
        assert_near(n.x, 2.0 / expectedNorm);
        assert_near(n.y, 3.0 / expectedNorm);
        assert_near(n.z, 4.0 / expectedNorm);
        fmt::print("✅ Normalized quaternion\n");
    }

    // Inverse Quaternion
    {
        Quaternion q(1.0, 2.0, 3.0, 4.0);
        Quaternion inv = q.inverse();
        double n_sq = 1.0 + 4.0 + 9.0 + 16.0;
        assert_near(inv.w, 1.0 / n_sq);
        assert_near(inv.x, -2.0 / n_sq);
        assert_near(inv.y, -3.0 / n_sq);
        assert_near(inv.z, -4.0 / n_sq);
        fmt::print("✅ Inverse quaternion\n");
    }

    // Dot Product
    {
        Quaternion q1(1.0, 2.0, 3.0, 4.0);
        Quaternion q2(5.0, 6.0, 7.0, 8.0);
        double dot = q1.dot(q2);
        double expected = 1.0 * 5.0 + 2.0 * 6.0 + 3.0 * 7.0 + 4.0 * 8.0;
        assert_near(dot, expected);
        fmt::print("✅ Dot product\n");
    }

    // Quaternion Multiplication
    {
        Quaternion q1(1.0, 2.0, 3.0, 4.0);
        Quaternion q2(5.0, 6.0, 7.0, 8.0);
        Quaternion result = q1 * q2;
        assert_near(result.w, -60.0);
        assert_near(result.x, 12.0);
        assert_near(result.y, 30.0);
        assert_near(result.z, 24.0);
        fmt::print("✅ Quaternion multiplication\n");
    }

    // Vector Rotation by Quaternion
    {
        // Create a quaternion representing a 90° (pi/2) rotation around the z-axis.
        double angle = M_PI / 2.0;
        double cos_val = std::cos(angle / 2.0);
        double sin_val = std::sin(angle / 2.0);
        Quaternion q(cos_val, 0.0, 0.0, sin_val); // axis (0, 0, 1)

        // Original vector (1, 0, 0)
        Vector3 v(1.0, 0.0, 0.0);

        // Rotate the vector using the quaternion.
        Vector3 rotated = q * v;

        // The expected result of rotating (1,0,0) 90° around z-axis is (0, 1, 0).
        assert_near(rotated.x, 0.0);
        assert_near(rotated.y, 1.0);
        assert_near(rotated.z, 0.0);
        fmt::print("✅ Vector rotation by quaternion\n");
    }
}

void test_operations()
{
    using namespace linalg3d;

    // Vector3 and Quaternion Multiplication
    {
        Vector3 v(1.0, 2.0, 3.0);
        Quaternion q(1.0, 2.0, 3.0, 4.0);
        Vector3 result = v * q;
        assert_near(result.x, 20.0);
        assert_near(result.y, 16.0);
        assert_near(result.z, 24.0);
        fmt::print("✅ Vector3 and Quaternion multiplication\n");
    }

    // Quaternion to Rotation Matrix
    {
        Quaternion q(1.0, 2.0, 3.0, 4.0);
        Matrix3x3 m = toRotationMatrix(q);
        assert_near(m.m[0][0], -7.0);
        assert_near(m.m[0][1], 8.0);
        assert_near(m.m[0][2], 3.0);
        assert_near(m.m[1][0], 6.0);
        assert_near(m.m[1][1], 5.0);
        assert_near(m.m[1][2], -4.0);
        assert_near(m.m[2][0], 9.0);
        assert_near(m.m[2][1], 2.0);
        assert_near(m.m[2][2], -1.0);
        fmt::print("✅ Quaternion to rotation matrix\n");
    }

    // Quaternion from Euler Angles
    {
        EulerAngles e(1.0, 2.0, 3.0);
        Quaternion q = fromEulerAngles(e);
        assert_near(q.w, 0.983347);
        assert_near(q.x, 0.034270);
        assert_near(q.y, 0.106020);
        assert_near(q.z, 0.143572);
        fmt::print("✅ Quaternion from Euler angles\n");
    }

    // Euler Angles from Quaternion
    {
        Quaternion q(0.983347, 0.034270, 0.106020, 0.143572);
        EulerAngles e = fromQuaternion(q);
        assert_near(e.pitch.value(), 1.0);
        assert_near(e.yaw.value(), 2.0);
        assert_near(e.roll.value(), 3.0);
        fmt::print("✅ Euler angles from quaternion\n");
    }

    // Quaternion Multiplication Identity
    {
        Quaternion q(1.0, 2.0, 3.0, 4.0);
        Quaternion identity(1.0, 0.0, 0.0, 0.0);
        Quaternion result = q * identity;
        assert(result.w == q.w && result.x == q.x && result.y == q.y && result.z == q.z);
        fmt::print("✅ Quaternion multiplication identity\n");
    }

    // Quaternion Multiplication Associativity
    {
        Quaternion q1(1.0, 2.0, 3.0, 4.0);
        Quaternion q2(5.0, 6.0, 7.0, 8.0);
        Quaternion q3(9.0, 10.0, 11.0, 12.0);
        Quaternion result1 = (q1 * q2) * q3;
        Quaternion result2 = q1 * (q2 * q3);
        assert(result1.w == result2.w && result1.x == result2.x && result1.y == result2.y && result1.z == result2.z);
        fmt::print("✅ Quaternion multiplication associativity\n");
    }

    // Quaternion Multiplication Distributivity
    {
        Quaternion q1(1.0, 2.0, 3.0, 4.0);
        Quaternion q2(5.0, 6.0, 7.0, 8.0);
        Quaternion q3(9.0, 10.0, 11.0, 12.0);
        Quaternion result1 = q1 * (q2 + q3);
        Quaternion result2 = q1 * q2 + q1 * q3;
        assert(result1.w == result2.w && result1.x == result2.x && result1.y == result2.y && result1.z == result2.z);
        fmt::print("✅ Quaternion multiplication distributivity\n");
    }
}

int main()
{
    test_angle();
    test_vector3();
    test_vector3_operators();
    test_vector3_compare_operators();
    test_matrix3x3();
    test_euler_angles();
    test_quaternion();
    fmt::print("🎉 All Vector3 tests passed successfully!\n");
    return 0;
}