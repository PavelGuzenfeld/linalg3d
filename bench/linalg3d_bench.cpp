#define ANKERL_NANOBENCH_IMPLEMENT
#include "linalg3d/linalg.hpp"
#include <nanobench.h>

using namespace linalg3d;

int main()
{
    ankerl::nanobench::Bench bench;
    bench.title("linalg3d").warmup(100).minEpochIterations(1000);

    // =========================================================================
    // Vector2
    // =========================================================================

    const Vector2 v2a(3.0, 4.0);
    const Vector2 v2b(1.0, 2.0);

    bench.run("Vector2::norm", [&] {
        ankerl::nanobench::doNotOptimizeAway(v2a.norm());
    });

    bench.run("Vector2::normalized", [&] {
        ankerl::nanobench::doNotOptimizeAway(v2a.normalized());
    });

    bench.run("Vector2::dot", [&] {
        ankerl::nanobench::doNotOptimizeAway(v2a.dot(v2b));
    });

    bench.run("Vector2::cross", [&] {
        ankerl::nanobench::doNotOptimizeAway(v2a.cross(v2b));
    });

    bench.run("Vector2::operator+", [&] {
        ankerl::nanobench::doNotOptimizeAway(v2a + v2b);
    });

    // =========================================================================
    // Vector3
    // =========================================================================

    const Vector3 v3a(1.0, 2.0, 3.0);
    const Vector3 v3b(4.0, 5.0, 6.0);

    bench.run("Vector3::norm", [&] {
        ankerl::nanobench::doNotOptimizeAway(v3a.norm());
    });

    bench.run("Vector3::normalized", [&] {
        ankerl::nanobench::doNotOptimizeAway(v3a.normalized());
    });

    bench.run("Vector3::dot", [&] {
        ankerl::nanobench::doNotOptimizeAway(v3a.dot(v3b));
    });

    bench.run("Vector3::cross", [&] {
        ankerl::nanobench::doNotOptimizeAway(v3a.cross(v3b));
    });

    bench.run("Vector3::operator+", [&] {
        ankerl::nanobench::doNotOptimizeAway(v3a + v3b);
    });

    bench.run("Vector3::operator*scalar", [&] {
        ankerl::nanobench::doNotOptimizeAway(v3a * 2.5);
    });

    // =========================================================================
    // Vector4
    // =========================================================================

    const Vector4 v4a(1.0, 2.0, 3.0, 4.0);
    const Vector4 v4b(5.0, 6.0, 7.0, 8.0);

    bench.run("Vector4::norm", [&] {
        ankerl::nanobench::doNotOptimizeAway(v4a.norm());
    });

    bench.run("Vector4::dot", [&] {
        ankerl::nanobench::doNotOptimizeAway(v4a.dot(v4b));
    });

    bench.run("Vector4::operator+", [&] {
        ankerl::nanobench::doNotOptimizeAway(v4a + v4b);
    });

    // =========================================================================
    // Matrix2
    // =========================================================================

    const Matrix2 m2a(1.0, 2.0, 3.0, 4.0);
    const Matrix2 m2b(5.0, 6.0, 7.0, 8.0);

    bench.run("Matrix2::determinant", [&] {
        ankerl::nanobench::doNotOptimizeAway(m2a.determinant());
    });

    bench.run("Matrix2::inverse", [&] {
        ankerl::nanobench::doNotOptimizeAway(m2a.inverse());
    });

    bench.run("Matrix2::transpose", [&] {
        ankerl::nanobench::doNotOptimizeAway(m2a.transpose());
    });

    bench.run("Matrix2::operator*mat", [&] {
        ankerl::nanobench::doNotOptimizeAway(m2a * m2b);
    });

    bench.run("Matrix2*Vector2", [&] {
        ankerl::nanobench::doNotOptimizeAway(m2a * v2a);
    });

    // =========================================================================
    // Matrix3
    // =========================================================================

    const Matrix3 m3a(2.0, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 2.0);
    const Matrix3 m3b(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);

    bench.run("Matrix3::determinant", [&] {
        ankerl::nanobench::doNotOptimizeAway(m3a.determinant());
    });

    bench.run("Matrix3::inverse", [&] {
        ankerl::nanobench::doNotOptimizeAway(m3a.inverse());
    });

    bench.run("Matrix3::transpose", [&] {
        ankerl::nanobench::doNotOptimizeAway(m3a.transpose());
    });

    bench.run("Matrix3::operator*mat", [&] {
        ankerl::nanobench::doNotOptimizeAway(m3a * m3b);
    });

    bench.run("Matrix3*Vector3", [&] {
        ankerl::nanobench::doNotOptimizeAway(m3a * v3a);
    });

    // =========================================================================
    // Matrix4
    // =========================================================================

    const Matrix4 m4a(2.0, 0.0, 0.0, 1.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 1.0, 0.0, 0.0, 2.0);
    const Matrix4 m4b = Matrix4::identity();

    bench.run("Matrix4::determinant", [&] {
        ankerl::nanobench::doNotOptimizeAway(m4a.determinant());
    });

    bench.run("Matrix4::inverse", [&] {
        ankerl::nanobench::doNotOptimizeAway(m4a.inverse());
    });

    bench.run("Matrix4::transpose", [&] {
        ankerl::nanobench::doNotOptimizeAway(m4a.transpose());
    });

    bench.run("Matrix4::operator*mat", [&] {
        ankerl::nanobench::doNotOptimizeAway(m4a * m4b);
    });

    bench.run("Matrix4*Vector4", [&] {
        ankerl::nanobench::doNotOptimizeAway(m4a * v4a);
    });

    // =========================================================================
    // Quaternion
    // =========================================================================

    const Quaternion qa(0.70710678, 0.70710678, 0.0, 0.0);
    const Quaternion qb(0.5, 0.5, 0.5, 0.5);

    bench.run("Quaternion::normalized", [&] {
        ankerl::nanobench::doNotOptimizeAway(qa.normalized());
    });

    bench.run("Quaternion::inverse", [&] {
        ankerl::nanobench::doNotOptimizeAway(qa.inverse());
    });

    bench.run("Quaternion::dot", [&] {
        ankerl::nanobench::doNotOptimizeAway(qa.dot(qb));
    });

    bench.run("Quaternion::operator*quat", [&] {
        ankerl::nanobench::doNotOptimizeAway(qa * qb);
    });

    bench.run("Quaternion*Vector3", [&] {
        ankerl::nanobench::doNotOptimizeAway(qa * v3a);
    });

    bench.run("slerp", [&] {
        ankerl::nanobench::doNotOptimizeAway(slerp(qa, qb, 0.5));
    });

    // =========================================================================
    // Cross-type conversions
    // =========================================================================

    bench.run("quaternion_to_matrix", [&] {
        ankerl::nanobench::doNotOptimizeAway(quaternion_to_matrix(qa));
    });

    bench.run("quaternion_to_euler_angles", [&] {
        ankerl::nanobench::doNotOptimizeAway(quaternion_to_euler_angles(qa));
    });

    const EulerAngles<AngleType::RADIANS> euler(0.1, 0.2, 0.3);
    bench.run("euler_angles_to_quaternion", [&] {
        ankerl::nanobench::doNotOptimizeAway(euler_angles_to_quaternion(euler));
    });

    // =========================================================================
    // Angle
    // =========================================================================

    const Angle<AngleType::RADIANS> angle_rad(1.5);

    bench.run("Angle::to_degrees", [&] {
        ankerl::nanobench::doNotOptimizeAway(angle_rad.to_degrees());
    });

    bench.run("Angle::sin", [&] {
        ankerl::nanobench::doNotOptimizeAway(angle_rad.sin());
    });

    bench.run("Angle::cos", [&] {
        ankerl::nanobench::doNotOptimizeAway(angle_rad.cos());
    });

    return 0;
}
