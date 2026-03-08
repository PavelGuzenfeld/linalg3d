#define ANKERL_NANOBENCH_IMPLEMENT
#include "linalg3d/linalg.hpp"
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <nanobench.h>

using namespace linalg3d;

int main()
{
    ankerl::nanobench::Bench bench;
    bench.title("linalg3d vs Eigen").warmup(100).minEpochIterations(1000);

    // =========================================================================
    // Vector3
    // =========================================================================

    const Vector3 v3a(1.0, 2.0, 3.0);
    const Vector3 v3b(4.0, 5.0, 6.0);
    const Eigen::Vector3d ev3a(1.0, 2.0, 3.0);
    const Eigen::Vector3d ev3b(4.0, 5.0, 6.0);

    bench.run("Vector3::dot", [&] { ankerl::nanobench::doNotOptimizeAway(v3a.dot(v3b)); });
    bench.run("Eigen::Vector3d::dot", [&] { ankerl::nanobench::doNotOptimizeAway(ev3a.dot(ev3b)); });

    bench.run("Vector3::cross", [&] { ankerl::nanobench::doNotOptimizeAway(v3a.cross(v3b)); });
    bench.run("Eigen::Vector3d::cross", [&] { ankerl::nanobench::doNotOptimizeAway(ev3a.cross(ev3b)); });

    bench.run("Vector3::norm", [&] { ankerl::nanobench::doNotOptimizeAway(v3a.norm()); });
    bench.run("Eigen::Vector3d::norm", [&] { ankerl::nanobench::doNotOptimizeAway(ev3a.norm()); });

    bench.run("Vector3::normalized", [&] { ankerl::nanobench::doNotOptimizeAway(v3a.normalized()); });
    bench.run("Eigen::Vector3d::normalized", [&] { ankerl::nanobench::doNotOptimizeAway(ev3a.normalized()); });

    bench.run("Vector3::operator+", [&] { ankerl::nanobench::doNotOptimizeAway(v3a + v3b); });
    bench.run("Eigen::Vector3d::operator+", [&] { ankerl::nanobench::doNotOptimizeAway((ev3a + ev3b).eval()); });

    bench.run("Vector3::operator*scalar", [&] { ankerl::nanobench::doNotOptimizeAway(v3a * 2.5); });
    bench.run("Eigen::Vector3d::operator*scalar", [&] { ankerl::nanobench::doNotOptimizeAway((ev3a * 2.5).eval()); });

    // =========================================================================
    // Matrix3
    // =========================================================================

    const Matrix3 m3a(2.0, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 2.0);
    const Matrix3 m3b(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    Eigen::Matrix3d em3a;
    em3a << 2.0, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 2.0;
    Eigen::Matrix3d em3b;
    em3b << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0;

    bench.run("Matrix3::determinant", [&] { ankerl::nanobench::doNotOptimizeAway(m3a.determinant()); });
    bench.run("Eigen::Matrix3d::determinant", [&] { ankerl::nanobench::doNotOptimizeAway(em3a.determinant()); });

    bench.run("Matrix3::inverse", [&] { ankerl::nanobench::doNotOptimizeAway(m3a.inverse()); });
    bench.run("Eigen::Matrix3d::inverse", [&] { ankerl::nanobench::doNotOptimizeAway(em3a.inverse().eval()); });

    bench.run("Matrix3::transpose", [&] { ankerl::nanobench::doNotOptimizeAway(m3a.transpose()); });
    bench.run("Eigen::Matrix3d::transpose", [&] { ankerl::nanobench::doNotOptimizeAway(em3a.transpose().eval()); });

    bench.run("Matrix3::operator*mat", [&] { ankerl::nanobench::doNotOptimizeAway(m3a * m3b); });
    bench.run("Eigen::Matrix3d::operator*mat", [&] { ankerl::nanobench::doNotOptimizeAway((em3a * em3b).eval()); });

    bench.run("Matrix3*Vector3", [&] { ankerl::nanobench::doNotOptimizeAway(m3a * v3a); });
    bench.run("Eigen::Matrix3d*Vector3d", [&] { ankerl::nanobench::doNotOptimizeAway((em3a * ev3a).eval()); });

    // =========================================================================
    // Matrix4
    // =========================================================================

    const Matrix4 m4a(2.0, 0.0, 0.0, 1.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 1.0, 0.0, 0.0, 2.0);
    const Matrix4 m4b = Matrix4::identity();
    Eigen::Matrix4d em4a;
    em4a << 2.0, 0.0, 0.0, 1.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 1.0, 0.0, 0.0, 2.0;
    const Eigen::Matrix4d em4b = Eigen::Matrix4d::Identity();

    bench.run("Matrix4::determinant", [&] { ankerl::nanobench::doNotOptimizeAway(m4a.determinant()); });
    bench.run("Eigen::Matrix4d::determinant", [&] { ankerl::nanobench::doNotOptimizeAway(em4a.determinant()); });

    bench.run("Matrix4::inverse", [&] { ankerl::nanobench::doNotOptimizeAway(m4a.inverse()); });
    bench.run("Eigen::Matrix4d::inverse", [&] { ankerl::nanobench::doNotOptimizeAway(em4a.inverse().eval()); });

    bench.run("Matrix4::operator*mat", [&] { ankerl::nanobench::doNotOptimizeAway(m4a * m4b); });
    bench.run("Eigen::Matrix4d::operator*mat", [&] { ankerl::nanobench::doNotOptimizeAway((em4a * em4b).eval()); });

    // =========================================================================
    // Quaternion
    // =========================================================================

    const Quaternion qa(0.70710678, 0.70710678, 0.0, 0.0);
    const Quaternion qb(0.5, 0.5, 0.5, 0.5);
    const Eigen::Quaterniond eqa(0.70710678, 0.70710678, 0.0, 0.0);
    const Eigen::Quaterniond eqb(0.5, 0.5, 0.5, 0.5);

    bench.run("Quaternion::normalized", [&] { ankerl::nanobench::doNotOptimizeAway(qa.normalized()); });
    bench.run("Eigen::Quaterniond::normalized", [&] { ankerl::nanobench::doNotOptimizeAway(eqa.normalized()); });

    bench.run("Quaternion::inverse", [&] { ankerl::nanobench::doNotOptimizeAway(qa.inverse()); });
    bench.run("Eigen::Quaterniond::inverse", [&] { ankerl::nanobench::doNotOptimizeAway(eqa.inverse()); });

    bench.run("Quaternion::operator*quat", [&] { ankerl::nanobench::doNotOptimizeAway(qa * qb); });
    bench.run("Eigen::Quaterniond::operator*quat", [&] { ankerl::nanobench::doNotOptimizeAway(eqa * eqb); });

    bench.run("Quaternion*Vector3", [&] { ankerl::nanobench::doNotOptimizeAway(qa * v3a); });
    bench.run("Eigen::Quaterniond*Vector3d", [&] { ankerl::nanobench::doNotOptimizeAway((eqa * ev3a).eval()); });

    bench.run("slerp", [&] { ankerl::nanobench::doNotOptimizeAway(slerp(qa, qb, 0.5)); });
    bench.run("Eigen::slerp", [&] { ankerl::nanobench::doNotOptimizeAway(eqa.slerp(0.5, eqb)); });

    // =========================================================================
    // Cross-type conversions
    // =========================================================================

    bench.run("quaternion_to_matrix", [&] { ankerl::nanobench::doNotOptimizeAway(quaternion_to_matrix(qa)); });
    bench.run("Eigen::quat_to_matrix", [&] { ankerl::nanobench::doNotOptimizeAway(eqa.toRotationMatrix()); });

    bench.run("quaternion_to_euler_angles",
              [&] { ankerl::nanobench::doNotOptimizeAway(quaternion_to_euler_angles(qa)); });
    bench.run("Eigen::matrix_to_euler",
              [&] { ankerl::nanobench::doNotOptimizeAway(eqa.toRotationMatrix().eulerAngles(0, 1, 2)); });

    const EulerAngles<AngleType::RADIANS> euler(0.1, 0.2, 0.3);
    bench.run("euler_angles_to_quaternion",
              [&] { ankerl::nanobench::doNotOptimizeAway(euler_angles_to_quaternion(euler)); });

    // =========================================================================
    // Angle trig (linalg3d only — no Eigen equivalent)
    // =========================================================================

    const Angle<AngleType::RADIANS> angle_rad(1.5);

    bench.run("Angle::sin", [&] { ankerl::nanobench::doNotOptimizeAway(angle_rad.sin()); });
    bench.run("Angle::cos", [&] { ankerl::nanobench::doNotOptimizeAway(angle_rad.cos()); });

    return 0;
}
