#pragma once
#include "euler_angles.hpp"
#include "matrix3x3.hpp"
#include "quaternion.hpp"
#include "vector3.hpp"
#include <algorithm> // std::clamp

namespace linalg3d
{
    [[nodiscard]] constexpr Vector3 operator*(const Vector3 &v, const Quaternion &q) noexcept
    {
        Quaternion v_quat{0.0, v.x, v.y, v.z};
        Quaternion result = q * v_quat * q.inverse();
        return Vector3{result.x, result.y, result.z};
    }

    [[nodiscard]] constexpr Matrix3 toRotationMatrix(const Quaternion &q) noexcept
    {
        double xx = q.x * q.x, yy = q.y * q.y, zz = q.z * q.z;
        double wx = q.w * q.x, wy = q.w * q.y, wz = q.w * q.z;
        double xy = q.x * q.y, xz = q.x * q.z, yz = q.y * q.z;

        return Matrix3{
            1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy),
            2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx),
            2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)};
    }

    [[nodiscard]] constexpr EulerAngles<AngleType::RADIANS> fromQuaternion(const Quaternion &q) noexcept
    {
        return EulerAngles<AngleType::RADIANS>{
            gcem::atan2(2.0 * (q.w * q.x + q.y * q.z), 1.0 - 2.0 * (q.x * q.x + q.y * q.y)), // Pitch
            gcem::atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)), // Yaw
            gcem::asin(std::clamp(2.0 * (q.w * q.y - q.z * q.x), -1.0, 1.0))                 // Roll
        };
    }

    [[nodiscard]] constexpr Quaternion fromEulerAngles(EulerAngles<AngleType::RADIANS> const &angles) noexcept
    {
        double cy = gcem::cos(angles.yaw * 0.5);
        double sy = gcem::sin(angles.yaw * 0.5);
        double cp = gcem::cos(angles.pitch * 0.5);
        double sp = gcem::sin(angles.pitch * 0.5);
        double cr = gcem::cos(angles.roll * 0.5);
        double sr = gcem::sin(angles.roll * 0.5);

        return Quaternion{
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy};
    }

    [[nodiscard]] constexpr linalg3d::Vector3 operator*(const linalg3d::Matrix3 &mat, const linalg3d::Vector3 &vec)
    {
        return linalg3d::Vector3(
            mat.m[0][0] * vec.x + mat.m[0][1] * vec.y + mat.m[0][2] * vec.z,
            mat.m[1][0] * vec.x + mat.m[1][1] * vec.y + mat.m[1][2] * vec.z,
            mat.m[2][0] * vec.x + mat.m[2][1] * vec.y + mat.m[2][2] * vec.z);
    }

} // namespace linalg3d
