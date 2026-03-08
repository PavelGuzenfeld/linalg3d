#pragma once
#include "euler_angles.hpp"
#include "matrix2x2.hpp"
#include "matrix3x3.hpp"
#include "matrix4x4.hpp"
#include "quaternion.hpp"
#include "vector2.hpp"
#include "vector3.hpp"
#include "vector4.hpp"

namespace linalg3d
{
    // --- Vector * Quaternion rotation ---

    [[nodiscard]] constexpr Vector3 operator*(const Vector3 &v, const Quaternion &q) noexcept
    {
        Quaternion v_quat{0.0, v.x, v.y, v.z};
        Quaternion result = q * v_quat * q.inverse();
        return Vector3{result.x, result.y, result.z};
    }

    // --- Quaternion <-> Matrix3 ---

    [[nodiscard]] constexpr Matrix3 quaternion_to_matrix(const Quaternion &q) noexcept
    {
        double xx = q.x * q.x, yy = q.y * q.y, zz = q.z * q.z;
        double wx = q.w * q.x, wy = q.w * q.y, wz = q.w * q.z;
        double xy = q.x * q.y, xz = q.x * q.z, yz = q.y * q.z;

        return Matrix3{
            1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy),
            2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx),
            2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)};
    }

    // --- Quaternion <-> EulerAngles ---

    [[nodiscard]] constexpr EulerAngles<AngleType::RADIANS> quaternion_to_euler_angles(const Quaternion &q) noexcept
    {
        return EulerAngles<AngleType::RADIANS>{
            gcem::asin(clamp(2.0 * (q.w * q.y - q.z * q.x), -1.0, 1.0)),
            gcem::atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)),
            gcem::atan2(2.0 * (q.w * q.x + q.y * q.z), 1.0 - 2.0 * (q.x * q.x + q.y * q.y))};
    }

    [[nodiscard]] constexpr Vector3 quaternion_to_vector3(const Quaternion &q) noexcept
    {
        return Vector3{q.x, q.y, q.z};
    }

    [[nodiscard]] constexpr Quaternion euler_angles_to_quaternion(EulerAngles<AngleType::RADIANS> const &angles) noexcept
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

    // --- Matrix * Vector operations ---

    [[nodiscard]] constexpr Vector2 operator*(const Matrix2 &mat, const Vector2 &vec) noexcept
    {
        return Vector2{
            mat.m[0][0] * vec.x + mat.m[0][1] * vec.y,
            mat.m[1][0] * vec.x + mat.m[1][1] * vec.y};
    }

    [[nodiscard]] constexpr Vector3 operator*(const Matrix3 &mat, const Vector3 &vec) noexcept
    {
        return Vector3{
            mat.m[0][0] * vec.x + mat.m[0][1] * vec.y + mat.m[0][2] * vec.z,
            mat.m[1][0] * vec.x + mat.m[1][1] * vec.y + mat.m[1][2] * vec.z,
            mat.m[2][0] * vec.x + mat.m[2][1] * vec.y + mat.m[2][2] * vec.z};
    }

    [[nodiscard]] constexpr Vector4 operator*(const Matrix4 &mat, const Vector4 &vec) noexcept
    {
        return Vector4{
            mat.m[0][0] * vec.x + mat.m[0][1] * vec.y + mat.m[0][2] * vec.z + mat.m[0][3] * vec.w,
            mat.m[1][0] * vec.x + mat.m[1][1] * vec.y + mat.m[1][2] * vec.z + mat.m[1][3] * vec.w,
            mat.m[2][0] * vec.x + mat.m[2][1] * vec.y + mat.m[2][2] * vec.z + mat.m[2][3] * vec.w,
            mat.m[3][0] * vec.x + mat.m[3][1] * vec.y + mat.m[3][2] * vec.z + mat.m[3][3] * vec.w};
    }

    // --- EulerAngles <-> Vector3 ---

    template <AngleType T>
    [[nodiscard]] constexpr Vector3 euler_angles_to_vector3(const EulerAngles<T> &angles) noexcept
    {
        return Vector3{angles.pitch.value(), angles.roll.value(), angles.yaw.value()};
    }

    [[nodiscard]] constexpr EulerAngles<AngleType::RADIANS> vector3_to_euler_angles(const Vector3 &vec) noexcept
    {
        return EulerAngles<AngleType::RADIANS>{vec.y, vec.z, vec.x};
    }

} // namespace linalg3d
