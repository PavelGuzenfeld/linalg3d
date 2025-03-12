#pragma once
#include "euler_angles.hpp"
#include "matrix3x3.hpp"
#include "vector3.hpp"
#include <cmath>

namespace linalg3d
{

    class Quaternion
    {
    public:
        double w{}, x{}, y{}, z{};

        explicit constexpr Quaternion(double w = 0.0, double x = 0.0, double y = 0.0, double z = 0.0) noexcept
            : w{w}, x{x}, y{y}, z{z} {}

        [[nodiscard]] constexpr Quaternion normalized() const noexcept
        {
            double n = norm();
            return (n > 0.0) ? Quaternion{w / n, x / n, y / n, z / n} : Quaternion{};
        }

        [[nodiscard]] constexpr Quaternion inverse() const noexcept
        {
            double n_sq = w * w + x * x + y * y + z * z;
            return (n_sq > 0.0) ? Quaternion{w / n_sq, -x / n_sq, -y / n_sq, -z / n_sq} : Quaternion{};
        }

        [[nodiscard]] constexpr double dot(const Quaternion &other) const noexcept
        {
            return w * other.w + x * other.x + y * other.y + z * other.z;
        }

        [[nodiscard]] constexpr Quaternion operator+(const Quaternion &q) const noexcept
        {
            return Quaternion{
                w + q.w,
                x + q.x,
                y + q.y,
                z + q.z};
        }

        [[nodiscard]] constexpr Quaternion operator-(const Quaternion &q) const noexcept
        {
            return Quaternion{
                w - q.w,
                x - q.x,
                y - q.y,
                z - q.z};
        }

        [[nodiscard]] constexpr Quaternion operator*(const Quaternion &q) const noexcept
        {
            return Quaternion{
                w * q.w - x * q.x - y * q.y - z * q.z,
                w * q.x + x * q.w + y * q.z - z * q.y,
                w * q.y - x * q.z + y * q.w + z * q.x,
                w * q.z + x * q.y - y * q.x + z * q.w};
        }

        [[nodiscard]] constexpr Quaternion operator*(double scalar) const noexcept
        {
            return Quaternion{w * scalar, x * scalar, y * scalar, z * scalar};
        }

        [[nodiscard]] constexpr Quaternion operator/(double scalar) const noexcept
        {
            return Quaternion{w / scalar, x / scalar, y / scalar, z / scalar};
        }

        [[nodiscard]] constexpr bool operator==(const Quaternion &q) const noexcept
        {
            return w == q.w && x == q.x && y == q.y && z == q.z;
        }

        [[nodiscard]] constexpr bool operator!=(const Quaternion &q) const noexcept
        {
            return !(*this == q);
        }

        [[nodiscard]] constexpr double norm_sq() const noexcept
        {
            return w * w + x * x + y * y + z * z;
        }

        [[nodiscard]] constexpr double norm() const noexcept
        {
            return std::sqrt(norm_sq());
        }
    };

} // namespace linalg3d
