#pragma once
#include "constexpr_math.hpp"
#include "gcem.hpp"
#include <compare>

namespace linalg3d
{

    struct Vector4
    {
        double x{}, y{}, z{}, w{};

        explicit constexpr Vector4(double x_a = 0.0, double y_a = 0.0, double z_a = 0.0, double w_a = 0.0) noexcept
            : x{x_a}, y{y_a}, z{z_a}, w{w_a} {}

        [[nodiscard]] constexpr double norm() const noexcept
        {
            return gcem::sqrt(norm_sq());
        }

        [[nodiscard]] constexpr double norm_sq() const noexcept
        {
            return x * x + y * y + z * z + w * w;
        }

        [[nodiscard]] constexpr Vector4 normalized() const noexcept
        {
            double n = norm();
            return (n > 0.0) ? Vector4{x / n, y / n, z / n, w / n} : Vector4{};
        }

        [[nodiscard]] constexpr double dot(const Vector4 &other) const noexcept
        {
            return x * other.x + y * other.y + z * other.z + w * other.w;
        }

        [[nodiscard]] constexpr Vector4 operator+(const Vector4 &other) const noexcept
        {
            return Vector4{x + other.x, y + other.y, z + other.z, w + other.w};
        }

        [[nodiscard]] constexpr Vector4 operator-(const Vector4 &other) const noexcept
        {
            return Vector4{x - other.x, y - other.y, z - other.z, w - other.w};
        }

        [[nodiscard]] constexpr Vector4 operator-() const noexcept
        {
            return Vector4{-x, -y, -z, -w};
        }

        [[nodiscard]] constexpr Vector4 operator*(double scalar) const noexcept
        {
            return Vector4{x * scalar, y * scalar, z * scalar, w * scalar};
        }

        [[nodiscard]] constexpr Vector4 operator/(double scalar) const noexcept
        {
            return Vector4{x / scalar, y / scalar, z / scalar, w / scalar};
        }

        constexpr Vector4 &operator+=(const Vector4 &other) noexcept
        {
            x += other.x;
            y += other.y;
            z += other.z;
            w += other.w;
            return *this;
        }

        constexpr Vector4 &operator-=(const Vector4 &other) noexcept
        {
            x -= other.x;
            y -= other.y;
            z -= other.z;
            w -= other.w;
            return *this;
        }

        constexpr Vector4 &operator*=(double scalar) noexcept
        {
            x *= scalar;
            y *= scalar;
            z *= scalar;
            w *= scalar;
            return *this;
        }

        constexpr Vector4 &operator/=(double scalar) noexcept
        {
            x /= scalar;
            y /= scalar;
            z /= scalar;
            w /= scalar;
            return *this;
        }

        [[nodiscard]] constexpr auto operator<=>(const Vector4 &other) const noexcept
        {
            if (auto cmp = compare_double(x, other.x); cmp != 0)
                return cmp;
            if (auto cmp = compare_double(y, other.y); cmp != 0)
                return cmp;
            if (auto cmp = compare_double(z, other.z); cmp != 0)
                return cmp;
            return compare_double(w, other.w);
        }

        [[nodiscard]] constexpr bool operator==(const Vector4 &other) const noexcept
        {
            return x == other.x && y == other.y && z == other.z && w == other.w;
        }

        [[nodiscard]] constexpr Vector4 abs() const noexcept
        {
            return Vector4{std::abs(x), std::abs(y), std::abs(z), std::abs(w)};
        }
    };

} // namespace linalg3d
