#pragma once
#include "constexpr_math.hpp"
#include <compare>
#include <gcem.hpp>

namespace linalg3d
{

struct Vector3
{
    double x{}, y{}, z{};

    explicit constexpr Vector3(double x_a = 0.0, double y_a = 0.0, double z_a = 0.0) noexcept : x{x_a}, y{y_a}, z{z_a}
    {
    }

    [[nodiscard]] constexpr double norm() const noexcept
    {
        return gcem::sqrt(norm_sq());
    }

    [[nodiscard]] constexpr double norm_sq() const noexcept
    {
        return x * x + y * y + z * z;
    }

    [[nodiscard]] constexpr Vector3 normalized() const noexcept
    {
        double n = norm();
        return (n > 0.0) ? Vector3{x / n, y / n, z / n} : Vector3{};
    }

    [[nodiscard]] constexpr double dot(const Vector3 &other) const noexcept
    {
        return x * other.x + y * other.y + z * other.z;
    }

    [[nodiscard]] constexpr Vector3 cross(const Vector3 &other) const noexcept
    {
        return Vector3{y * other.z - z * other.y, z * other.x - x * other.z, x * other.y - y * other.x};
    }

    [[nodiscard]] constexpr Vector3 operator+(const Vector3 &other) const noexcept
    {
        return Vector3{x + other.x, y + other.y, z + other.z};
    }

    [[nodiscard]] constexpr Vector3 operator-(const Vector3 &other) const noexcept
    {
        return Vector3{x - other.x, y - other.y, z - other.z};
    }

    [[nodiscard]] constexpr Vector3 operator-() const noexcept
    {
        return Vector3{-x, -y, -z};
    }

    [[nodiscard]] constexpr Vector3 operator*(double scalar) const noexcept
    {
        return Vector3{x * scalar, y * scalar, z * scalar};
    }

    [[nodiscard]] constexpr Vector3 operator/(double scalar) const noexcept
    {
        return Vector3{x / scalar, y / scalar, z / scalar};
    }

    constexpr Vector3 &operator+=(const Vector3 &other) noexcept
    {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }

    constexpr Vector3 &operator-=(const Vector3 &other) noexcept
    {
        x -= other.x;
        y -= other.y;
        z -= other.z;
        return *this;
    }

    constexpr Vector3 &operator*=(double scalar) noexcept
    {
        x *= scalar;
        y *= scalar;
        z *= scalar;
        return *this;
    }

    constexpr Vector3 &operator/=(double scalar) noexcept
    {
        x /= scalar;
        y /= scalar;
        z /= scalar;
        return *this;
    }

    [[nodiscard]] constexpr auto operator<=>(const Vector3 &other) const noexcept
    {
        if (auto cmp = compare_double(x, other.x); cmp != 0)
            return cmp;
        if (auto cmp = compare_double(y, other.y); cmp != 0)
            return cmp;
        return compare_double(z, other.z);
    }

    [[nodiscard]] constexpr bool operator==(const Vector3 &other) const noexcept
    {
        return x == other.x && y == other.y && z == other.z;
    }

    [[nodiscard]] constexpr bool operator!=(const Vector3 &other) const noexcept
    {
        return !(*this == other);
    }

    [[nodiscard]] constexpr bool operator<(const Vector3 &other) const noexcept
    {
        return (*this <=> other) == std::weak_ordering::less;
    }

    [[nodiscard]] constexpr bool operator>(const Vector3 &other) const noexcept
    {
        return (*this <=> other) == std::weak_ordering::greater;
    }

    [[nodiscard]] constexpr bool operator<=(const Vector3 &other) const noexcept
    {
        return (*this <=> other) != std::weak_ordering::greater;
    }

    [[nodiscard]] constexpr bool operator>=(const Vector3 &other) const noexcept
    {
        return (*this <=> other) != std::weak_ordering::less;
    }

    [[nodiscard]] constexpr Vector3 abs() const noexcept
    {
        return Vector3{fabs(x), fabs(y), fabs(z)};
    }
};

} // namespace linalg3d
