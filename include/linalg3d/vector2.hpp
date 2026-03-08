#pragma once
#include "constexpr_math.hpp"
#include <compare>
#include <gcem.hpp>

namespace linalg3d
{

struct Vector2
{
    double x{}, y{};

    explicit constexpr Vector2(double x_a = 0.0, double y_a = 0.0) noexcept : x{x_a}, y{y_a}
    {
    }

    [[nodiscard]] constexpr double norm() const noexcept
    {
        return gcem::sqrt(norm_sq());
    }

    [[nodiscard]] constexpr double norm_sq() const noexcept
    {
        return x * x + y * y;
    }

    [[nodiscard]] constexpr Vector2 normalized() const noexcept
    {
        double n = norm();
        return (n > 0.0) ? Vector2{x / n, y / n} : Vector2{};
    }

    [[nodiscard]] constexpr double dot(const Vector2 &other) const noexcept
    {
        return x * other.x + y * other.y;
    }

    [[nodiscard]] constexpr double cross(const Vector2 &other) const noexcept
    {
        return x * other.y - y * other.x;
    }

    [[nodiscard]] constexpr Vector2 operator+(const Vector2 &other) const noexcept
    {
        return Vector2{x + other.x, y + other.y};
    }

    [[nodiscard]] constexpr Vector2 operator-(const Vector2 &other) const noexcept
    {
        return Vector2{x - other.x, y - other.y};
    }

    [[nodiscard]] constexpr Vector2 operator-() const noexcept
    {
        return Vector2{-x, -y};
    }

    [[nodiscard]] constexpr Vector2 operator*(double scalar) const noexcept
    {
        return Vector2{x * scalar, y * scalar};
    }

    [[nodiscard]] constexpr Vector2 operator/(double scalar) const noexcept
    {
        return Vector2{x / scalar, y / scalar};
    }

    constexpr Vector2 &operator+=(const Vector2 &other) noexcept
    {
        x += other.x;
        y += other.y;
        return *this;
    }

    constexpr Vector2 &operator-=(const Vector2 &other) noexcept
    {
        x -= other.x;
        y -= other.y;
        return *this;
    }

    constexpr Vector2 &operator*=(double scalar) noexcept
    {
        x *= scalar;
        y *= scalar;
        return *this;
    }

    constexpr Vector2 &operator/=(double scalar) noexcept
    {
        x /= scalar;
        y /= scalar;
        return *this;
    }

    [[nodiscard]] constexpr auto operator<=>(const Vector2 &other) const noexcept
    {
        if (auto cmp = compare_double(x, other.x); cmp != 0)
            return cmp;
        return compare_double(y, other.y);
    }

    [[nodiscard]] constexpr bool operator==(const Vector2 &other) const noexcept
    {
        return x == other.x && y == other.y;
    }

    [[nodiscard]] constexpr Vector2 abs() const noexcept
    {
        return Vector2{fabs(x), fabs(y)};
    }
};

} // namespace linalg3d
