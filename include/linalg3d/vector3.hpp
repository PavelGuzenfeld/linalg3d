#pragma once
#include <cmath>
#include <compare> // for std::partial_ordering

namespace linalg3d
{

    constexpr std::weak_ordering compareDouble(double a, double b)
    {
        // Example: interpret NaNs as “always greater,” or whichever logic you prefer.
        // This is a simplistic approach; real robust code might do bit tricks for total ordering.
        if (std::isnan(a) && std::isnan(b))
            return std::weak_ordering::equivalent;
        if (std::isnan(a))
            return std::weak_ordering::greater;
        if (std::isnan(b))
            return std::weak_ordering::less;
        // Fallback to normal comparison
        if (a < b)
            return std::weak_ordering::less;
        if (a > b)
            return std::weak_ordering::greater;
        return std::weak_ordering::equivalent;
    }

    struct Vector3
    {
        double x{}, y{}, z{};

        explicit constexpr Vector3(double x = 0.0, double y = 0.0, double z = 0.0) noexcept
            : x{x}, y{y}, z{z} {}

        [[nodiscard]] constexpr double norm() const noexcept
        {
            return std::sqrt(norm_sq());
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
            return Vector3{
                y * other.z - z * other.y,
                z * other.x - x * other.z,
                x * other.y - y * other.x};
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

        [[nodiscard]] constexpr auto operator<=>(const Vector3 &other) const noexcept
        {
            if (auto cmp = compareDouble(x, other.x); cmp != 0)
                return cmp;
            if (auto cmp = compareDouble(y, other.y); cmp != 0)
                return cmp;
            return compareDouble(z, other.z);
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
    };

} // namespace linalg3d