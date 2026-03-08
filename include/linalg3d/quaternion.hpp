#pragma once
#include "constexpr_math.hpp"
#include "euler_angles.hpp"
#include "matrix3x3.hpp"
#include "vector3.hpp"

namespace linalg3d
{

class Quaternion
{
public:
    double w{1.0}, x{}, y{}, z{};

    static constexpr Quaternion identity() noexcept
    {
        return Quaternion{};
    }

    explicit constexpr Quaternion(double w_a = 1.0, double x_a = 0.0, double y_a = 0.0, double z_a = 0.0) noexcept
        : w{w_a}, x{x_a}, y{y_a}, z{z_a}
    {
    }

    [[nodiscard]] constexpr Quaternion normalized() const noexcept
    {
        const double n = norm();
        return (n > 0.0) ? Quaternion{w / n, x / n, y / n, z / n} : Quaternion{};
    }

    [[nodiscard]] constexpr Quaternion inverse() const noexcept
    {
        const double n_sq = w * w + x * x + y * y + z * z;
        return (n_sq > 0.0) ? Quaternion{w / n_sq, -x / n_sq, -y / n_sq, -z / n_sq} : Quaternion{};
    }

    [[nodiscard]] constexpr double dot(const Quaternion &other) const noexcept
    {
        return w * other.w + x * other.x + y * other.y + z * other.z;
    }

    [[nodiscard]] constexpr Quaternion operator+(const Quaternion &q) const noexcept
    {
        return Quaternion{w + q.w, x + q.x, y + q.y, z + q.z};
    }

    [[nodiscard]] constexpr Quaternion operator-(const Quaternion &q) const noexcept
    {
        return Quaternion{w - q.w, x - q.x, y - q.y, z - q.z};
    }

    [[nodiscard]] constexpr Quaternion operator*(const Quaternion &q) const noexcept
    {
        return Quaternion{w * q.w - x * q.x - y * q.y - z * q.z,
                          w * q.x + x * q.w + y * q.z - z * q.y,
                          w * q.y - x * q.z + y * q.w + z * q.x,
                          w * q.z + x * q.y - y * q.x + z * q.w};
    }

    [[nodiscard]] constexpr Quaternion operator*(double scalar) const noexcept
    {
        return Quaternion{w * scalar, x * scalar, y * scalar, z * scalar};
    }

    [[nodiscard]] constexpr Vector3 operator*(const Vector3 &v) const noexcept
    {
        const Quaternion v_quat{0.0, v.x, v.y, v.z};
        const Quaternion result = *this * v_quat * inverse();
        return Vector3{result.x, result.y, result.z};
    }

    [[nodiscard]] constexpr Quaternion operator/(double scalar) const noexcept
    {
        return Quaternion{w / scalar, x / scalar, y / scalar, z / scalar};
    }

    constexpr Quaternion &operator+=(const Quaternion &q) noexcept
    {
        w += q.w;
        x += q.x;
        y += q.y;
        z += q.z;
        return *this;
    }

    constexpr Quaternion &operator-=(const Quaternion &q) noexcept
    {
        w -= q.w;
        x -= q.x;
        y -= q.y;
        z -= q.z;
        return *this;
    }

    constexpr Quaternion &operator*=(const Quaternion &q) noexcept
    {
        *this = *this * q;
        return *this;
    }

    constexpr Quaternion &operator*=(double scalar) noexcept
    {
        w *= scalar;
        x *= scalar;
        y *= scalar;
        z *= scalar;
        return *this;
    }

    constexpr Quaternion &operator/=(double scalar) noexcept
    {
        w /= scalar;
        x /= scalar;
        y /= scalar;
        z /= scalar;
        return *this;
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
        return ce_sqrt(norm_sq());
    }
};

[[nodiscard]] constexpr Quaternion slerp(const Quaternion &a, const Quaternion &b, double t) noexcept
{
    double cos_theta = a.dot(b);

    const Quaternion b_adj = cos_theta < 0.0 ? Quaternion{-b.w, -b.x, -b.y, -b.z} : b;
    cos_theta = cos_theta < 0.0 ? -cos_theta : cos_theta;

    if (cos_theta > 0.9995)
    {
        const Quaternion result{a.w + t * (b_adj.w - a.w),
                                a.x + t * (b_adj.x - a.x),
                                a.y + t * (b_adj.y - a.y),
                                a.z + t * (b_adj.z - a.z)};
        return result.normalized();
    }

    const double theta = ce_acos(cos_theta);
    const double sin_theta = ce_sin(theta);
    const double wa = ce_sin((1.0 - t) * theta) / sin_theta;
    const double wb = ce_sin(t * theta) / sin_theta;

    return Quaternion{wa * a.w + wb * b_adj.w,
                      wa * a.x + wb * b_adj.x,
                      wa * a.y + wb * b_adj.y,
                      wa * a.z + wb * b_adj.z};
}

} // namespace linalg3d
