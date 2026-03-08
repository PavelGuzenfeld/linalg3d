#pragma once
#include "constexpr_math.hpp"
#include <expected>

namespace linalg3d
{
class Matrix3
{
public:
    double m[3][3]{};

    constexpr Matrix3() noexcept = default;

    constexpr Matrix3(double m00,
                      double m01,
                      double m02,
                      double m10,
                      double m11,
                      double m12,
                      double m20,
                      double m21,
                      double m22) noexcept
        : m{{m00, m01, m02}, {m10, m11, m12}, {m20, m21, m22}}
    {
    }

    [[nodiscard]] static constexpr Matrix3 identity() noexcept
    {
        return Matrix3{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    }

    [[nodiscard]] constexpr Matrix3 transpose() const noexcept
    {
        return Matrix3{m[0][0], m[1][0], m[2][0], m[0][1], m[1][1], m[2][1], m[0][2], m[1][2], m[2][2]};
    }

    [[nodiscard]] constexpr double determinant() const noexcept
    {
        return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
               m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    }

    [[nodiscard]] constexpr std::expected<Matrix3, MatrixError> inverse() const noexcept
    {
        double det = determinant();
        if (det == 0.0)
            return std::unexpected{MatrixError::singular};

        double inv_det = 1.0 / det;
        return Matrix3{(m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det,
                       (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det,
                       (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det,

                       (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det,
                       (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det,
                       (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det,

                       (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det,
                       (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det,
                       (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det};
    }

    [[nodiscard]] constexpr bool operator==(const Matrix3 &other) const noexcept
    {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                if (m[i][j] != other.m[i][j])
                    return false;
        return true;
    }

    [[nodiscard]] constexpr bool operator!=(const Matrix3 &other) const noexcept
    {
        return !(*this == other);
    }

    [[nodiscard]] constexpr Matrix3 operator+(const Matrix3 &other) const noexcept
    {
        return Matrix3{m[0][0] + other.m[0][0],
                       m[0][1] + other.m[0][1],
                       m[0][2] + other.m[0][2],
                       m[1][0] + other.m[1][0],
                       m[1][1] + other.m[1][1],
                       m[1][2] + other.m[1][2],
                       m[2][0] + other.m[2][0],
                       m[2][1] + other.m[2][1],
                       m[2][2] + other.m[2][2]};
    }

    [[nodiscard]] constexpr Matrix3 operator-(const Matrix3 &other) const noexcept
    {
        return Matrix3{m[0][0] - other.m[0][0],
                       m[0][1] - other.m[0][1],
                       m[0][2] - other.m[0][2],
                       m[1][0] - other.m[1][0],
                       m[1][1] - other.m[1][1],
                       m[1][2] - other.m[1][2],
                       m[2][0] - other.m[2][0],
                       m[2][1] - other.m[2][1],
                       m[2][2] - other.m[2][2]};
    }

    [[nodiscard]] constexpr Matrix3 operator*(double scalar) const noexcept
    {
        return Matrix3{m[0][0] * scalar,
                       m[0][1] * scalar,
                       m[0][2] * scalar,
                       m[1][0] * scalar,
                       m[1][1] * scalar,
                       m[1][2] * scalar,
                       m[2][0] * scalar,
                       m[2][1] * scalar,
                       m[2][2] * scalar};
    }

    [[nodiscard]] constexpr Matrix3 operator*(const Matrix3 &other) const noexcept
    {
        return Matrix3{m[0][0] * other.m[0][0] + m[0][1] * other.m[1][0] + m[0][2] * other.m[2][0],
                       m[0][0] * other.m[0][1] + m[0][1] * other.m[1][1] + m[0][2] * other.m[2][1],
                       m[0][0] * other.m[0][2] + m[0][1] * other.m[1][2] + m[0][2] * other.m[2][2],

                       m[1][0] * other.m[0][0] + m[1][1] * other.m[1][0] + m[1][2] * other.m[2][0],
                       m[1][0] * other.m[0][1] + m[1][1] * other.m[1][1] + m[1][2] * other.m[2][1],
                       m[1][0] * other.m[0][2] + m[1][1] * other.m[1][2] + m[1][2] * other.m[2][2],

                       m[2][0] * other.m[0][0] + m[2][1] * other.m[1][0] + m[2][2] * other.m[2][0],
                       m[2][0] * other.m[0][1] + m[2][1] * other.m[1][1] + m[2][2] * other.m[2][1],
                       m[2][0] * other.m[0][2] + m[2][1] * other.m[1][2] + m[2][2] * other.m[2][2]};
    }

    [[nodiscard]] constexpr Matrix3 operator/(double scalar) const noexcept
    {
        return Matrix3{m[0][0] / scalar,
                       m[0][1] / scalar,
                       m[0][2] / scalar,
                       m[1][0] / scalar,
                       m[1][1] / scalar,
                       m[1][2] / scalar,
                       m[2][0] / scalar,
                       m[2][1] / scalar,
                       m[2][2] / scalar};
    }

    constexpr Matrix3 &operator+=(const Matrix3 &other) noexcept
    {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                m[i][j] += other.m[i][j];
        return *this;
    }

    constexpr Matrix3 &operator-=(const Matrix3 &other) noexcept
    {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                m[i][j] -= other.m[i][j];
        return *this;
    }

    constexpr Matrix3 &operator*=(double scalar) noexcept
    {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                m[i][j] *= scalar;
        return *this;
    }

    constexpr Matrix3 &operator*=(const Matrix3 &other) noexcept
    {
        *this = *this * other;
        return *this;
    }

    constexpr Matrix3 &operator/=(double scalar) noexcept
    {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                m[i][j] /= scalar;
        return *this;
    }
};

} // namespace linalg3d
