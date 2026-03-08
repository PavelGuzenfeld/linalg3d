#pragma once
#include "constexpr_math.hpp"
#include <expected>

namespace linalg3d
{
class Matrix2
{
public:
    double m[2][2]{};

    constexpr Matrix2() noexcept = default;

    constexpr Matrix2(double m00, double m01, double m10, double m11) noexcept : m{{m00, m01}, {m10, m11}}
    {
    }

    [[nodiscard]] static constexpr Matrix2 identity() noexcept
    {
        return Matrix2{1.0, 0.0, 0.0, 1.0};
    }

    [[nodiscard]] constexpr Matrix2 transpose() const noexcept
    {
        return Matrix2{m[0][0], m[1][0], m[0][1], m[1][1]};
    }

    [[nodiscard]] constexpr double determinant() const noexcept
    {
        return m[0][0] * m[1][1] - m[0][1] * m[1][0];
    }

    [[nodiscard]] constexpr std::expected<Matrix2, MatrixError> inverse() const noexcept
    {
        const double det = determinant();
        if (det == 0.0)
            return std::unexpected{MatrixError::singular};

        const double inv_det = 1.0 / det;
        return Matrix2{m[1][1] * inv_det, -m[0][1] * inv_det, -m[1][0] * inv_det, m[0][0] * inv_det};
    }

    [[nodiscard]] constexpr bool operator==(const Matrix2 &other) const noexcept
    {
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                if (m[i][j] != other.m[i][j])
                    return false;
        return true;
    }

    [[nodiscard]] constexpr bool operator!=(const Matrix2 &other) const noexcept
    {
        return !(*this == other);
    }

    [[nodiscard]] constexpr Matrix2 operator+(const Matrix2 &other) const noexcept
    {
        return Matrix2{m[0][0] + other.m[0][0],
                       m[0][1] + other.m[0][1],
                       m[1][0] + other.m[1][0],
                       m[1][1] + other.m[1][1]};
    }

    [[nodiscard]] constexpr Matrix2 operator-(const Matrix2 &other) const noexcept
    {
        return Matrix2{m[0][0] - other.m[0][0],
                       m[0][1] - other.m[0][1],
                       m[1][0] - other.m[1][0],
                       m[1][1] - other.m[1][1]};
    }

    [[nodiscard]] constexpr Matrix2 operator*(double scalar) const noexcept
    {
        return Matrix2{m[0][0] * scalar, m[0][1] * scalar, m[1][0] * scalar, m[1][1] * scalar};
    }

    [[nodiscard]] constexpr Matrix2 operator*(const Matrix2 &other) const noexcept
    {
        return Matrix2{m[0][0] * other.m[0][0] + m[0][1] * other.m[1][0],
                       m[0][0] * other.m[0][1] + m[0][1] * other.m[1][1],
                       m[1][0] * other.m[0][0] + m[1][1] * other.m[1][0],
                       m[1][0] * other.m[0][1] + m[1][1] * other.m[1][1]};
    }

    [[nodiscard]] constexpr Matrix2 operator/(double scalar) const noexcept
    {
        return Matrix2{m[0][0] / scalar, m[0][1] / scalar, m[1][0] / scalar, m[1][1] / scalar};
    }

    constexpr Matrix2 &operator+=(const Matrix2 &other) noexcept
    {
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                m[i][j] += other.m[i][j];
        return *this;
    }

    constexpr Matrix2 &operator-=(const Matrix2 &other) noexcept
    {
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                m[i][j] -= other.m[i][j];
        return *this;
    }

    constexpr Matrix2 &operator*=(double scalar) noexcept
    {
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                m[i][j] *= scalar;
        return *this;
    }

    constexpr Matrix2 &operator*=(const Matrix2 &other) noexcept
    {
        *this = *this * other;
        return *this;
    }

    constexpr Matrix2 &operator/=(double scalar) noexcept
    {
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                m[i][j] /= scalar;
        return *this;
    }
};

} // namespace linalg3d
