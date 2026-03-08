#pragma once
#include "constexpr_math.hpp"
#include <expected>

namespace linalg3d
{
    class Matrix4
    {
    public:
        double m[4][4]{};

        constexpr Matrix4() noexcept = default;

        constexpr Matrix4(double m00, double m01, double m02, double m03,
                          double m10, double m11, double m12, double m13,
                          double m20, double m21, double m22, double m23,
                          double m30, double m31, double m32, double m33) noexcept
            : m{{m00, m01, m02, m03},
                {m10, m11, m12, m13},
                {m20, m21, m22, m23},
                {m30, m31, m32, m33}} {}

        [[nodiscard]] static constexpr Matrix4 identity() noexcept
        {
            return Matrix4{
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0};
        }

        [[nodiscard]] constexpr Matrix4 transpose() const noexcept
        {
            return Matrix4{
                m[0][0], m[1][0], m[2][0], m[3][0],
                m[0][1], m[1][1], m[2][1], m[3][1],
                m[0][2], m[1][2], m[2][2], m[3][2],
                m[0][3], m[1][3], m[2][3], m[3][3]};
        }

        [[nodiscard]] constexpr double determinant() const noexcept
        {
            double s0 = m[2][0] * m[3][1] - m[2][1] * m[3][0];
            double s1 = m[2][0] * m[3][2] - m[2][2] * m[3][0];
            double s2 = m[2][0] * m[3][3] - m[2][3] * m[3][0];
            double s3 = m[2][1] * m[3][2] - m[2][2] * m[3][1];
            double s4 = m[2][1] * m[3][3] - m[2][3] * m[3][1];
            double s5 = m[2][2] * m[3][3] - m[2][3] * m[3][2];

            double c0 = m[0][0] * m[1][1] - m[0][1] * m[1][0];
            double c1 = m[0][0] * m[1][2] - m[0][2] * m[1][0];
            double c2 = m[0][0] * m[1][3] - m[0][3] * m[1][0];
            double c3 = m[0][1] * m[1][2] - m[0][2] * m[1][1];
            double c4 = m[0][1] * m[1][3] - m[0][3] * m[1][1];
            double c5 = m[0][2] * m[1][3] - m[0][3] * m[1][2];

            return c0 * s5 - c1 * s4 + c2 * s3 + c3 * s2 - c4 * s1 + c5 * s0;
        }

        [[nodiscard]] constexpr std::expected<Matrix4, MatrixError> inverse() const noexcept
        {
            double s0 = m[2][0] * m[3][1] - m[2][1] * m[3][0];
            double s1 = m[2][0] * m[3][2] - m[2][2] * m[3][0];
            double s2 = m[2][0] * m[3][3] - m[2][3] * m[3][0];
            double s3 = m[2][1] * m[3][2] - m[2][2] * m[3][1];
            double s4 = m[2][1] * m[3][3] - m[2][3] * m[3][1];
            double s5 = m[2][2] * m[3][3] - m[2][3] * m[3][2];

            double c0 = m[0][0] * m[1][1] - m[0][1] * m[1][0];
            double c1 = m[0][0] * m[1][2] - m[0][2] * m[1][0];
            double c2 = m[0][0] * m[1][3] - m[0][3] * m[1][0];
            double c3 = m[0][1] * m[1][2] - m[0][2] * m[1][1];
            double c4 = m[0][1] * m[1][3] - m[0][3] * m[1][1];
            double c5 = m[0][2] * m[1][3] - m[0][3] * m[1][2];

            double det = c0 * s5 - c1 * s4 + c2 * s3 + c3 * s2 - c4 * s1 + c5 * s0;
            if (det == 0.0)
                return std::unexpected{MatrixError::singular};

            double id = 1.0 / det;
            return Matrix4{
                (m[1][1] * s5 - m[1][2] * s4 + m[1][3] * s3) * id,
                (-m[0][1] * s5 + m[0][2] * s4 - m[0][3] * s3) * id,
                (m[3][1] * c5 - m[3][2] * c4 + m[3][3] * c3) * id,
                (-m[2][1] * c5 + m[2][2] * c4 - m[2][3] * c3) * id,

                (-m[1][0] * s5 + m[1][2] * s2 - m[1][3] * s1) * id,
                (m[0][0] * s5 - m[0][2] * s2 + m[0][3] * s1) * id,
                (-m[3][0] * c5 + m[3][2] * c2 - m[3][3] * c1) * id,
                (m[2][0] * c5 - m[2][2] * c2 + m[2][3] * c1) * id,

                (m[1][0] * s4 - m[1][1] * s2 + m[1][3] * s0) * id,
                (-m[0][0] * s4 + m[0][1] * s2 - m[0][3] * s0) * id,
                (m[3][0] * c4 - m[3][1] * c2 + m[3][3] * c0) * id,
                (-m[2][0] * c4 + m[2][1] * c2 - m[2][3] * c0) * id,

                (-m[1][0] * s3 + m[1][1] * s1 - m[1][2] * s0) * id,
                (m[0][0] * s3 - m[0][1] * s1 + m[0][2] * s0) * id,
                (-m[3][0] * c3 + m[3][1] * c1 - m[3][2] * c0) * id,
                (m[2][0] * c3 - m[2][1] * c1 + m[2][2] * c0) * id};
        }

        [[nodiscard]] constexpr bool operator==(const Matrix4 &other) const noexcept
        {
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    if (m[i][j] != other.m[i][j])
                        return false;
            return true;
        }

        [[nodiscard]] constexpr bool operator!=(const Matrix4 &other) const noexcept
        {
            return !(*this == other);
        }

        [[nodiscard]] constexpr Matrix4 operator+(const Matrix4 &other) const noexcept
        {
            Matrix4 r;
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    r.m[i][j] = m[i][j] + other.m[i][j];
            return r;
        }

        [[nodiscard]] constexpr Matrix4 operator-(const Matrix4 &other) const noexcept
        {
            Matrix4 r;
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    r.m[i][j] = m[i][j] - other.m[i][j];
            return r;
        }

        [[nodiscard]] constexpr Matrix4 operator*(double scalar) const noexcept
        {
            Matrix4 r;
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    r.m[i][j] = m[i][j] * scalar;
            return r;
        }

        [[nodiscard]] constexpr Matrix4 operator*(const Matrix4 &other) const noexcept
        {
            Matrix4 r;
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    for (int k = 0; k < 4; ++k)
                        r.m[i][j] += m[i][k] * other.m[k][j];
            return r;
        }

        [[nodiscard]] constexpr Matrix4 operator/(double scalar) const noexcept
        {
            Matrix4 r;
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    r.m[i][j] = m[i][j] / scalar;
            return r;
        }

        constexpr Matrix4 &operator+=(const Matrix4 &other) noexcept
        {
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    m[i][j] += other.m[i][j];
            return *this;
        }

        constexpr Matrix4 &operator-=(const Matrix4 &other) noexcept
        {
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    m[i][j] -= other.m[i][j];
            return *this;
        }

        constexpr Matrix4 &operator*=(double scalar) noexcept
        {
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    m[i][j] *= scalar;
            return *this;
        }

        constexpr Matrix4 &operator*=(const Matrix4 &other) noexcept
        {
            *this = *this * other;
            return *this;
        }

        constexpr Matrix4 &operator/=(double scalar) noexcept
        {
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    m[i][j] /= scalar;
            return *this;
        }
    };

} // namespace linalg3d
