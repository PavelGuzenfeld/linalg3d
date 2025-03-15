#pragma once
namespace linalg3d
{
    class Matrix3x3
    {
    public:
        double m[3][3]{};

        constexpr Matrix3x3() noexcept = default;

        constexpr Matrix3x3(double m00, double m01, double m02,
                            double m10, double m11, double m12,
                            double m20, double m21, double m22) noexcept
            : m{{m00, m01, m02}, {m10, m11, m12}, {m20, m21, m22}} {}

        [[nodiscard]] constexpr Matrix3x3 transpose() const noexcept
        {
            return Matrix3x3{
                m[0][0], m[1][0], m[2][0],
                m[0][1], m[1][1], m[2][1],
                m[0][2], m[1][2], m[2][2]};
        }

        [[nodiscard]] constexpr bool operator==(const Matrix3x3 &other) const noexcept
        {
            for (int i = 0; i < 3; ++i)
            {
                for (int j = 0; j < 3; ++j)
                {
                    if (m[i][j] != other.m[i][j])
                    {
                        return false;
                    }
                }
            }
            return true;
        }

        [[nodiscard]] bool constexpr operator!=(const Matrix3x3 &other) const noexcept
        {
            return !(*this == other);
        }

        [[nodiscard]] constexpr Matrix3x3 operator+(const Matrix3x3 &other) const noexcept
        {
            return Matrix3x3{
                m[0][0] + other.m[0][0], m[0][1] + other.m[0][1], m[0][2] + other.m[0][2],
                m[1][0] + other.m[1][0], m[1][1] + other.m[1][1], m[1][2] + other.m[1][2],
                m[2][0] + other.m[2][0], m[2][1] + other.m[2][1], m[2][2] + other.m[2][2]};
        }

        [[nodiscard]] constexpr Matrix3x3 operator-(const Matrix3x3 &other) const noexcept
        {
            return Matrix3x3{
                m[0][0] - other.m[0][0], m[0][1] - other.m[0][1], m[0][2] - other.m[0][2],
                m[1][0] - other.m[1][0], m[1][1] - other.m[1][1], m[1][2] - other.m[1][2],
                m[2][0] - other.m[2][0], m[2][1] - other.m[2][1], m[2][2] - other.m[2][2]};
        }

        [[nodiscard]] constexpr Matrix3x3 operator*(double scalar) const noexcept
        {
            return Matrix3x3{
                m[0][0] * scalar, m[0][1] * scalar, m[0][2] * scalar,
                m[1][0] * scalar, m[1][1] * scalar, m[1][2] * scalar,
                m[2][0] * scalar, m[2][1] * scalar, m[2][2] * scalar};
        }

        [[nodiscard]] constexpr Matrix3x3 operator/(double scalar) const noexcept
        {
            return Matrix3x3{
                m[0][0] / scalar, m[0][1] / scalar, m[0][2] / scalar,
                m[1][0] / scalar, m[1][1] / scalar, m[1][2] / scalar,
                m[2][0] / scalar, m[2][1] / scalar, m[2][2] / scalar};
        }
    };

} // namespace linalg3d