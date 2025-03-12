#pragma once
#include <cmath>

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
    };

} // namespace linalg3d