#pragma once
#include "angle.hpp"

namespace linalg3d
{
    template <AngleType T = AngleType::RADIANS>
    struct EulerAngles
    {
        Angle<T> pitch{}, yaw{}, roll{};

        constexpr EulerAngles() noexcept = default;
        constexpr EulerAngles(Angle<T> pitch, Angle<T> yaw, Angle<T> roll) noexcept
            : pitch{pitch}, yaw{yaw}, roll{roll} {}

        constexpr EulerAngles(double pitch, double yaw, double roll) noexcept
            : pitch{Angle<T>::from_radians(pitch)},
              yaw{Angle<T>::from_radians(yaw)},
              roll{Angle<T>::from_radians(roll)} {}
    };
} // namespace linalg3d