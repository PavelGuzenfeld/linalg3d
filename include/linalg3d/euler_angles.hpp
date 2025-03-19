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
        constexpr EulerAngles<AngleType::RADIANS> to_radians() const noexcept
        {
            if constexpr (T == AngleType::RADIANS)
            {
                return *this;
            }
            else
            {
                return EulerAngles<AngleType::RADIANS>{pitch.to_radians(), yaw.to_radians(), roll.to_radians()};
            }
        }
        constexpr EulerAngles<AngleType::DEGREES> to_degrees() const noexcept
        {
            if constexpr (T == AngleType::DEGREES)
            {
                return *this;
            }
            else
            {
                return EulerAngles<AngleType::DEGREES>{pitch.to_degrees(), yaw.to_degrees(), roll.to_degrees()};
            }
        }
    };
} // namespace linalg3d
