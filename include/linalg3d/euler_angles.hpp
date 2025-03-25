#pragma once
#include "angle.hpp"

namespace linalg3d
{
    template <AngleType T = AngleType::RADIANS>
    struct EulerAngles
    {
        Angle<T> pitch{}, yaw{}, roll{};

        constexpr EulerAngles() noexcept = default;
        constexpr EulerAngles(Angle<T> pitch_a, Angle<T> yaw_a, Angle<T> roll_a) noexcept
            : pitch{pitch_a}, yaw{yaw_a}, roll{roll_a} {}

        constexpr EulerAngles(double pitch_a, double yaw_a, double roll_a) noexcept
            : pitch{Angle<T>::from_radians(pitch_a)},
              yaw{Angle<T>::from_radians(yaw_a)},
              roll{Angle<T>::from_radians(roll_a)} {}
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
