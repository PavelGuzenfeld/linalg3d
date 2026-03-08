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
        : pitch{pitch_a}, yaw{yaw_a}, roll{roll_a}
    {
    }

    constexpr EulerAngles(double pitch_a, double yaw_a, double roll_a) noexcept
        : pitch{Angle<T>::from_radians(pitch_a)}, yaw{Angle<T>::from_radians(yaw_a)},
          roll{Angle<T>::from_radians(roll_a)}
    {
    }

    // Construct from any strong type with .get() (interprets values as radians)
    template <typename S>
        requires requires(S s) {
            { s.get() } -> std::convertible_to<double>;
        } && (!std::same_as<std::remove_cvref_t<S>, double>) && (!std::same_as<std::remove_cvref_t<S>, Angle<T>>)
    constexpr EulerAngles(S pitch_a, S yaw_a, S roll_a) noexcept
        : pitch{Angle<T>::from_radians(static_cast<double>(pitch_a.get()))},
          yaw{Angle<T>::from_radians(static_cast<double>(yaw_a.get()))},
          roll{Angle<T>::from_radians(static_cast<double>(roll_a.get()))}
    {
    }
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
