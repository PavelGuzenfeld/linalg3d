#pragma once
#include "constexpr_math.hpp"
#include <algorithm> // for clamp

namespace linalg3d
{
    enum class AngleType
    {
        RADIANS,
        DEGREES
    };

    template <AngleType T = AngleType::RADIANS>
    class Angle
    {
    public:
        static constexpr double MAX_DEGREES = 1.0e308;                    // ~ max double value safely clamped
        static constexpr double MAX_RADIANS = MAX_DEGREES * (PI / 180.0); // converted once

        explicit constexpr Angle(double value = 0.0) noexcept : value_{sanitize(value)} {}

        [[nodiscard]] constexpr double value() const noexcept
        {
            return value_;
        }

        [[nodiscard]] constexpr double sin() const noexcept
        {
            if constexpr (T == AngleType::RADIANS)
            {
                return gcem::sin(value_);
            }
            else
            {
                return gcem::sin(value_ * PI / 180.0);
            }
        }

        [[nodiscard]] constexpr double cos() const noexcept
        {
            if constexpr (T == AngleType::RADIANS)
            {
                return gcem::cos(value_);
            }
            else
            {
                return gcem::cos(value_ * PI / 180.0);
            }
        }

        [[nodiscard]] constexpr double tan() const noexcept
        {
            if constexpr (T == AngleType::RADIANS)
            {
                return gcem::tan(value_);
            }
            else
            {
                return gcem::tan(value_ * PI / 180.0);
            }
        }

        [[nodiscard]] constexpr Angle<AngleType::RADIANS> to_radians() const noexcept
        {
            if constexpr (T == AngleType::RADIANS)
            {
                return Angle<AngleType::RADIANS>{value_};
            }
            else
            {
                return Angle<AngleType::RADIANS>{value_ * PI / 180.0};
            }
        }

        [[nodiscard]] constexpr Angle<AngleType::DEGREES> to_degrees() const noexcept
        {
            if constexpr (T == AngleType::DEGREES)
            {
                return Angle<AngleType::DEGREES>{value_};
            }
            else
            {
                return Angle<AngleType::DEGREES>{value_ * 180.0 / PI};
            }
        }

        [[nodiscard]] constexpr operator Angle<AngleType::RADIANS>() const noexcept
        {
            return to_radians();
        }

        [[nodiscard]] constexpr operator Angle<AngleType::DEGREES>() const noexcept
        {
            return to_degrees();
        }

        [[nodiscard]] static constexpr Angle from_radians(double value) noexcept
        {
            if constexpr (T == AngleType::RADIANS)
            {
                return Angle{value};
            }
            else
            {
                return Angle{value * 180.0 / PI};
            }
        }

        [[nodiscard]] static constexpr Angle from_degrees(double value) noexcept
        {
            if constexpr (T == AngleType::DEGREES)
            {
                return Angle{value};
            }
            else
            {
                return Angle{value * PI / 180.0};
            }
        }

        [[nodiscard]] constexpr Angle operator-() const noexcept
        {
            return Angle{-value_};
        }

    private:
        [[nodiscard]] static constexpr double sanitize(double value) noexcept
        {
            if (is_nan(value) || is_inf(value))
            {
                return 0.0; // fail-safe fallback
            }

            if constexpr (T == AngleType::RADIANS)
            {
                return std::clamp(value, -MAX_RADIANS, MAX_RADIANS);
            }
            else
            {
                return std::clamp(value, -MAX_DEGREES, MAX_DEGREES);
            }
        }
        
        double value_{};
    };

    template <AngleType T>
    [[nodiscard]] constexpr Angle<T> operator+(const Angle<T> &lhs, const Angle<T> &rhs) noexcept
    {
        return Angle<T>{lhs.value() + rhs.value()};
    }

    template <AngleType T>
    [[nodiscard]] constexpr Angle<T> operator-(const Angle<T> &lhs, const Angle<T> &rhs) noexcept
    {
        return Angle<T>{lhs.value() - rhs.value()};
    }

    template <AngleType T>
    [[nodiscard]] constexpr double operator*(const Angle<T> &lhs, double scalar) noexcept
    {
        return lhs.value() * scalar;
    }

    template <AngleType T>
    [[nodiscard]] constexpr double operator/(const Angle<T> &lhs, double scalar) noexcept
    {
        return lhs.value() / scalar;
    }

    template <AngleType T>
    [[nodiscard]] constexpr bool operator==(const Angle<T> &lhs, double rhs) noexcept
    {
        return lhs.value() == rhs;
    }

    template <AngleType T>
    [[nodiscard]] constexpr bool operator!=(const Angle<T> &lhs, double rhs) noexcept
    {
        return !(lhs == rhs);
    }

    template <AngleType T>
    [[nodiscard]] constexpr bool operator<(const Angle<T> &lhs, double rhs) noexcept
    {
        return lhs.value() < rhs;
    }

    template <AngleType T>
    [[nodiscard]] constexpr bool operator<=(const Angle<T> &lhs, double rhs) noexcept
    {
        return lhs.value() <= rhs;
    }

    template <AngleType T>
    [[nodiscard]] constexpr bool operator>(const Angle<T> &lhs, double rhs) noexcept
    {
        return lhs.value() > rhs;
    }

    template <AngleType T>
    [[nodiscard]] constexpr bool operator>=(const Angle<T> &lhs, double rhs) noexcept
    {
        return lhs.value() >= rhs;
    }

} // namespace linalg3d