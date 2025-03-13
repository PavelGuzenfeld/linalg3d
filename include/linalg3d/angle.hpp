#pragma once
#include <cmath>

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
        explicit constexpr Angle(double value = 0.0) noexcept : value_{value} {}

        [[nodiscard]] constexpr double to_radians() const noexcept
        {
            if constexpr (T == AngleType::RADIANS)
            {
                return value_;
            }
            else
            {
                return value_ * M_PI / 180.0;
            }

            return value_;
        }

        [[nodiscard]] constexpr double to_degrees() const noexcept
        {
            if constexpr (T == AngleType::DEGREES)
            {
                return value_;
            }
            else
            {
                return value_ * 180.0 / M_PI;
            }

            return value_;
        }

        [[nodiscard]] constexpr double value() const noexcept
        {
            return value_;
        }

        [[nodiscard]] static constexpr Angle from_radians(double value) noexcept
        {
            if constexpr (T == AngleType::RADIANS)
            {
                return Angle{value};
            }
            else
            {
                return Angle{value * 180.0 / M_PI};
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
                return Angle{value * M_PI / 180.0};
            }
        }

    private:
        double value_{};
    };

    template<AngleType T>
    [[nodiscard]] constexpr Angle<T> operator+(const Angle<T> &lhs, const Angle<T> &rhs) noexcept
    {
        return Angle<T>{lhs.value() + rhs.value()};
    }

    template<AngleType T>
    [[nodiscard]] constexpr Angle<T> operator-(const Angle<T> &lhs, const Angle<T> &rhs) noexcept
    {
        return Angle<T>{lhs.value() - rhs.value()};
    }

    template<AngleType T>
    [[nodiscard]] constexpr double operator*(const Angle<T> &lhs, double scalar) noexcept
    {
        return lhs.value() * scalar;
    }

    template<AngleType T>
    [[nodiscard]] constexpr double operator/(const Angle<T> &lhs, double scalar) noexcept
    {
        return lhs.value() / scalar;
    }

    template<AngleType T>
    [[nodiscard]] constexpr bool operator==(const Angle<T> &lhs, double rhs) noexcept
    {
        return lhs.value() == rhs;
    }

    template<AngleType T>
    [[nodiscard]] constexpr bool operator!=(const Angle<T> &lhs, double rhs) noexcept
    {
        return !(lhs == rhs);
    }

    template<AngleType T>
    [[nodiscard]] constexpr bool operator<(const Angle<T> &lhs, double rhs) noexcept
    {
        return lhs.value() < rhs;
    }

    template<AngleType T>
    [[nodiscard]] constexpr bool operator<=(const Angle<T> &lhs, double rhs) noexcept
    {
        return lhs.value() <= rhs;
    }

    template<AngleType T>
    [[nodiscard]] constexpr bool operator>(const Angle<T> &lhs, double rhs) noexcept
    {
        return lhs.value() > rhs;
    }

    template<AngleType T>
    [[nodiscard]] constexpr bool operator>=(const Angle<T> &lhs, double rhs) noexcept
    {
        return lhs.value() >= rhs;
    }

} // namespace linalg3d