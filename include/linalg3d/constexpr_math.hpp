#pragma once
#include <compare>
#include <gcem.hpp>
#include <limits>

namespace linalg3d
{
constexpr double PI = static_cast<double>(GCEM_PI);
constexpr double HALF_PI = static_cast<double>(GCEM_HALF_PI);

[[nodiscard]] constexpr double fabs(double x) noexcept
{
    return x < 0.0 ? -x : x;
}

[[nodiscard]] constexpr bool is_nan(double x) noexcept
{
    return x != x;
}

[[nodiscard]] constexpr bool is_inf(double x) noexcept
{
    return x == std::numeric_limits<double>::infinity() || x == -std::numeric_limits<double>::infinity();
}

[[nodiscard]] constexpr double clamp(double value, double lo, double hi) noexcept
{
    return value < lo ? lo : (value > hi ? hi : value);
}

[[nodiscard]] constexpr std::weak_ordering compare_double(double a, double b) noexcept
{
    if (is_nan(a) && is_nan(b))
        return std::weak_ordering::equivalent;
    if (is_nan(a))
        return std::weak_ordering::greater;
    if (is_nan(b))
        return std::weak_ordering::less;
    if (a < b)
        return std::weak_ordering::less;
    if (a > b)
        return std::weak_ordering::greater;
    return std::weak_ordering::equivalent;
}

enum class MatrixError
{
    singular
};

} // namespace linalg3d
