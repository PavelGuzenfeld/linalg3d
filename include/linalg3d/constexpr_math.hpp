#pragma once
#include <cmath>
#include <compare>
#include <gcem.hpp>
#include <limits>

namespace linalg3d
{
constexpr double PI = static_cast<double>(GCEM_PI);
constexpr double HALF_PI = static_cast<double>(GCEM_HALF_PI);

// consteval-dispatch: gcem at compile time, std at runtime
[[nodiscard]] constexpr double ce_sqrt(double x) noexcept
{
    if consteval
    {
        return gcem::sqrt(x);
    }
    else
    {
        return std::sqrt(x);
    }
}

[[nodiscard]] constexpr double ce_sin(double x) noexcept
{
    if consteval
    {
        return gcem::sin(x);
    }
    else
    {
        return std::sin(x);
    }
}

[[nodiscard]] constexpr double ce_cos(double x) noexcept
{
    if consteval
    {
        return gcem::cos(x);
    }
    else
    {
        return std::cos(x);
    }
}

[[nodiscard]] constexpr double ce_tan(double x) noexcept
{
    if consteval
    {
        return gcem::tan(x);
    }
    else
    {
        return std::tan(x);
    }
}

[[nodiscard]] constexpr double ce_asin(double x) noexcept
{
    if consteval
    {
        return gcem::asin(x);
    }
    else
    {
        return std::asin(x);
    }
}

[[nodiscard]] constexpr double ce_acos(double x) noexcept
{
    if consteval
    {
        return gcem::acos(x);
    }
    else
    {
        return std::acos(x);
    }
}

[[nodiscard]] constexpr double ce_atan2(double y, double x) noexcept
{
    if consteval
    {
        return gcem::atan2(y, x);
    }
    else
    {
        return std::atan2(y, x);
    }
}

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
