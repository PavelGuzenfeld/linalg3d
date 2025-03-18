#pragma once
#include "gcem.hpp"

namespace linalg3d
{
    constexpr double PI = static_cast<double>(GCEM_PI);
    constexpr double HALF_PI = static_cast<double>(GCEM_HALF_PI);

    constexpr bool fabs(double x)
    {
        return x < 0.0 ? -x : x;
    }

    constexpr bool is_nan(double x)
    {
        return x != x;
    }

    constexpr bool is_inf(double x)
    {
        return x == std::numeric_limits<double>::infinity() ||
               x == -std::numeric_limits<double>::infinity();
    }

} // namespace linalg3d
