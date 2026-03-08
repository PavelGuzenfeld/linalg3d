#pragma once
#include "euler_angles.hpp"
#include "matrix2x2.hpp"
#include "matrix3x3.hpp"
#include "matrix4x4.hpp"
#include "quaternion.hpp"
#include "vector2.hpp"
#include "vector3.hpp"
#include "vector4.hpp"
#include <fmt/format.h>

template <>
struct fmt::formatter<linalg3d::Vector2> : fmt::formatter<double>
{
    auto format(const linalg3d::Vector2 &v, fmt::format_context &ctx) const
    {
        return fmt::format_to(ctx.out(), "({}, {})", v.x, v.y);
    }
};

template <>
struct fmt::formatter<linalg3d::Vector3> : fmt::formatter<double>
{
    auto format(const linalg3d::Vector3 &v, fmt::format_context &ctx) const
    {
        return fmt::format_to(ctx.out(), "({}, {}, {})", v.x, v.y, v.z);
    }
};

template <>
struct fmt::formatter<linalg3d::Vector4> : fmt::formatter<double>
{
    auto format(const linalg3d::Vector4 &v, fmt::format_context &ctx) const
    {
        return fmt::format_to(ctx.out(), "({}, {}, {}, {})", v.x, v.y, v.z, v.w);
    }
};

template <>
struct fmt::formatter<linalg3d::Quaternion> : fmt::formatter<double>
{
    auto format(const linalg3d::Quaternion &q, fmt::format_context &ctx) const
    {
        return fmt::format_to(ctx.out(), "Quaternion(w={}, x={}, y={}, z={})", q.w, q.x, q.y, q.z);
    }
};

template <>
struct fmt::formatter<linalg3d::Matrix2> : fmt::formatter<double>
{
    auto format(const linalg3d::Matrix2 &m, fmt::format_context &ctx) const
    {
        return fmt::format_to(ctx.out(), "[[{}, {}], [{}, {}]]", m.m[0][0], m.m[0][1], m.m[1][0], m.m[1][1]);
    }
};

template <>
struct fmt::formatter<linalg3d::Matrix3> : fmt::formatter<double>
{
    auto format(const linalg3d::Matrix3 &m, fmt::format_context &ctx) const
    {
        return fmt::format_to(ctx.out(),
                              "[[{}, {}, {}], [{}, {}, {}], [{}, {}, {}]]",
                              m.m[0][0],
                              m.m[0][1],
                              m.m[0][2],
                              m.m[1][0],
                              m.m[1][1],
                              m.m[1][2],
                              m.m[2][0],
                              m.m[2][1],
                              m.m[2][2]);
    }
};

template <>
struct fmt::formatter<linalg3d::Matrix4> : fmt::formatter<double>
{
    auto format(const linalg3d::Matrix4 &m, fmt::format_context &ctx) const
    {
        return fmt::format_to(ctx.out(),
                              "[[{}, {}, {}, {}], [{}, {}, {}, {}], [{}, {}, {}, {}], [{}, {}, {}, {}]]",
                              m.m[0][0],
                              m.m[0][1],
                              m.m[0][2],
                              m.m[0][3],
                              m.m[1][0],
                              m.m[1][1],
                              m.m[1][2],
                              m.m[1][3],
                              m.m[2][0],
                              m.m[2][1],
                              m.m[2][2],
                              m.m[2][3],
                              m.m[3][0],
                              m.m[3][1],
                              m.m[3][2],
                              m.m[3][3]);
    }
};

template <linalg3d::AngleType T>
struct fmt::formatter<linalg3d::Angle<T>> : fmt::formatter<double>
{
    auto format(const linalg3d::Angle<T> &a, fmt::format_context &ctx) const
    {
        if constexpr (T == linalg3d::AngleType::RADIANS)
            return fmt::format_to(ctx.out(), "{} rad", a.value());
        else
            return fmt::format_to(ctx.out(), "{} deg", a.value());
    }
};

template <linalg3d::AngleType T>
struct fmt::formatter<linalg3d::EulerAngles<T>> : fmt::formatter<double>
{
    auto format(const linalg3d::EulerAngles<T> &e, fmt::format_context &ctx) const
    {
        return fmt::format_to(ctx.out(),
                              "EulerAngles(pitch={}, yaw={}, roll={})",
                              e.pitch.value(),
                              e.yaw.value(),
                              e.roll.value());
    }
};
