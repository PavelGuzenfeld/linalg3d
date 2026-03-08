# linalg3d

Header-only constexpr linear algebra library for C++23. Provides type-safe 2D, 3D, and 4D math primitives that evaluate entirely at compile time.

## Features

- **Fully constexpr** — all operations evaluate at compile time via [gcem](https://github.com/kthohr/gcem), with `if consteval` dispatch to `std::` math at runtime for hardware-accelerated performance
- **Type-safe angles** — `Angle<RADIANS>` vs `Angle<DEGREES>` prevents unit-mismatch bugs at the type level
- **2D, 3D, 4D** — `Vector2`/`Matrix2`, `Vector3`/`Matrix3`, `Vector4`/`Matrix4`
- **Quaternions** — multiplication, rotation, SLERP interpolation, Euler angle conversion
- **`std::expected`** — matrix `inverse()` returns `std::expected<Matrix, MatrixError>` instead of silent failure
- **fmt support** — optional `format.hpp` header with `fmt::formatter` specializations for all types
- **Zero dependencies at runtime** — gcem is header-only (compile-time only), fmt is optional

## Usage

```cpp
#include <linalg3d/linalg.hpp>

using namespace linalg3d;

// Compile-time vector math
constexpr Vector3 a{1.0, 2.0, 3.0};
constexpr Vector3 b{4.0, 5.0, 6.0};
constexpr auto c = a.cross(b);
static_assert(c == Vector3{-3.0, 6.0, -3.0});

// Type-safe angles
constexpr Angle<AngleType::DEGREES> heading{90.0};
constexpr auto rad = heading.to_radians();

// Quaternion rotation
constexpr auto q = euler_angles_to_quaternion(EulerAngles{0.0, 0.0, PI / 2.0});
constexpr Vector3 rotated = q * Vector3{1.0, 0.0, 0.0};

// Safe matrix inversion
constexpr Matrix3 m{2.0, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 2.0};
constexpr auto inv = m.inverse();
static_assert(inv.has_value());

// SLERP interpolation
constexpr auto mid = slerp(Quaternion::identity(), q, 0.5);

// 2D and 4D
constexpr Vector2 v2{3.0, 4.0};
static_assert(v2.norm_sq() == 25.0);
constexpr Vector4 v4{1.0, 2.0, 3.0, 4.0};
constexpr auto identity4 = Matrix4::identity();
```

For `fmt::print` support:

```cpp
#include <linalg3d/format.hpp>
fmt::print("{}\n", Vector3{1.0, 2.0, 3.0}); // (1, 2, 3)
fmt::print("{}\n", Quaternion::identity());   // Quaternion(w=1, x=0, y=0, z=0)
```

## Types

| Type | Header | Description |
|------|--------|-------------|
| `Vector2` | `vector2.hpp` | 2D vector with dot/cross(scalar)/norm |
| `Vector3` | `vector3.hpp` | 3D vector with dot/cross/norm |
| `Vector4` | `vector4.hpp` | 4D vector with dot/norm |
| `Matrix2` | `matrix2x2.hpp` | 2x2 matrix with inverse/determinant |
| `Matrix3` | `matrix3x3.hpp` | 3x3 matrix with inverse/determinant |
| `Matrix4` | `matrix4x4.hpp` | 4x4 matrix with inverse/determinant |
| `Quaternion` | `quaternion.hpp` | Unit quaternion with rotation/SLERP |
| `Angle<T>` | `angle.hpp` | Type-safe angle (radians/degrees) |
| `EulerAngles<T>` | `euler_angles.hpp` | Pitch/yaw/roll triplet |

## Benchmarks

Measured with [nanobench](https://github.com/martinus/nanobench) (GCC 14, `-O2`, Ubuntu 24.04). Compared against [Eigen 3.4](https://eigen.tuxfamily.org/) on equivalent operations:

| Operation | linalg3d (ns) | Eigen (ns) | Notes |
|---|---|---|---|
| Vector3 dot | 0.70 | 0.51 | |
| Vector3 cross | 0.89 | 0.95 | |
| Vector3 norm | 1.39 | 1.38 | |
| Vector3 normalized | 3.23 | 3.22 | |
| Matrix3 multiply | 4.05 | 3.03 | |
| Matrix3 inverse | 12.65 | 4.51 | Eigen uses SIMD specialization |
| Matrix4 inverse | 37.92 | 11.03 | Eigen uses SIMD specialization |
| Matrix4 multiply | 6.59 | 5.84 | |
| Quaternion multiply | 2.37 | 1.88 | |
| Quaternion*Vector3 | 6.01 | 2.68 | |
| slerp | 19.54 | 20.96 | |
| quaternion_to_euler | 16.11 | 25.88 | |
| Angle::sin | 3.92 | — | hardware fsin via `if consteval` |
| Angle::cos | 3.01 | — | hardware fcos via `if consteval` |

linalg3d trades SIMD-optimized matrix operations for compile-time evaluation (`constexpr` everywhere) and type safety. Trig-heavy operations (slerp, euler conversion) use `if consteval` to dispatch to `std::` math at runtime, matching hardware performance.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target linalg3d_bench
./build/linalg3d_bench
```

## Build

Requires CMake 3.25+ and a C++23 compiler. Dependencies (gcem, fmt, doctest, nanobench) are fetched automatically via FetchContent if not found on the system.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build
```

## License

MIT
