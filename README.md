# linalg3d

Header-only constexpr linear algebra library for C++23. Provides type-safe 2D, 3D, and 4D math primitives that evaluate entirely at compile time.

## Features

- **Fully constexpr** — all operations evaluate at compile time via [gcem](https://github.com/kthohr/gcem), with `if consteval` dispatch to `std::` math and SIMD at runtime
- **SIMD optimized** — SSE2/AVX/FMA (x86) and NEON (ARM) intrinsics for matrix multiply/inverse via automatic `if consteval` dispatch
- **Python bindings** — [nanobind](https://github.com/wjakob/nanobind)-based package with zero-copy NumPy and SciPy interop
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

// Angle between two vectors (radians)
constexpr auto theta = angle_between(Vector3{1.0, 0.0, 0.0}, Vector3{0.0, 1.0, 0.0});
static_assert(theta > 1.57 && theta < 1.58); // pi/2

// Angle between two quaternion orientations (shortest arc, radians)
auto rot_angle = angle_between(Quaternion::identity(), q);

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
| `Quaternion` | `quaternion.hpp` | Unit quaternion with rotation/SLERP/angle |
| `Angle<T>` | `angle.hpp` | Type-safe angle (radians/degrees) |
| `EulerAngles<T>` | `euler_angles.hpp` | Pitch/yaw/roll triplet |

## Benchmarks

Measured with [nanobench](https://github.com/martinus/nanobench) (GCC 14, `-O3 -march=native`, Ubuntu 24.04). Compared against [Eigen 3.4](https://eigen.tuxfamily.org/) on equivalent operations:

| Operation | linalg3d (ns) | Eigen (ns) | Ratio |
|---|---|---|---|
| Vector3 dot | 0.76 | 0.54 | 1.4x |
| Vector3 cross | 0.80 | 0.96 | **0.8x** |
| Vector3 norm | 1.49 | 1.49 | **1.0x** |
| Vector3 normalized | 3.50 | 3.48 | 1.0x |
| Matrix3 multiply | 3.60 | 2.64 | 1.4x |
| Matrix3 inverse | 7.11 | 7.23 | **0.98x** |
| Matrix4 inverse | 14.78 | 10.72 | 1.4x |
| Matrix4 multiply | 2.90 | 2.50 | 1.2x |
| Quaternion multiply | 1.75 | 1.21 | 1.4x |
| Quaternion inverse | 2.02 | 8.77 | **0.23x** |
| Quaternion*Vector3 | 2.01 | 2.42 | **0.8x** |
| slerp | 21.16 | 22.12 | **0.96x** |
| quaternion_to_euler | 18.80 | 26.67 | **0.7x** |
| Angle::sin | 4.22 | — | — |
| Angle::cos | 3.23 | — | — |

Ratio = linalg3d / Eigen (lower is better for linalg3d; **bold** = linalg3d wins).

### Performance summary vs Eigen

- **linalg3d faster:** Matrix3 inverse (0.98x), quaternion inverse (4.3x faster), cross product (0.8x), quaternion*vector (0.8x), slerp (0.96x), quaternion-to-euler (0.7x)
- **On par (0.9-1.2x):** vector norm, normalized, add, matrix transpose, matrix4 multiply
- **Eigen faster:** matrix3 multiply (1.4x), matrix4 inverse (1.4x), quaternion multiply (1.4x), dot product (1.4x)
- **Unique to linalg3d:** all operations are `constexpr` (Eigen has none), type-safe angles, `std::expected` error handling, SIMD dispatched via `if consteval`

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target linalg3d_bench
./build/linalg3d_bench
```

## Examples

### IMU attitude interpolation

Smoothly interpolate between two IMU quaternion readings for sub-frame timing.

```cpp
#include <linalg3d/linalg.hpp>
using namespace linalg3d;

// Two IMU samples at different timestamps
const Quaternion q_t0{0.998, 0.01, 0.02, 0.05};
const Quaternion q_t1{0.997, 0.01, 0.03, 0.06};

// Interpolate at 60% between the two samples
auto q_mid = slerp(q_t0, q_t1, 0.6);

// How far apart are these orientations?
double delta = angle_between(q_t0, q_t1); // radians
```

### Rotating a body-frame vector to NED

Convert a sensor reading from body frame to North-East-Down using an attitude quaternion.

```cpp
// Sensor reports acceleration in body frame
constexpr Vector3 accel_body{0.1, -0.05, -9.81};

// Current attitude from IMU (quaternion: w, x, y, z)
const Quaternion attitude{0.998, 0.01, 0.02, 0.05};

// Rotate to NED frame
Vector3 accel_ned = attitude * accel_body;
```

### Compile-time coordinate transform validation

Verify a rotation matrix at compile time -- errors caught before the code even runs.

```cpp
constexpr auto q = euler_angles_to_quaternion(EulerAngles{0.0, 0.0, PI / 2.0});
constexpr auto mat = quaternion_to_matrix(q);
constexpr auto v = mat * Vector3{1.0, 0.0, 0.0};
// 90 deg yaw: x-axis maps to y-axis
static_assert(fabs(v.x) < 1e-10);
static_assert(fabs(v.y - 1.0) < 1e-10);
```

## Python

### Installation

```bash
pip install linalg3d
```

Or build from source:

```bash
pip install "./python[test]"
```

### Python usage

```python
from linalg3d import Vector3, Matrix3, Quaternion, slerp
import numpy as np

# Vectors
v = Vector3(1.0, 2.0, 3.0)
print(v.norm(), v.normalized())

# Zero-copy NumPy interop — no data copying
arr = np.asarray(v.numpy())     # shares memory with v
arr[0] = 99.0
assert v.x == 99.0              # modification reflected

# Matrices
m = Matrix3.identity()
arr = np.asarray(m.numpy())     # zero-copy 3x3 view
det = m.determinant()
inv = m.inverse()               # returns None if singular

# Quaternion rotation
import math
q = Quaternion(math.cos(math.pi/4), 0, 0, math.sin(math.pi/4))
rotated = q.rotate(Vector3(1, 0, 0))

# SLERP
mid = slerp(Quaternion(), q, 0.5)

# SciPy interop (convention: [x, y, z, w])
from scipy.spatial.transform import Rotation
scipy_arr = np.asarray(q.to_scipy())   # [x,y,z,w]
rot = Rotation.from_quat(scipy_arr)
q2 = Quaternion.from_scipy(scipy_arr)  # back to linalg3d
```

## Build (C++)

Requires CMake 3.25+ and a C++23 compiler. Dependencies (gcem, fmt, doctest, nanobench) are fetched automatically via FetchContent if not found on the system.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build
```

## License

MIT
