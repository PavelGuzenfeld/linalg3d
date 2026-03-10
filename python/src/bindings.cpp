#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include "linalg3d/linalg.hpp"

namespace nb = nanobind;
using namespace linalg3d;

// Helper: expose buffer protocol for contiguous double storage
template <typename T, size_t N>
void def_vector_buffer(nb::class_<T> &cls)
{
    cls.def("__buffer__",
            [](T &self) {
                return nb::ndarray<nb::numpy, double, nb::shape<N>>(
                    &reinterpret_cast<double &>(self), {N}, nb::handle());
            })
        .def_static(
            "from_numpy",
            [](nb::ndarray<double, nb::shape<N>> arr) {
                T result;
                auto *dst = &reinterpret_cast<double &>(result);
                for (size_t i = 0; i < N; ++i)
                    dst[i] = arr(i);
                return result;
            },
            nb::arg("arr"));
}

NB_MODULE(_core, m)
{
    m.doc() = "linalg3d: fast linear algebra with zero-copy NumPy interop";

    // =========================================================================
    // Vector2
    // =========================================================================
    auto vec2 = nb::class_<Vector2>(m, "Vector2")
                    .def(nb::init<double, double>(), nb::arg("x") = 0.0, nb::arg("y") = 0.0)
                    .def_rw("x", &Vector2::x)
                    .def_rw("y", &Vector2::y)
                    .def("norm", &Vector2::norm)
                    .def("norm_sq", &Vector2::norm_sq)
                    .def("normalized", &Vector2::normalized)
                    .def("dot", &Vector2::dot, nb::arg("other"))
                    .def("cross", &Vector2::cross, nb::arg("other"))
                    .def("abs", &Vector2::abs)
                    .def("__add__", &Vector2::operator+)
                    .def("__sub__", static_cast<Vector2 (Vector2::*)(const Vector2 &) const>(&Vector2::operator-))
                    .def("__neg__", static_cast<Vector2 (Vector2::*)() const>(&Vector2::operator-))
                    .def("__mul__", &Vector2::operator*, nb::arg("scalar"))
                    .def("__truediv__", &Vector2::operator/, nb::arg("scalar"))
                    .def("__eq__", &Vector2::operator==)
                    .def("__repr__", [](const Vector2 &v) {
                        return "Vector2(" + std::to_string(v.x) + ", " + std::to_string(v.y) + ")";
                    });

    // Zero-copy buffer protocol
    vec2.def("numpy", [](nb::handle_t<Vector2> h) {
        auto &self = nb::cast<Vector2 &>(h);
        return nb::ndarray<nb::numpy, double, nb::shape<2>>(&self.x, {2}, h);
    }, "Zero-copy view as NumPy array");
    vec2.def_static("from_numpy", [](nb::ndarray<double, nb::shape<2>> arr) {
        return Vector2{arr(0), arr(1)};
    }, nb::arg("arr"));

    // =========================================================================
    // Vector3
    // =========================================================================
    auto vec3 = nb::class_<Vector3>(m, "Vector3")
                    .def(nb::init<double, double, double>(), nb::arg("x") = 0.0, nb::arg("y") = 0.0, nb::arg("z") = 0.0)
                    .def_rw("x", &Vector3::x)
                    .def_rw("y", &Vector3::y)
                    .def_rw("z", &Vector3::z)
                    .def("norm", &Vector3::norm)
                    .def("norm_sq", &Vector3::norm_sq)
                    .def("normalized", &Vector3::normalized)
                    .def("dot", &Vector3::dot, nb::arg("other"))
                    .def("cross", &Vector3::cross, nb::arg("other"))
                    .def("abs", &Vector3::abs)
                    .def("__add__", &Vector3::operator+)
                    .def("__sub__", static_cast<Vector3 (Vector3::*)(const Vector3 &) const>(&Vector3::operator-))
                    .def("__neg__", static_cast<Vector3 (Vector3::*)() const>(&Vector3::operator-))
                    .def("__mul__", static_cast<Vector3 (Vector3::*)(double) const>(&Vector3::operator*))
                    .def("__truediv__", &Vector3::operator/, nb::arg("scalar"))
                    .def("__eq__", &Vector3::operator==)
                    .def("__repr__", [](const Vector3 &v) {
                        return "Vector3(" + std::to_string(v.x) + ", " + std::to_string(v.y) + ", " +
                               std::to_string(v.z) + ")";
                    });

    vec3.def("numpy", [](nb::handle_t<Vector3> h) {
        auto &self = nb::cast<Vector3 &>(h);
        return nb::ndarray<nb::numpy, double, nb::shape<3>>(&self.x, {3}, h);
    }, "Zero-copy view as NumPy array");
    vec3.def_static("from_numpy", [](nb::ndarray<double, nb::shape<3>> arr) {
        return Vector3{arr(0), arr(1), arr(2)};
    }, nb::arg("arr"));

    // =========================================================================
    // Vector4
    // =========================================================================
    auto vec4 = nb::class_<Vector4>(m, "Vector4")
                    .def(nb::init<double, double, double, double>(),
                         nb::arg("x") = 0.0, nb::arg("y") = 0.0, nb::arg("z") = 0.0, nb::arg("w") = 0.0)
                    .def_rw("x", &Vector4::x)
                    .def_rw("y", &Vector4::y)
                    .def_rw("z", &Vector4::z)
                    .def_rw("w", &Vector4::w)
                    .def("norm", &Vector4::norm)
                    .def("norm_sq", &Vector4::norm_sq)
                    .def("normalized", &Vector4::normalized)
                    .def("dot", &Vector4::dot, nb::arg("other"))
                    .def("abs", &Vector4::abs)
                    .def("__add__", &Vector4::operator+)
                    .def("__sub__", static_cast<Vector4 (Vector4::*)(const Vector4 &) const>(&Vector4::operator-))
                    .def("__neg__", static_cast<Vector4 (Vector4::*)() const>(&Vector4::operator-))
                    .def("__mul__", &Vector4::operator*, nb::arg("scalar"))
                    .def("__truediv__", &Vector4::operator/, nb::arg("scalar"))
                    .def("__eq__", &Vector4::operator==)
                    .def("__repr__", [](const Vector4 &v) {
                        return "Vector4(" + std::to_string(v.x) + ", " + std::to_string(v.y) + ", " +
                               std::to_string(v.z) + ", " + std::to_string(v.w) + ")";
                    });

    vec4.def("numpy", [](nb::handle_t<Vector4> h) {
        auto &self = nb::cast<Vector4 &>(h);
        return nb::ndarray<nb::numpy, double, nb::shape<4>>(&self.x, {4}, h);
    }, "Zero-copy view as NumPy array");
    vec4.def_static("from_numpy", [](nb::ndarray<double, nb::shape<4>> arr) {
        return Vector4{arr(0), arr(1), arr(2), arr(3)};
    }, nb::arg("arr"));

    // =========================================================================
    // Matrix2
    // =========================================================================
    nb::class_<Matrix2>(m, "Matrix2")
        .def(nb::init<double, double, double, double>(),
             nb::arg("m00") = 0.0, nb::arg("m01") = 0.0, nb::arg("m10") = 0.0, nb::arg("m11") = 0.0)
        .def_static("identity", &Matrix2::identity)
        .def("transpose", &Matrix2::transpose)
        .def("determinant", &Matrix2::determinant)
        .def("inverse", [](const Matrix2 &self) -> nb::object {
            auto result = self.inverse();
            if (result.has_value())
                return nb::cast(result.value());
            return nb::none();
        })
        .def("__add__", &Matrix2::operator+)
        .def("__sub__", &Matrix2::operator-)
        .def("__mul__", static_cast<Matrix2 (Matrix2::*)(double) const>(&Matrix2::operator*), nb::arg("scalar"))
        .def("__matmul__", static_cast<Matrix2 (Matrix2::*)(const Matrix2 &) const>(&Matrix2::operator*))
        .def("__truediv__", &Matrix2::operator/, nb::arg("scalar"))
        .def("__eq__", &Matrix2::operator==)
        .def("numpy", [](nb::handle_t<Matrix2> h) {
            auto &self = nb::cast<Matrix2 &>(h);
            return nb::ndarray<nb::numpy, double, nb::shape<2, 2>>(&self.m[0][0], {2, 2}, h);
        }, "Zero-copy view as NumPy 2x2 array")
        .def_static("from_numpy", [](nb::ndarray<double, nb::shape<2, 2>> arr) {
            return Matrix2{arr(0, 0), arr(0, 1), arr(1, 0), arr(1, 1)};
        }, nb::arg("arr"))
        .def("__repr__", [](const Matrix2 &m) {
            return "Matrix2([[" + std::to_string(m.m[0][0]) + ", " + std::to_string(m.m[0][1]) + "], [" +
                   std::to_string(m.m[1][0]) + ", " + std::to_string(m.m[1][1]) + "]])";
        });

    // =========================================================================
    // Matrix3
    // =========================================================================
    nb::class_<Matrix3>(m, "Matrix3")
        .def(nb::init<double, double, double, double, double, double, double, double, double>(),
             nb::arg("m00") = 0.0, nb::arg("m01") = 0.0, nb::arg("m02") = 0.0,
             nb::arg("m10") = 0.0, nb::arg("m11") = 0.0, nb::arg("m12") = 0.0,
             nb::arg("m20") = 0.0, nb::arg("m21") = 0.0, nb::arg("m22") = 0.0)
        .def_static("identity", &Matrix3::identity)
        .def("transpose", &Matrix3::transpose)
        .def("determinant", &Matrix3::determinant)
        .def("inverse", [](const Matrix3 &self) -> nb::object {
            auto result = self.inverse();
            if (result.has_value())
                return nb::cast(result.value());
            return nb::none();
        })
        .def("__add__", &Matrix3::operator+)
        .def("__sub__", &Matrix3::operator-)
        .def("__mul__", static_cast<Matrix3 (Matrix3::*)(double) const>(&Matrix3::operator*), nb::arg("scalar"))
        .def("__matmul__", static_cast<Matrix3 (Matrix3::*)(const Matrix3 &) const>(&Matrix3::operator*))
        .def("__truediv__", &Matrix3::operator/, nb::arg("scalar"))
        .def("__eq__", &Matrix3::operator==)
        .def("numpy", [](nb::handle_t<Matrix3> h) {
            auto &self = nb::cast<Matrix3 &>(h);
            return nb::ndarray<nb::numpy, double, nb::shape<3, 3>>(&self.m[0][0], {3, 3}, h);
        }, "Zero-copy view as NumPy 3x3 array")
        .def_static("from_numpy", [](nb::ndarray<double, nb::shape<3, 3>> arr) {
            return Matrix3{arr(0, 0), arr(0, 1), arr(0, 2),
                           arr(1, 0), arr(1, 1), arr(1, 2),
                           arr(2, 0), arr(2, 1), arr(2, 2)};
        }, nb::arg("arr"))
        .def("__repr__", [](const Matrix3 &mat) {
            return "Matrix3(...)";
        });

    // =========================================================================
    // Matrix4
    // =========================================================================
    nb::class_<Matrix4>(m, "Matrix4")
        .def(nb::init<double, double, double, double,
                      double, double, double, double,
                      double, double, double, double,
                      double, double, double, double>(),
             nb::arg("m00") = 0.0, nb::arg("m01") = 0.0, nb::arg("m02") = 0.0, nb::arg("m03") = 0.0,
             nb::arg("m10") = 0.0, nb::arg("m11") = 0.0, nb::arg("m12") = 0.0, nb::arg("m13") = 0.0,
             nb::arg("m20") = 0.0, nb::arg("m21") = 0.0, nb::arg("m22") = 0.0, nb::arg("m23") = 0.0,
             nb::arg("m30") = 0.0, nb::arg("m31") = 0.0, nb::arg("m32") = 0.0, nb::arg("m33") = 0.0)
        .def_static("identity", &Matrix4::identity)
        .def("transpose", &Matrix4::transpose)
        .def("determinant", &Matrix4::determinant)
        .def("inverse", [](const Matrix4 &self) -> nb::object {
            auto result = self.inverse();
            if (result.has_value())
                return nb::cast(result.value());
            return nb::none();
        })
        .def("__add__", &Matrix4::operator+)
        .def("__sub__", &Matrix4::operator-)
        .def("__mul__", static_cast<Matrix4 (Matrix4::*)(double) const>(&Matrix4::operator*), nb::arg("scalar"))
        .def("__matmul__", static_cast<Matrix4 (Matrix4::*)(const Matrix4 &) const>(&Matrix4::operator*))
        .def("__truediv__", &Matrix4::operator/, nb::arg("scalar"))
        .def("__eq__", &Matrix4::operator==)
        .def("numpy", [](nb::handle_t<Matrix4> h) {
            auto &self = nb::cast<Matrix4 &>(h);
            return nb::ndarray<nb::numpy, double, nb::shape<4, 4>>(&self.m[0][0], {4, 4}, h);
        }, "Zero-copy view as NumPy 4x4 array")
        .def_static("from_numpy", [](nb::ndarray<double, nb::shape<4, 4>> arr) {
            return Matrix4{arr(0, 0), arr(0, 1), arr(0, 2), arr(0, 3),
                           arr(1, 0), arr(1, 1), arr(1, 2), arr(1, 3),
                           arr(2, 0), arr(2, 1), arr(2, 2), arr(2, 3),
                           arr(3, 0), arr(3, 1), arr(3, 2), arr(3, 3)};
        }, nb::arg("arr"))
        .def("__repr__", [](const Matrix4 &mat) {
            return "Matrix4(...)";
        });

    // =========================================================================
    // Quaternion
    // =========================================================================
    nb::class_<Quaternion>(m, "Quaternion")
        .def(nb::init<double, double, double, double>(),
             nb::arg("w") = 1.0, nb::arg("x") = 0.0, nb::arg("y") = 0.0, nb::arg("z") = 0.0)
        .def_rw("w", &Quaternion::w)
        .def_rw("x", &Quaternion::x)
        .def_rw("y", &Quaternion::y)
        .def_rw("z", &Quaternion::z)
        .def_static("identity", &Quaternion::identity)
        .def("normalized", &Quaternion::normalized)
        .def("inverse", &Quaternion::inverse)
        .def("norm", &Quaternion::norm)
        .def("norm_sq", &Quaternion::norm_sq)
        .def("dot", &Quaternion::dot, nb::arg("other"))
        .def("__add__", &Quaternion::operator+)
        .def("__sub__", static_cast<Quaternion (Quaternion::*)(const Quaternion &) const>(&Quaternion::operator-))
        .def("__mul__", static_cast<Quaternion (Quaternion::*)(const Quaternion &) const>(&Quaternion::operator*),
             nb::arg("other"))
        .def("mul_scalar", static_cast<Quaternion (Quaternion::*)(double) const>(&Quaternion::operator*),
             nb::arg("scalar"))
        .def("rotate", static_cast<Vector3 (Quaternion::*)(const Vector3 &) const>(&Quaternion::operator*),
             nb::arg("v"), "Rotate vector by this quaternion")
        .def("__truediv__", &Quaternion::operator/, nb::arg("scalar"))
        .def("__eq__", &Quaternion::operator==)
        // Zero-copy: [w, x, y, z] layout
        .def("numpy", [](nb::handle_t<Quaternion> h) {
            auto &self = nb::cast<Quaternion &>(h);
            return nb::ndarray<nb::numpy, double, nb::shape<4>>(&self.w, {4}, h);
        }, "Zero-copy view as NumPy array [w, x, y, z]")
        .def_static("from_numpy", [](nb::ndarray<double, nb::shape<4>> arr) {
            return Quaternion{arr(0), arr(1), arr(2), arr(3)};
        }, nb::arg("arr"), "Create from [w, x, y, z] array")
        // SciPy interop: [x, y, z, w] layout
        .def("to_scipy", [](const Quaternion &self) {
            double *data = new double[4]{self.x, self.y, self.z, self.w};
            nb::capsule owner(data, [](void *p) noexcept { delete[] static_cast<double *>(p); });
            return nb::ndarray<nb::numpy, double, nb::shape<4>>(data, {4}, owner);
        }, "Return [x, y, z, w] array (SciPy convention)")
        .def_static("from_scipy", [](nb::ndarray<double, nb::shape<4>> arr) {
            return Quaternion{arr(3), arr(0), arr(1), arr(2)};  // [x,y,z,w] -> Quaternion(w,x,y,z)
        }, nb::arg("arr"), "Create from [x, y, z, w] array (SciPy convention)")
        .def("__repr__", [](const Quaternion &q) {
            return "Quaternion(w=" + std::to_string(q.w) + ", x=" + std::to_string(q.x) +
                   ", y=" + std::to_string(q.y) + ", z=" + std::to_string(q.z) + ")";
        });

    // =========================================================================
    // Angle (Radians and Degrees as separate types)
    // =========================================================================
    using Radians = Angle<AngleType::RADIANS>;
    using Degrees = Angle<AngleType::DEGREES>;

    nb::class_<Radians>(m, "Radians")
        .def(nb::init<double>(), nb::arg("value") = 0.0)
        .def("value", &Radians::value)
        .def("sin", &Radians::sin)
        .def("cos", &Radians::cos)
        .def("tan", &Radians::tan)
        .def("to_degrees", &Radians::to_degrees)
        .def("__neg__", &Radians::operator-)
        .def("__repr__", [](const Radians &a) {
            return "Radians(" + std::to_string(a.value()) + ")";
        });

    nb::class_<Degrees>(m, "Degrees")
        .def(nb::init<double>(), nb::arg("value") = 0.0)
        .def("value", &Degrees::value)
        .def("sin", &Degrees::sin)
        .def("cos", &Degrees::cos)
        .def("tan", &Degrees::tan)
        .def("to_radians", &Degrees::to_radians)
        .def("__neg__", &Degrees::operator-)
        .def("__repr__", [](const Degrees &a) {
            return "Degrees(" + std::to_string(a.value()) + ")";
        });

    // =========================================================================
    // EulerAngles
    // =========================================================================
    using EulerRad = EulerAngles<AngleType::RADIANS>;

    nb::class_<EulerRad>(m, "EulerAngles")
        .def(nb::init<double, double, double>(),
             nb::arg("pitch") = 0.0, nb::arg("yaw") = 0.0, nb::arg("roll") = 0.0)
        .def_prop_ro("pitch", [](const EulerRad &e) { return e.pitch.value(); })
        .def_prop_ro("yaw", [](const EulerRad &e) { return e.yaw.value(); })
        .def_prop_ro("roll", [](const EulerRad &e) { return e.roll.value(); })
        .def("__repr__", [](const EulerRad &e) {
            return "EulerAngles(pitch=" + std::to_string(e.pitch.value()) +
                   ", yaw=" + std::to_string(e.yaw.value()) +
                   ", roll=" + std::to_string(e.roll.value()) + ")";
        });

    // =========================================================================
    // Free functions
    // =========================================================================
    m.def("quaternion_to_matrix", &quaternion_to_matrix, nb::arg("q"));
    m.def("quaternion_to_euler_angles", &quaternion_to_euler_angles, nb::arg("q"));
    m.def("euler_angles_to_quaternion", &euler_angles_to_quaternion, nb::arg("angles"));

    m.def("slerp", &slerp, nb::arg("a"), nb::arg("b"), nb::arg("t"));

    m.def("angle_between_vectors",
          static_cast<double (*)(const Vector3 &, const Vector3 &)>(&angle_between),
          nb::arg("a"), nb::arg("b"));
    m.def("angle_between_quaternions",
          static_cast<double (*)(const Quaternion &, const Quaternion &)>(&angle_between),
          nb::arg("a"), nb::arg("b"));

    // Matrix * Vector operations
    m.def("mat_mul", static_cast<Vector2 (*)(const Matrix2 &, const Vector2 &)>(&operator*),
          nb::arg("mat"), nb::arg("vec"));
    m.def("mat_mul", static_cast<Vector3 (*)(const Matrix3 &, const Vector3 &)>(&operator*),
          nb::arg("mat"), nb::arg("vec"));
    m.def("mat_mul", static_cast<Vector4 (*)(const Matrix4 &, const Vector4 &)>(&operator*),
          nb::arg("mat"), nb::arg("vec"));
}
