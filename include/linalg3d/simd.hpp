#pragma once

// =============================================================================
// Platform detection
// =============================================================================

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define LINALG3D_X86 1
#endif

#if defined(__AVX2__)
#define LINALG3D_AVX2 1
#define LINALG3D_AVX 1
#elif defined(__AVX__)
#define LINALG3D_AVX 1
#endif

#if defined(__FMA__)
#define LINALG3D_FMA 1
#endif

#if defined(__SSE2__) || defined(LINALG3D_AVX) || (defined(LINALG3D_X86) && defined(__x86_64__))
#define LINALG3D_SSE2 1
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#define LINALG3D_ARM64 1
#define LINALG3D_NEON 1
#endif

// Include intrinsic headers
#if defined(LINALG3D_AVX)
#include <immintrin.h>
#elif defined(LINALG3D_SSE2)
#include <emmintrin.h>
#endif

#if defined(LINALG3D_NEON)
#include <arm_neon.h>
#endif

namespace linalg3d::simd
{

// =============================================================================
// Matrix3 multiply: r = a * b  (3x3 row-major doubles)
// =============================================================================

inline void mat3_multiply(const double a[3][3], const double b[3][3], double r[3][3])
{
#if defined(LINALG3D_SSE2)
    // Process 2 columns at a time with SSE2, 3rd column scalar
    for (int i = 0; i < 3; ++i)
    {
        const __m128d ai0 = _mm_set1_pd(a[i][0]);
        const __m128d ai1 = _mm_set1_pd(a[i][1]);
        const __m128d ai2 = _mm_set1_pd(a[i][2]);

        const __m128d b0 = _mm_loadu_pd(&b[0][0]);
        const __m128d b1 = _mm_loadu_pd(&b[1][0]);
        const __m128d b2 = _mm_loadu_pd(&b[2][0]);

        __m128d res = _mm_add_pd(_mm_add_pd(_mm_mul_pd(ai0, b0), _mm_mul_pd(ai1, b1)), _mm_mul_pd(ai2, b2));
        _mm_storeu_pd(&r[i][0], res);

        r[i][2] = a[i][0] * b[0][2] + a[i][1] * b[1][2] + a[i][2] * b[2][2];
    }
#elif defined(LINALG3D_NEON)
    for (int i = 0; i < 3; ++i)
    {
        const float64x2_t ai0 = vdupq_n_f64(a[i][0]);
        const float64x2_t ai1 = vdupq_n_f64(a[i][1]);
        const float64x2_t ai2 = vdupq_n_f64(a[i][2]);

        const float64x2_t b0 = vld1q_f64(&b[0][0]);
        const float64x2_t b1 = vld1q_f64(&b[1][0]);
        const float64x2_t b2 = vld1q_f64(&b[2][0]);

        float64x2_t res = vfmaq_f64(vfmaq_f64(vmulq_f64(ai0, b0), ai1, b1), ai2, b2);
        vst1q_f64(&r[i][0], res);

        r[i][2] = a[i][0] * b[0][2] + a[i][1] * b[1][2] + a[i][2] * b[2][2];
    }
#else
    // Scalar fallback
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            r[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
#endif
}

// =============================================================================
// Matrix3 inverse: r = inverse(m), returns false if singular
// =============================================================================

inline bool mat3_inverse(const double m[3][3], double r[3][3])
{
    // Cofactors (transposed into adjugate directly)
    const double c00 = m[1][1] * m[2][2] - m[1][2] * m[2][1];
    const double c01 = m[0][2] * m[2][1] - m[0][1] * m[2][2];
    const double c02 = m[0][1] * m[1][2] - m[0][2] * m[1][1];

    const double c10 = m[1][2] * m[2][0] - m[1][0] * m[2][2];
    const double c11 = m[0][0] * m[2][2] - m[0][2] * m[2][0];
    const double c12 = m[0][2] * m[1][0] - m[0][0] * m[1][2];

    const double c20 = m[1][0] * m[2][1] - m[1][1] * m[2][0];
    const double c21 = m[0][1] * m[2][0] - m[0][0] * m[2][1];
    const double c22 = m[0][0] * m[1][1] - m[0][1] * m[1][0];

    const double det = m[0][0] * c00 + m[0][1] * c10 + m[0][2] * c20;
    if (det == 0.0)
        return false;

    const double inv_det = 1.0 / det;

#if defined(LINALG3D_SSE2)
    const __m128d vd = _mm_set1_pd(inv_det);

    // Row 0: c00, c01
    __m128d row0 = _mm_set_pd(c01, c00);
    _mm_storeu_pd(&r[0][0], _mm_mul_pd(row0, vd));
    r[0][2] = c02 * inv_det;

    // Row 1: c10, c11
    __m128d row1 = _mm_set_pd(c11, c10);
    _mm_storeu_pd(&r[1][0], _mm_mul_pd(row1, vd));
    r[1][2] = c12 * inv_det;

    // Row 2: c20, c21
    __m128d row2 = _mm_set_pd(c21, c20);
    _mm_storeu_pd(&r[2][0], _mm_mul_pd(row2, vd));
    r[2][2] = c22 * inv_det;
#elif defined(LINALG3D_NEON)
    const float64x2_t vd = vdupq_n_f64(inv_det);

    float64x2_t row0 = {c00, c01};
    vst1q_f64(&r[0][0], vmulq_f64(row0, vd));
    r[0][2] = c02 * inv_det;

    float64x2_t row1 = {c10, c11};
    vst1q_f64(&r[1][0], vmulq_f64(row1, vd));
    r[1][2] = c12 * inv_det;

    float64x2_t row2 = {c20, c21};
    vst1q_f64(&r[2][0], vmulq_f64(row2, vd));
    r[2][2] = c22 * inv_det;
#else
    r[0][0] = c00 * inv_det;
    r[0][1] = c01 * inv_det;
    r[0][2] = c02 * inv_det;
    r[1][0] = c10 * inv_det;
    r[1][1] = c11 * inv_det;
    r[1][2] = c12 * inv_det;
    r[2][0] = c20 * inv_det;
    r[2][1] = c21 * inv_det;
    r[2][2] = c22 * inv_det;
#endif

    return true;
}

// =============================================================================
// Matrix4 multiply: r = a * b  (4x4 row-major doubles)
// =============================================================================

inline void mat4_multiply(const double a[4][4], const double b[4][4], double r[4][4])
{
#if defined(LINALG3D_AVX)
    // AVX: process 4 doubles per row in one 256-bit register
    const __m256d b0 = _mm256_loadu_pd(&b[0][0]);
    const __m256d b1 = _mm256_loadu_pd(&b[1][0]);
    const __m256d b2 = _mm256_loadu_pd(&b[2][0]);
    const __m256d b3 = _mm256_loadu_pd(&b[3][0]);

    for (int i = 0; i < 4; ++i)
    {
        const __m256d ai0 = _mm256_set1_pd(a[i][0]);
        const __m256d ai1 = _mm256_set1_pd(a[i][1]);
        const __m256d ai2 = _mm256_set1_pd(a[i][2]);
        const __m256d ai3 = _mm256_set1_pd(a[i][3]);

#if defined(LINALG3D_FMA)
        __m256d res = _mm256_mul_pd(ai3, b3);
        res = _mm256_fmadd_pd(ai2, b2, res);
        res = _mm256_fmadd_pd(ai1, b1, res);
        res = _mm256_fmadd_pd(ai0, b0, res);
#else
        __m256d res = _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(ai0, b0), _mm256_mul_pd(ai1, b1)),
                                    _mm256_add_pd(_mm256_mul_pd(ai2, b2), _mm256_mul_pd(ai3, b3)));
#endif
        _mm256_storeu_pd(&r[i][0], res);
    }
#elif defined(LINALG3D_SSE2)
    // SSE2: process 2 doubles at a time
    const __m128d b0_lo = _mm_loadu_pd(&b[0][0]);
    const __m128d b0_hi = _mm_loadu_pd(&b[0][2]);
    const __m128d b1_lo = _mm_loadu_pd(&b[1][0]);
    const __m128d b1_hi = _mm_loadu_pd(&b[1][2]);
    const __m128d b2_lo = _mm_loadu_pd(&b[2][0]);
    const __m128d b2_hi = _mm_loadu_pd(&b[2][2]);
    const __m128d b3_lo = _mm_loadu_pd(&b[3][0]);
    const __m128d b3_hi = _mm_loadu_pd(&b[3][2]);

    for (int i = 0; i < 4; ++i)
    {
        const __m128d ai0 = _mm_set1_pd(a[i][0]);
        const __m128d ai1 = _mm_set1_pd(a[i][1]);
        const __m128d ai2 = _mm_set1_pd(a[i][2]);
        const __m128d ai3 = _mm_set1_pd(a[i][3]);

        __m128d lo = _mm_add_pd(_mm_add_pd(_mm_mul_pd(ai0, b0_lo), _mm_mul_pd(ai1, b1_lo)),
                                _mm_add_pd(_mm_mul_pd(ai2, b2_lo), _mm_mul_pd(ai3, b3_lo)));
        __m128d hi = _mm_add_pd(_mm_add_pd(_mm_mul_pd(ai0, b0_hi), _mm_mul_pd(ai1, b1_hi)),
                                _mm_add_pd(_mm_mul_pd(ai2, b2_hi), _mm_mul_pd(ai3, b3_hi)));
        _mm_storeu_pd(&r[i][0], lo);
        _mm_storeu_pd(&r[i][2], hi);
    }
#elif defined(LINALG3D_NEON)
    const float64x2_t b0_lo = vld1q_f64(&b[0][0]);
    const float64x2_t b0_hi = vld1q_f64(&b[0][2]);
    const float64x2_t b1_lo = vld1q_f64(&b[1][0]);
    const float64x2_t b1_hi = vld1q_f64(&b[1][2]);
    const float64x2_t b2_lo = vld1q_f64(&b[2][0]);
    const float64x2_t b2_hi = vld1q_f64(&b[2][2]);
    const float64x2_t b3_lo = vld1q_f64(&b[3][0]);
    const float64x2_t b3_hi = vld1q_f64(&b[3][2]);

    for (int i = 0; i < 4; ++i)
    {
        const float64x2_t ai0 = vdupq_n_f64(a[i][0]);
        const float64x2_t ai1 = vdupq_n_f64(a[i][1]);
        const float64x2_t ai2 = vdupq_n_f64(a[i][2]);
        const float64x2_t ai3 = vdupq_n_f64(a[i][3]);

        float64x2_t lo = vfmaq_f64(vfmaq_f64(vfmaq_f64(vmulq_f64(ai3, b3_lo), ai2, b2_lo), ai1, b1_lo), ai0, b0_lo);
        float64x2_t hi = vfmaq_f64(vfmaq_f64(vfmaq_f64(vmulq_f64(ai3, b3_hi), ai2, b2_hi), ai1, b1_hi), ai0, b0_hi);
        vst1q_f64(&r[i][0], lo);
        vst1q_f64(&r[i][2], hi);
    }
#else
    // Scalar fallback
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            r[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j] + a[i][3] * b[3][j];
#endif
}

// =============================================================================
// Matrix4 inverse: r = inverse(m), returns false if singular
// Uses sub-product algorithm (s0-s5, c0-c5) with SIMD for final multiply
// =============================================================================

inline bool mat4_inverse(const double m[4][4], double r[4][4])
{
    // Sub-products from rows 2,3
    const double s0 = m[2][0] * m[3][1] - m[2][1] * m[3][0];
    const double s1 = m[2][0] * m[3][2] - m[2][2] * m[3][0];
    const double s2 = m[2][0] * m[3][3] - m[2][3] * m[3][0];
    const double s3 = m[2][1] * m[3][2] - m[2][2] * m[3][1];
    const double s4 = m[2][1] * m[3][3] - m[2][3] * m[3][1];
    const double s5 = m[2][2] * m[3][3] - m[2][3] * m[3][2];

    // Sub-products from rows 0,1
    const double c0 = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    const double c1 = m[0][0] * m[1][2] - m[0][2] * m[1][0];
    const double c2 = m[0][0] * m[1][3] - m[0][3] * m[1][0];
    const double c3 = m[0][1] * m[1][2] - m[0][2] * m[1][1];
    const double c4 = m[0][1] * m[1][3] - m[0][3] * m[1][1];
    const double c5 = m[0][2] * m[1][3] - m[0][3] * m[1][2];

    const double det = c0 * s5 - c1 * s4 + c2 * s3 + c3 * s2 - c4 * s1 + c5 * s0;
    if (det == 0.0)
        return false;

    const double id = 1.0 / det;

#if defined(LINALG3D_AVX)
    // Compute adjugate rows using AVX: 4 elements at once
    // Each row = coeff_a * [s5/c5] + coeff_b * [s4/c4] + coeff_c * [s3/c3] (then * inv_det)
    // Row pattern: [m1x*sN, -m0x*sN, m3x*cN, -m2x*cN] with alternating sub-products
    const __m256d vid = _mm256_set1_pd(id);

    // Row 0: uses s5,s4,s3 and c5,c4,c3
    {
        const __m256d sp5 = _mm256_set_pd(-c5, c5, -s5, s5);
        const __m256d sp4 = _mm256_set_pd(c4, -c4, s4, -s4);
        const __m256d sp3 = _mm256_set_pd(-c3, c3, -s3, s3);
        const __m256d co_a = _mm256_set_pd(m[2][1], m[3][1], m[0][1], m[1][1]);
        const __m256d co_b = _mm256_set_pd(m[2][2], m[3][2], m[0][2], m[1][2]);
        const __m256d co_c = _mm256_set_pd(m[2][3], m[3][3], m[0][3], m[1][3]);

#if defined(LINALG3D_FMA)
        __m256d row = _mm256_mul_pd(co_c, sp3);
        row = _mm256_fmadd_pd(co_b, sp4, row);
        row = _mm256_fmadd_pd(co_a, sp5, row);
#else
        __m256d row = _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(co_a, sp5), _mm256_mul_pd(co_b, sp4)),
                                    _mm256_mul_pd(co_c, sp3));
#endif
        _mm256_storeu_pd(&r[0][0], _mm256_mul_pd(row, vid));
    }

    // Row 1: uses s5,s2,s1 and c5,c2,c1
    {
        const __m256d sp5 = _mm256_set_pd(c5, -c5, s5, -s5);
        const __m256d sp2 = _mm256_set_pd(-c2, c2, -s2, s2);
        const __m256d sp1 = _mm256_set_pd(c1, -c1, s1, -s1);
        const __m256d co_a = _mm256_set_pd(m[2][0], m[3][0], m[0][0], m[1][0]);
        const __m256d co_b = _mm256_set_pd(m[2][2], m[3][2], m[0][2], m[1][2]);
        const __m256d co_c = _mm256_set_pd(m[2][3], m[3][3], m[0][3], m[1][3]);

#if defined(LINALG3D_FMA)
        __m256d row = _mm256_mul_pd(co_c, sp1);
        row = _mm256_fmadd_pd(co_b, sp2, row);
        row = _mm256_fmadd_pd(co_a, sp5, row);
#else
        __m256d row = _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(co_a, sp5), _mm256_mul_pd(co_b, sp2)),
                                    _mm256_mul_pd(co_c, sp1));
#endif
        _mm256_storeu_pd(&r[1][0], _mm256_mul_pd(row, vid));
    }

    // Row 2: uses s4,s2,s0 and c4,c2,c0
    {
        const __m256d sp4 = _mm256_set_pd(-c4, c4, -s4, s4);
        const __m256d sp2 = _mm256_set_pd(c2, -c2, s2, -s2);
        const __m256d sp0 = _mm256_set_pd(-c0, c0, -s0, s0);
        const __m256d co_a = _mm256_set_pd(m[2][0], m[3][0], m[0][0], m[1][0]);
        const __m256d co_b = _mm256_set_pd(m[2][1], m[3][1], m[0][1], m[1][1]);
        const __m256d co_c = _mm256_set_pd(m[2][3], m[3][3], m[0][3], m[1][3]);

#if defined(LINALG3D_FMA)
        __m256d row = _mm256_mul_pd(co_c, sp0);
        row = _mm256_fmadd_pd(co_b, sp2, row);
        row = _mm256_fmadd_pd(co_a, sp4, row);
#else
        __m256d row = _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(co_a, sp4), _mm256_mul_pd(co_b, sp2)),
                                    _mm256_mul_pd(co_c, sp0));
#endif
        _mm256_storeu_pd(&r[2][0], _mm256_mul_pd(row, vid));
    }

    // Row 3: uses s3,s1,s0 and c3,c1,c0
    {
        const __m256d sp3 = _mm256_set_pd(c3, -c3, s3, -s3);
        const __m256d sp1 = _mm256_set_pd(-c1, c1, -s1, s1);
        const __m256d sp0 = _mm256_set_pd(c0, -c0, s0, -s0);
        const __m256d co_a = _mm256_set_pd(m[2][0], m[3][0], m[0][0], m[1][0]);
        const __m256d co_b = _mm256_set_pd(m[2][1], m[3][1], m[0][1], m[1][1]);
        const __m256d co_c = _mm256_set_pd(m[2][2], m[3][2], m[0][2], m[1][2]);

#if defined(LINALG3D_FMA)
        __m256d row = _mm256_mul_pd(co_c, sp0);
        row = _mm256_fmadd_pd(co_b, sp1, row);
        row = _mm256_fmadd_pd(co_a, sp3, row);
#else
        __m256d row = _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(co_a, sp3), _mm256_mul_pd(co_b, sp1)),
                                    _mm256_mul_pd(co_c, sp0));
#endif
        _mm256_storeu_pd(&r[3][0], _mm256_mul_pd(row, vid));
    }
#else
    // Scalar + SSE2/NEON path: compute adjugate then scale
    // clang-format off
    r[0][0] = ( m[1][1]*s5 - m[1][2]*s4 + m[1][3]*s3) * id;
    r[0][1] = (-m[0][1]*s5 + m[0][2]*s4 - m[0][3]*s3) * id;
    r[0][2] = ( m[3][1]*c5 - m[3][2]*c4 + m[3][3]*c3) * id;
    r[0][3] = (-m[2][1]*c5 + m[2][2]*c4 - m[2][3]*c3) * id;

    r[1][0] = (-m[1][0]*s5 + m[1][2]*s2 - m[1][3]*s1) * id;
    r[1][1] = ( m[0][0]*s5 - m[0][2]*s2 + m[0][3]*s1) * id;
    r[1][2] = (-m[3][0]*c5 + m[3][2]*c2 - m[3][3]*c1) * id;
    r[1][3] = ( m[2][0]*c5 - m[2][2]*c2 + m[2][3]*c1) * id;

    r[2][0] = ( m[1][0]*s4 - m[1][1]*s2 + m[1][3]*s0) * id;
    r[2][1] = (-m[0][0]*s4 + m[0][1]*s2 - m[0][3]*s0) * id;
    r[2][2] = ( m[3][0]*c4 - m[3][1]*c2 + m[3][3]*c0) * id;
    r[2][3] = (-m[2][0]*c4 + m[2][1]*c2 - m[2][3]*c0) * id;

    r[3][0] = (-m[1][0]*s3 + m[1][1]*s1 - m[1][2]*s0) * id;
    r[3][1] = ( m[0][0]*s3 - m[0][1]*s1 + m[0][2]*s0) * id;
    r[3][2] = (-m[3][0]*c3 + m[3][1]*c1 - m[3][2]*c0) * id;
    r[3][3] = ( m[2][0]*c3 - m[2][1]*c1 + m[2][2]*c0) * id;
    // clang-format on
#endif

    return true;
}

// =============================================================================
// Quaternion multiply: Hamilton product
// Input/output layout: [w, x, y, z] as contiguous doubles
// =============================================================================

inline void quat_multiply(const double *a, const double *b, double *r)
{
    const double aw = a[0], ax = a[1], ay = a[2], az = a[3];
    const double bw = b[0], bx = b[1], by = b[2], bz = b[3];

#if defined(LINALG3D_AVX)
    // Pack result components: all 4 at once using AVX
    // w = aw*bw - ax*bx - ay*by - az*bz
    // x = aw*bx + ax*bw + ay*bz - az*by
    // y = aw*by - ax*bz + ay*bw + az*bx
    // z = aw*bz + ax*by - ay*bx + az*bw

    const __m256d va = _mm256_set1_pd(aw);
    const __m256d col_w = _mm256_set_pd(bz, by, bx, bw);

    const __m256d vb = _mm256_set1_pd(ax);
    const __m256d col_x = _mm256_set_pd(by, -bz, bw, -bx);

    const __m256d vc = _mm256_set1_pd(ay);
    const __m256d col_y = _mm256_set_pd(-bx, bw, bz, -by);

    const __m256d vd = _mm256_set1_pd(az);
    const __m256d col_z = _mm256_set_pd(bw, bx, -by, -bz);

#if defined(LINALG3D_FMA)
    __m256d res = _mm256_mul_pd(vd, col_z);
    res = _mm256_fmadd_pd(vc, col_y, res);
    res = _mm256_fmadd_pd(vb, col_x, res);
    res = _mm256_fmadd_pd(va, col_w, res);
#else
    __m256d res = _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(va, col_w), _mm256_mul_pd(vb, col_x)),
                                _mm256_add_pd(_mm256_mul_pd(vc, col_y), _mm256_mul_pd(vd, col_z)));
#endif
    _mm256_storeu_pd(r, res);

#elif defined(LINALG3D_SSE2)
    // SSE2: process 2 components at a time
    // [w, x] pair
    const __m128d aw_v = _mm_set1_pd(aw);
    const __m128d ax_v = _mm_set1_pd(ax);
    const __m128d ay_v = _mm_set1_pd(ay);
    const __m128d az_v = _mm_set1_pd(az);

    const __m128d bwx = _mm_set_pd(bx, bw);
    const __m128d col_x_wx = _mm_set_pd(bw, -bx);
    const __m128d col_y_wx = _mm_set_pd(bz, -by);
    const __m128d col_z_wx = _mm_set_pd(-by, -bz);

    __m128d wx = _mm_add_pd(_mm_add_pd(_mm_mul_pd(aw_v, bwx), _mm_mul_pd(ax_v, col_x_wx)),
                            _mm_add_pd(_mm_mul_pd(ay_v, col_y_wx), _mm_mul_pd(az_v, col_z_wx)));
    _mm_storeu_pd(r, wx);

    // [y, z] pair
    const __m128d byz = _mm_set_pd(bz, by);
    const __m128d col_x_yz = _mm_set_pd(by, -bz);
    const __m128d col_y_yz = _mm_set_pd(-bx, bw);
    const __m128d col_z_yz = _mm_set_pd(bw, bx);

    __m128d yz = _mm_add_pd(_mm_add_pd(_mm_mul_pd(aw_v, byz), _mm_mul_pd(ax_v, col_x_yz)),
                            _mm_add_pd(_mm_mul_pd(ay_v, col_y_yz), _mm_mul_pd(az_v, col_z_yz)));
    _mm_storeu_pd(r + 2, yz);

#elif defined(LINALG3D_NEON)
    const float64x2_t aw_v = vdupq_n_f64(aw);
    const float64x2_t ax_v = vdupq_n_f64(ax);
    const float64x2_t ay_v = vdupq_n_f64(ay);
    const float64x2_t az_v = vdupq_n_f64(az);

    // [w, x]
    const float64x2_t bwx = {bw, bx};
    const float64x2_t cx_wx = {-bx, bw};
    const float64x2_t cy_wx = {-by, bz};
    const float64x2_t cz_wx = {-bz, -by};

    float64x2_t wx = vfmaq_f64(vfmaq_f64(vfmaq_f64(vmulq_f64(az_v, cz_wx), ay_v, cy_wx), ax_v, cx_wx), aw_v, bwx);
    vst1q_f64(r, wx);

    // [y, z]
    const float64x2_t byz = {by, bz};
    const float64x2_t cx_yz = {-bz, by};
    const float64x2_t cy_yz = {bw, -bx};
    const float64x2_t cz_yz = {bx, bw};

    float64x2_t yz = vfmaq_f64(vfmaq_f64(vfmaq_f64(vmulq_f64(az_v, cz_yz), ay_v, cy_yz), ax_v, cx_yz), aw_v, byz);
    vst1q_f64(r + 2, yz);

#else
    // Scalar fallback
    r[0] = aw * bw - ax * bx - ay * by - az * bz;
    r[1] = aw * bx + ax * bw + ay * bz - az * by;
    r[2] = aw * by - ax * bz + ay * bw + az * bx;
    r[3] = aw * bz + ax * by - ay * bx + az * bw;
#endif
}

} // namespace linalg3d::simd
