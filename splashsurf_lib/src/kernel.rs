//! SPH kernel function implementations
//!
//! Currently, the following SIMD implementations are provided:
//!  - `CubicSplineKernelAvxF32`: Only available on `x86` and `x86_64` targets, requires AVX2 and FMA support
//!  - `CubicSplineKernelNeonF32`: Only available on `aarch64` targets, requires NEON support
//!
//! Note that documentation of the SIMD kernels is only available on the respective target architectures.

use crate::{Real, RealConvert};
use nalgebra::Vector3;
use numeric_literals::replace_float_literals;

#[cfg(all(target_arch = "aarch64"))]
use core::arch::aarch64::float32x4_t;

#[cfg(target_arch = "x86")]
use core::arch::x86::__m256;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::__m256;

// TODO: Add reference for the kernel function, document formula

/// Utility functions for computing the volume of fluid particles
pub struct Volume;

impl Volume {
    /// Returns the volume of a particle representing a cube of fluid
    pub fn cube_particle<R: Real>(particle_radius: R) -> R {
        (particle_radius + particle_radius).powi(3)
    }

    /// Returns the volume of a particle representing a sphere of fluid
    pub fn sphere_particle<R: Real>(particle_radius: R) -> R {
        R::from_float(4.0) * R::frac_pi_3() * particle_radius.powi(3)
    }
}

/// Trait for symmetric kernel functions in three dimensions
pub trait SymmetricKernel3d<R: Real> {
    /// Returns the compact support radius of the kernel
    fn compact_support_radius(&self) -> R;
    /// Evaluates the kernel at the radial distance `r` relative to the origin
    fn evaluate(&self, r: R) -> R;
    /// Evaluates the kernel gradient at the position `x` relative to the origin
    fn evaluate_gradient(&self, x: Vector3<R>) -> Vector3<R>;
    /// Evaluates the norm of the kernel gradient at the radial distance `r` relative to the origin, this may be faster than computing the full gradient
    fn evaluate_gradient_norm(&self, r: R) -> R;
}

/// The commonly used cubic spline kernel
pub struct CubicSplineKernel<R: Real> {
    /// Compact support radius of the kernel
    compact_support_radius: R,
    /// Kernel normalization factor (sigma)
    normalization: R,
}

impl<R: Real> CubicSplineKernel<R> {
    /// Initializes a cubic spline kernel with the given compact support radius
    #[replace_float_literals(R::from_float(literal))]
    pub fn new(compact_support_radius: R) -> Self {
        let h = compact_support_radius;
        let sigma = 8.0 / (h * h * h);

        Self {
            compact_support_radius,
            normalization: sigma,
        }
    }

    /// The cubic spline function used by the cubic spline kernel
    #[replace_float_literals(R::from_float(literal))]
    fn cubic_function(q: R) -> R {
        if q < R::one() {
            (3.0 / (2.0 * R::pi())) * ((2.0 / 3.0) - q * q + 0.5 * q * q * q)
        } else if q < 2.0 {
            let x = 2.0 - q;
            return (R::one() / (4.0 * R::pi())) * x * x * x;
        } else {
            return 0.0;
        }
    }

    /// The derivative of the cubic spline function used by the cubic spline kernel w.r.t to the parameter `q`
    #[replace_float_literals(R::from_float(literal))]
    fn cubic_function_dq(q: R) -> R {
        if q < 1.0 {
            (3.0 / (4.0 * R::pi())) * (-4.0 * q + 3.0 * q * q)
        } else if q < 2.0 {
            let x = 2.0 - q;
            return -(3.0 / (4.0 * R::pi())) * x * x;
        } else {
            return 0.0;
        }
    }
}

impl<R: Real> SymmetricKernel3d<R> for CubicSplineKernel<R> {
    fn compact_support_radius(&self) -> R {
        self.compact_support_radius
    }

    /// Evaluates the cubic spline kernel at the radial distance `r`
    fn evaluate(&self, r: R) -> R {
        let q = (r + r) / self.compact_support_radius;
        self.normalization * Self::cubic_function(q)
    }

    /// Evaluates the gradient of the cubic spline kernel at the position `x`
    fn evaluate_gradient(&self, x: Vector3<R>) -> Vector3<R> {
        // Kernel gradient is given by
        //   df/dq * dq/dr * dr/dx
        // where:
        //   f is the cubic spline
        //   q is the spline parameter
        //   r is the radial distance
        //   x is the position where the kernel gradient is evaluated

        // Radial distance is norm of position
        let r = x.norm();
        // Normalize the position vector: points into direction of gradient due to symmetry
        let drdx = x.unscale(r);

        let q = (r + r) / self.compact_support_radius;

        let dfdq = Self::cubic_function_dq(q);
        let dqdr = (R::one() + R::one()) / self.compact_support_radius;

        drdx.scale(self.normalization * dfdq * dqdr)
    }

    /// Evaluates the norm of the gradient of the cubic spline kernel at the radial distance `r`
    fn evaluate_gradient_norm(&self, r: R) -> R {
        let q = (r + r) / self.compact_support_radius;

        let dfdq = Self::cubic_function_dq(q);
        let dqdr = (R::one() + R::one()) / self.compact_support_radius;

        self.normalization * dfdq * dqdr
    }
}

#[test]
fn test_cubic_kernel_r_compact_support() {
    let hs = [0.025, 0.1, 2.0];
    for &h in hs.iter() {
        let kernel = CubicSplineKernel::new(h);
        assert_eq!(kernel.evaluate(h), 0.0);
        assert_eq!(kernel.evaluate(2.0 * h), 0.0);
        assert_eq!(kernel.evaluate(10.0 * h), 0.0);
    }
}

#[test]
fn test_cubic_kernel_r_integral() {
    let hs = [0.025, 0.1, 2.0];
    let n = 10;

    for &h in hs.iter() {
        let kernel = CubicSplineKernel::new(h);

        let dr = h / (n as f64);
        let dvol = dr * dr * dr;

        let mut integral = 0.0;
        for i in -n..n {
            for j in -n..n {
                for k in -n..n {
                    let r_in = Vector3::new(i as f64, j as f64, k as f64) * dr;
                    let r_out = Vector3::new((i + 1) as f64, (j + 1) as f64, (k + 1) as f64) * dr;
                    let r = ((r_in + r_out) * 0.5).norm();

                    integral += dvol * kernel.evaluate(r);
                }
            }
        }

        assert!((integral - 1.0).abs() <= 1e-5);
    }
}

/// Vectorized implementation of the cubic spline kernel using NEON instructions. Only available on `aarch64` targets.
#[cfg(target_arch = "aarch64")]
pub struct CubicSplineKernelNeonF32 {
    compact_support_inv: f32,
    sigma: f32,
}

#[cfg(target_arch = "aarch64")]
impl CubicSplineKernelNeonF32 {
    /// Initializes a cubic spline kernel with the given compact support radius
    pub fn new(compact_support_radius: f32) -> Self {
        let r = compact_support_radius;
        let compact_support_inv = 1.0 / r;
        let rrr = r * r * r;
        let sigma = 8.0 / (std::f32::consts::PI * rrr);
        Self {
            compact_support_inv,
            sigma,
        }
    }

    /// Evaluates the cubic spline kernel at the specified radial distances
    #[target_feature(enable = "neon")]
    pub fn evaluate(&self, r: float32x4_t) -> float32x4_t {
        use core::arch::aarch64::*;

        let one = vdupq_n_f32(1.0);
        let half = vdupq_n_f32(0.5);
        let zero = vdupq_n_f32(0.0);

        // q = r / h, v = 1 - q
        let q = vmulq_n_f32(r, self.compact_support_inv);
        let v = vsubq_f32(one, q);
        // Clamp v to [0, 1] to implicitly zero-out contributions with q > 1
        let v = vmaxq_f32(v, zero);

        // Reuse v^2 and v^3 for both branches
        let v2 = vmulq_f32(v, v);
        let v3 = vmulq_f32(v2, v);

        // Outer branch (0.5 < q <= 1.0): 2*sigma*(1 - q)^3 = 2*sigma*v^3
        let res_outer = vmulq_n_f32(v3, 2.0 * self.sigma);

        // Inner branch (q <= 0.5) rewritten in terms of v to avoid computing q^2:
        // sigma * (1 - 6q^2 + 6q^3) == sigma * (1 - 6v + 12v^2 - 6v^3)
        let mut res_inner = vdupq_n_f32(self.sigma);
        res_inner = vmlsq_n_f32(res_inner, v, 6.0 * self.sigma); // -6*sigma*v
        res_inner = vmlaq_n_f32(res_inner, v2, 12.0 * self.sigma); // +12*sigma*v^2
        res_inner = vmlsq_n_f32(res_inner, v3, 6.0 * self.sigma); // -6*sigma*v^3

        // Select inner for q <= 0.5, else outer; v was clamped so q > 1 yields 0 automatically
        let leq_than_half = vcleq_f32(q, half);
        vbslq_f32(leq_than_half, res_inner, res_outer)
    }
}

#[test]
#[cfg_attr(
    not(all(target_arch = "aarch64", target_feature = "neon")),
    ignore = "Skipped on non-aarch64 targets"
)]
fn test_cubic_spline_kernel_neon() {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        use core::arch::aarch64::*;

        // Test a few representative compact support radii
        let hs: [f32; 3] = [0.025, 0.1, 2.0];
        for &h in hs.iter() {
            let scalar = CubicSplineKernel::new(h);
            let neon = CubicSplineKernelNeonF32::new(h);

            // Sample radii from 0 to 2h (beyond support should be 0)
            let n: usize = 1024;
            let mut r0: f32 = 0.0;
            let dr: f32 = (2.0 * h) / (n as f32);

            for _chunk in 0..(n / 4) {
                // Prepare 4 lanes of radii
                let rs = [r0, r0 + dr, r0 + 2.0 * dr, r0 + 3.0 * dr];
                let r_vec = unsafe { vld1q_f32(rs.as_ptr()) };

                // Evaluate NEON and store back to array
                let w_vec = unsafe { neon.evaluate(r_vec) };
                let mut w_neon = [0.0f32; 4];
                unsafe { vst1q_f32(w_neon.as_mut_ptr(), w_vec) };

                // Compare against scalar lane-wise
                for lane in 0..4 {
                    let r_lane = rs[lane];
                    let w_scalar = scalar.evaluate(r_lane);
                    let diff = (w_neon[lane] - w_scalar).abs();

                    // Absolute tolerance with mild relative component to be robust across scales
                    let tol = 5e-6_f32.max(2e-5_f32 * w_scalar.abs());
                    assert!(
                        diff <= tol,
                        "NEON kernel mismatch (h={}, r={}, lane={}): neon={}, scalar={}, diff={}, tol={}",
                        h,
                        r_lane,
                        lane,
                        w_neon[lane],
                        w_scalar,
                        diff,
                        tol
                    );
                }

                r0 += 4.0 * dr;
            }

            // Also check a couple of out-of-support points explicitly
            for &r in &[h * 1.01, h * 1.5, h * 2.0, h * 2.5] {
                let w_scalar = scalar.evaluate(r);
                let w_neon = {
                    let v = unsafe { vld1q_f32([r, r, r, r].as_ptr()) };
                    let w = unsafe { neon.evaluate(v) };
                    let mut tmp = [0.0f32; 4];
                    unsafe { vst1q_f32(tmp.as_mut_ptr(), w) };
                    tmp[0]
                };
                let diff = (w_neon - w_scalar).abs();
                let tol = 5e-6_f32.max(1e-5_f32 * w_scalar.abs());
                assert!(
                    diff <= tol,
                    "NEON kernel mismatch outside support (h={}, r={}): neon={}, scalar={}, diff={}, tol={}",
                    h,
                    r,
                    w_neon,
                    w_scalar,
                    diff,
                    tol
                );
            }
        }
    }
}

/// Vectorized implementation of the cubic spline kernel using AVX2 and FMA instructions. Only available on `x86` and `x86_64` targets.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub struct CubicSplineKernelAvxF32 {
    compact_support_inv: f32,
    sigma: f32,
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
impl CubicSplineKernelAvxF32 {
    /// Initializes a cubic spline kernel with the given compact support radius
    pub fn new(compact_support_radius: f32) -> Self {
        let r = compact_support_radius;
        let compact_support_inv = 1.0 / r;
        let rrr = r * r * r;
        let sigma = 8.0 / (std::f32::consts::PI * rrr);
        Self {
            compact_support_inv,
            sigma,
        }
    }

    /// Evaluates the cubic spline kernel at the specified radial distances
    #[target_feature(enable = "avx2,fma")]
    pub fn evaluate(&self, r: __m256) -> __m256 {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::*;

        let one = _mm256_set1_ps(1.0);
        let half = _mm256_set1_ps(0.5);
        let zero = _mm256_set1_ps(0.0);

        // q = r / h, v = 1 - q
        let q = _mm256_mul_ps(r, _mm256_set1_ps(self.compact_support_inv));
        let mut v = _mm256_sub_ps(one, q);
        // Clamp v to [0, 1] to implicitly zero-out contributions with q > 1
        v = _mm256_max_ps(v, zero);

        // v^2 and v^3
        let v2 = _mm256_mul_ps(v, v);
        let v3 = _mm256_mul_ps(v2, v);

        // Outer: 2*sigma*v^3
        let res_outer = _mm256_mul_ps(v3, _mm256_set1_ps(2.0 * self.sigma));

        // Inner: sigma * (1 - 6v + 12v^2 - 6v^3)
        let mut res_inner = _mm256_set1_ps(self.sigma);
        // res_inner = res_inner - 6*sigma*v
        res_inner = _mm256_fnmadd_ps(v, _mm256_set1_ps(6.0 * self.sigma), res_inner);
        // res_inner = res_inner + 12*sigma*v^2
        res_inner = _mm256_fmadd_ps(v2, _mm256_set1_ps(12.0 * self.sigma), res_inner);
        // res_inner = res_inner - 6*sigma*v^3
        res_inner = _mm256_fnmadd_ps(v3, _mm256_set1_ps(6.0 * self.sigma), res_inner);

        // Select inner for q <= 0.5, else outer
        let leq_than_half = _mm256_cmp_ps::<_CMP_LE_OQ>(q, half);
        _mm256_blendv_ps(res_outer, res_inner, leq_than_half)
    }
}

#[test]
#[cfg_attr(
    not(all(
        any(target_arch = "x86_64", target_arch = "x86"),
        target_feature = "avx2",
        target_feature = "fma"
    )),
    ignore = "Skipped on non-x86 targets"
)]
fn test_cubic_spline_kernel_avx() {
    #[cfg(all(
        any(target_arch = "x86_64", target_arch = "x86"),
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::*;

        // Test a few representative compact support radii
        let hs: [f32; 3] = [0.025, 0.1, 2.0];
        for &h in hs.iter() {
            let scalar = CubicSplineKernel::new(h);
            let avx = CubicSplineKernelAvxF32::new(h);

            // Sample radii from 0 to 2h (beyond support should be 0)
            let n: usize = 1024;
            let mut r0: f32 = 0.0;
            let dr: f32 = (2.0 * h) / (n as f32);

            for _chunk in 0..(n / 8) {
                // Prepare 8 lanes of radii
                let rs = [
                    r0,
                    r0 + dr,
                    r0 + 2.0 * dr,
                    r0 + 3.0 * dr,
                    r0 + 4.0 * dr,
                    r0 + 5.0 * dr,
                    r0 + 6.0 * dr,
                    r0 + 7.0 * dr,
                ];

                // Evaluate AVX and store back to array
                let r_vec = unsafe { _mm256_loadu_ps(rs.as_ptr()) };
                let w_vec = unsafe { avx.evaluate(r_vec) };
                let mut w_avx = [0.0f32; 8];
                unsafe { _mm256_storeu_ps(w_avx.as_mut_ptr(), w_vec) };

                // Compare against scalar lane-wise
                for lane in 0..8 {
                    let r_lane = rs[lane];
                    let w_scalar = scalar.evaluate(r_lane);
                    let diff = (w_avx[lane] - w_scalar).abs();

                    // Absolute tolerance with mild relative component to be robust across scales
                    let tol = 1e-6_f32.max(1e-5_f32 * w_scalar.abs());
                    assert!(
                        diff <= tol,
                        "AVX kernel mismatch (h={}, r={}, lane={}): avx={}, scalar={}, diff={}, tol={}",
                        h,
                        r_lane,
                        lane,
                        w_avx[lane],
                        w_scalar,
                        diff,
                        tol
                    );
                }

                r0 += 8.0 * dr;
            }

            // Also check a couple of out-of-support points explicitly
            for &r in &[h * 1.01, h * 1.5, h * 2.0, h * 2.5] {
                let w_scalar = scalar.evaluate(r);
                let w_avx = {
                    let v = unsafe { _mm256_set1_ps(r) };
                    let w = unsafe { avx.evaluate(v) };
                    let mut tmp = [0.0f32; 8];
                    unsafe { _mm256_storeu_ps(tmp.as_mut_ptr(), w) };
                    tmp[0]
                };
                let diff = (w_avx - w_scalar).abs();
                let tol = 1e-6_f32.max(1e-5_f32 * w_scalar.abs());
                assert!(
                    diff <= tol,
                    "AVX kernel mismatch outside support (h={}, r={}): avx={}, scalar={}, diff={}, tol={}",
                    h,
                    r,
                    w_avx,
                    w_scalar,
                    diff,
                    tol
                );
            }
        }
    }
}

/// Accelerator for efficient evaluation of a precomputed cubic kernel
///
/// This structure is used to pre-compute a discrete representation of the cubic kernel function.
/// In some computations that require many evaluations of the kernel, it is more efficient to evaluate
/// the kernel using a squared distance to avoid taking the square root.
/// To produce an appropriate quantization of the kernel for this use case, the compact support
/// radius of the kernel is divided into `n` segments with quadratically increasing width.
/// To be more precise, on a quadratic scale, the compact support `[0, h*h]` is divided into `n` equally sized
/// segments of width `dr`. For the actual pre-computation, the exact kernel `k(r)` is evaluated at the
/// midpoint `m_i` of every segment given by `m_i = sqrt(i * dr)` for `i ∈ [0, n]`.
/// This results in an array of kernel values `K` that can be evaluated at runtime using a squared radius
/// `s` by just mapping this radius back to the corresponding segment index `i` followed by a lookup
/// in the value array, i.e. `k(sqrt(s)) ≈ K[s/dr]` (while taking care of rounding and clamping to the
/// allowed index range).
pub struct DiscreteSquaredDistanceCubicKernel<R: Real> {
    /// Precomputed values of the kernel function
    values: Vec<R>,
    /// The radial resolution of the discretization on a quadratic scale
    dr: R,
}

impl<R: Real> DiscreteSquaredDistanceCubicKernel<R> {
    /// Precomputes the discrete cubic kernel with compact support radius `h`, the squared radius `h * h` is divided into `n` segments for the quantization
    pub fn new<PR: Real>(n: usize, h: R) -> Self {
        let mut values = Vec::with_capacity(n);

        let compact_support: PR = h
            .try_convert()
            .expect("Compact support radius `h` has to fit into kernel pre-computation type `PR`");
        let compact_support_squared = compact_support * compact_support;

        let kernel = CubicSplineKernel::new(compact_support);

        // Radial width of one discrete kernel value
        let dr = compact_support_squared
            / PR::from_usize(n)
                .expect("Number of discrete kernel steps `n` has to fit into kernel pre-computation type `PR`");
        // Evaluate the kernel per discrete segment
        for i in 0..n {
            let i_and_half = PR::from_usize(i).unwrap() + PR::from_float(0.5);
            let r_squared = dr * i_and_half;
            let r = r_squared.sqrt();

            let kernel_value = kernel.evaluate(r);
            values.push(
                kernel_value
                    .try_convert()
                    .expect("Kernel value has to fit into target type `R`"),
            );
        }

        let dr = dr.try_convert().unwrap();
        Self { values, dr }
    }

    /// Evaluates the precomputed kernel function at the specified squared radius, i.e. returns an approximate cubic kernel value at the radius `sqrt(r_squared)`
    #[inline(always)]
    pub fn evaluate(&self, r_squared: R) -> R {
        let normalized = (r_squared / self.dr).round();
        let bin = normalized.to_usize().unwrap().min(self.values.len() - 1);
        self.values[bin]
    }
}

#[test]
fn test_discrete_kernel() {
    let n = 10000;
    let h = 0.025;

    let discrete_kernel = DiscreteSquaredDistanceCubicKernel::new::<f64>(n, h);
    let kernel = CubicSplineKernel::new(h);

    // Test the pre-computed values using a linear stepping
    let dr = h / (n as f64);
    for i in 0..n {
        let r = (i as f64) * dr;
        let rr = r * r;

        let discrete = discrete_kernel.evaluate(rr);
        let continuous = kernel.evaluate(r);

        let diff = (discrete - continuous).abs();
        let rel_diff = diff / continuous;
        if rel_diff > 5e-2 && diff > 1e-1 {
            eprintln!(
                "at r={}, r/h={}, discrete: {}, continuous: {}, diff: {}, rel_diff: {}",
                r,
                r / h,
                discrete,
                continuous,
                diff,
                rel_diff
            );
            assert!(false);
        }
    }
}
