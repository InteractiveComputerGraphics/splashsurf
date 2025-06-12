//! SPH kernel function implementations

use crate::{Real, RealConvert};
use nalgebra::Vector3;
use numeric_literals::replace_float_literals;

// TODO: Add reference for the kernel function, document formula

/// Trait for symmetric kernel functions in three dimensions
pub trait SymmetricKernel3d<R: Real> {
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
            let i_and_half = PR::from_usize(i).unwrap() + PR::from_f64(0.5).unwrap();
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
