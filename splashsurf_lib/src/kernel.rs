use crate::Real;

const ALPHA: f64 = 3.0 / (2.0 * std::f64::consts::PI);
const TWO_THIRDS: f64 = 2.0 / 3.0;
const ONE_SIXTH: f64 = 1.0 / 6.0;

// TODO: Add reference for the kernel function, document formula

/// The cubic kernel function based on the parameter `q`
#[inline(always)]
fn cubic_function_f64(q: f64) -> f64 {
    if q < 1.0 {
        return ALPHA * (TWO_THIRDS - q * q + 0.5 * q * q * q);
    } else if q < 2.0 {
        let x = 2.0 - q;
        return ALPHA * ONE_SIXTH * x * x * x;
    } else {
        return 0.0;
    }
}

/// Evaluates the cubic kernel with compact support radius `h` at the radius `r`, `f64` version
#[inline(always)]
pub fn cubic_kernel_r_f64(r: f64, h: f64) -> f64 {
    let q = (2.0 * r) / h;
    8.0 * cubic_function_f64(q) / (h * h * h)
}

/// Evaluates the cubic kernel with compact support radius `h` at the radius `r`, generic version
#[inline(always)]
pub fn cubic_kernel_r<R: Real>(r: R, h: R) -> R {
    let r = r.to_f64().unwrap();
    let h = h.to_f64().unwrap();

    R::from_f64(cubic_kernel_r_f64(r, h)).unwrap()
}

#[test]
fn test_cubic_kernel_r_compact_support() {
    let hs = [0.025, 0.1, 2.0];
    for h in hs.iter().copied() {
        assert_eq!(cubic_kernel_r(h, h), 0.0);
        assert_eq!(cubic_kernel_r(2.0 * h, h), 0.0);
        assert_eq!(cubic_kernel_r(10.0 * h, h), 0.0);
    }
}

#[test]
fn test_cubic_kernel_r_integral() {
    use nalgebra::Vector3;

    let hs = [0.025, 0.1, 2.0];
    let n = 10;

    for h in hs.iter().copied() {
        let dr = h / (n as f64);
        let dvol = dr * dr * dr;

        let mut integral = 0.0;
        for i in -n..n {
            for j in -n..n {
                for k in -n..n {
                    let r_in = Vector3::new(i as f64, j as f64, k as f64) * dr;
                    let r_out = Vector3::new((i + 1) as f64, (j + 1) as f64, (k + 1) as f64) * dr;
                    let r = ((r_in + r_out) * 0.5).norm();

                    integral += dvol * cubic_kernel_r(r, h);
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
    pub fn new(n: usize, h: R) -> Self {
        let mut values = Vec::with_capacity(n);

        let compact_support = h;
        let compact_support_squared = compact_support * compact_support;

        let dr = compact_support_squared / R::from_usize(n).unwrap();
        for i in 0..n {
            let i_and_half = R::from_usize(i).unwrap() + R::from_f64(0.5).unwrap();
            let r_squared = dr * i_and_half;
            let r = r_squared.sqrt();
            values.push(cubic_kernel_r(r, h));
        }

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
    let kernel = DiscreteSquaredDistanceCubicKernel::new(n, h);

    // Test the pre-computed values using a linear stepping
    let dr = h / (n as f64);
    for i in 0..n {
        let r = (i as f64) * dr;
        let rr = r * r;

        let discrete = kernel.evaluate(rr);
        let continuous = cubic_kernel_r_f64(r, h);

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
