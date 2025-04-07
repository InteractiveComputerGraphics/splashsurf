use anyhow::{Context, anyhow};
use nalgebra::{SVector, Scalar};
use num_traits::Zero;
use std::fmt::Debug;

/// Converts a slice of scalar values to a vector of the same length, returns an error if conversion fails
pub fn try_convert_scalar_slice<
    ScalarFrom: Copy + Debug,
    ScalarTo,
    F: Fn(ScalarFrom) -> Option<ScalarTo>,
>(
    values: &[ScalarFrom],
    f: F,
) -> Result<Vec<ScalarTo>, anyhow::Error> {
    values
        .iter()
        .copied()
        .map(|v| {
            f(v).ok_or_else(|| {
                anyhow!(
                    "failed to convert value {:?} from type {} to {}",
                    v,
                    std::any::type_name::<ScalarFrom>(),
                    std::any::type_name::<ScalarTo>()
                )
            })
        })
        .try_collect_with_capacity(values.len())
}

/// Converts a slice of scalar values to a vector of [`nalgebra::SVector`], returns an error if conversion fails or the input slice's length is not a multiple of the vector length.
pub fn try_convert_scalar_slice_to_vectors<
    const N: usize,
    ScalarFrom: Copy + Debug,
    ScalarTo: Scalar + Zero,
    F: Fn(ScalarFrom) -> Option<ScalarTo>,
>(
    values: &[ScalarFrom],
    f: F,
) -> Result<Vec<SVector<ScalarTo, N>>, anyhow::Error> {
    {
        if values.len() % N != 0 {
            Err(anyhow!("input slice length is not a multiple of {}", N))
        } else {
            values
                .chunks_exact(N)
                .map(|v| {
                    let mut v_out = SVector::zeros();
                    for i in 0..N {
                        v_out[i] = f(v[i]).ok_or_else(|| {
                            anyhow!(
                                "failed to convert value {:?} from type {} to {}",
                                v,
                                std::any::type_name::<ScalarFrom>(),
                                std::any::type_name::<ScalarTo>()
                            )
                        })?;
                    }
                    Ok(v_out)
                })
                .try_collect_with_capacity(values.len() / N)
        }
    }
    .context(anyhow!(
        "failed to convert scalar slice to vectors of length {}",
        N
    ))
}

#[cfg(test)]
mod tests {
    use num_traits::FromPrimitive;
    #[test]
    fn test_try_convert_scalar_slice() {
        let values = vec![1, -1];
        assert!(super::try_convert_scalar_slice(&values, u64::from_i32).is_err());
        let values = vec![1, -1];
        assert_eq!(
            super::try_convert_scalar_slice(&values, f32::from_i64).unwrap(),
            vec![1.0, -1.0]
        );
    }
}

/// "Convert" an empty vector to preserve allocated memory if size and alignment matches
/// See https://users.rust-lang.org/t/pattern-how-to-reuse-a-vec-str-across-loop-iterations/61657/5
/// See https://github.com/rust-lang/rfcs/pull/2802
#[allow(unused)]
pub(crate) fn recycle<A, B>(mut v: Vec<A>) -> Vec<B> {
    v.clear();
    v.into_iter().map(|_| unreachable!()).collect()
}

/// Useful extension methods for iterators
pub(crate) trait IteratorExt {
    /// Tries to collect the items of the iterator into a `Vec` that reserves the given capacity and stops as soon as an error is encountered
    ///
    /// Motivation: <https://github.com/rust-lang/rust/issues/48994>
    fn try_collect_with_capacity<T, E>(self, capacity: usize) -> Result<Vec<T>, E>
    where
        Self: Sized + Iterator<Item = Result<T, E>>;
}

impl<Iter: Iterator> IteratorExt for Iter {
    fn try_collect_with_capacity<T, E>(mut self, capacity: usize) -> Result<Vec<T>, E>
    where
        Self: Sized + Iterator<Item = Result<T, E>>,
    {
        self.try_fold(Vec::with_capacity(capacity), |mut vec, item| {
            vec.push(item?);
            Ok(vec)
        })
    }
}
