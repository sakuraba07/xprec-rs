use super::d64;

impl d64 {
    /// Machine epsilon ϵ for d64.
    ///
    /// Lowest positive number `ϵ` such that for any normal number `x`,
    /// `(1 + ϵ) * x` is guaranteed to be distinct from `x`. Note that compared
    /// to IEEE floats (`f32`, `f64`), there are two complications:
    ///
    ///  1. arithmetic operations on d64 have an error bound of up to 3ϵ
    ///     rather than ϵ/2 in the case of regular floats.
    ///
    ///  2. `(1 + ϵ) * x` is not necessarily the next distinct d64 value.
    ///
    pub const EPSILON: d64 = d64 {
        hi: f64::EPSILON * f64::EPSILON / 2.0, lo: 0.0
    };

    /// Infinity (∞)
    pub const INFINITY: d64 = d64 {
        hi: f64::INFINITY, lo: 0.0
    };

    /// Negative infinity (∞)
    pub const NEG_INFINITY: d64 = d64 {
        hi: f64::NEG_INFINITY, lo: 0.0
    };

    /// Not a Number (NaN)
    ///
    /// Note that this is only one of the possible NaN values, and a quiet one.
    pub const NAN: d64 = d64 {
        hi: f64::NAN, lo: 0.0
    };

    /// Largest finite value.
    pub const MAX: d64 = d64 {
        hi: f64::MAX, lo: f64::MAX * f64::EPSILON / 4.0
    };

    /// Largest negative value by magnitude.
    pub const MIN: d64 = d64 {
        hi: f64::MIN, lo: f64::MIN * f64::EPSILON / 4.0
    };

    /// Smallest positive normal value.
    ///
    /// Note that d64 has a smaller range of normal numbers than f64, because
    /// it implies that both hi and lo part must be normal.
    pub const MIN_POSITIVE: d64 = d64 {
        hi: f64::MIN_POSITIVE / f64::EPSILON, lo: 0.0
    };

    /// The radix or base of the internal representation.
    pub const RADIX: u32 = f64::RADIX;

    /// Maximum exponent.
    ///
    /// Largest exponent `e` such that for any `m.abs() < 1`, `ldexp(m, e)` is
    /// a normal number, i.e., does not overflow.
    pub const MAX_EXP: i32 = f64::MAX_EXP;

    /// Minimum exponent.
    ///
    /// Smallest exponent `e` such that for any `m.abs() < 1`, `ldexp(m, e)` is
    /// a normal number, i.e., does not underflow or go into the subnormals.
    pub const MIN_EXP: i32 = f64::MIN_EXP + f64::MANTISSA_DIGITS as i32 - 2;

}

pub const PI: d64 = d64 {hi: 3.141592653589793, lo: 1.2246467991473532e-16};

pub const PI_HALF: d64 = d64 {hi: 1.5707963267948966, lo: 6.123233995736766e-17};




#[cfg(test)]
mod test
{
    // XXX
}