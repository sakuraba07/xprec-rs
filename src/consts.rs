use super::d64;
use std::num::FpCategory;

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

#[inline]
pub fn is_finite(x: d64) -> bool {
    return x.hi.is_finite();
}

#[inline]
pub fn is_infinite(x: d64) -> bool {
    return x.hi.is_infinite();
}

#[inline]
pub fn is_nan(x: d64) -> bool {
    return x.hi.is_nan();
}

#[inline]
pub fn is_normal(x: d64) -> bool
{
    // Denormalization is double-double is a bit of a strange concept,
    // since the lo part may be a denormalized number even if the whole
    // number is still "normal".
    return x.hi.is_normal() && (x.hi * f64::EPSILON).is_normal();
}

#[inline]
pub fn is_subnormal(x: d64) -> bool
{
    // Denormalization is double-double is a bit of a strange concept,
    // since the lo part may be a denormalized number even if the whole
    // number is still "normal".
    return x.hi.is_subnormal() || (x.hi * f64::EPSILON).is_subnormal();
}

#[inline]
pub fn is_zero(x: d64) -> bool {
    return x.hi == 0.0;
}

#[inline]
pub fn classify(x: d64) -> FpCategory
{
    // This also works with zero, since that can be determined from the
    // hi part alone
    return x.hi.classify();
}

#[cfg(test)]
mod test
{
    use super::*;
    use super::super::checks;

    fn check_class(x: d64, cat: FpCategory) {
        assert!(is_normal(x) == (cat == FpCategory::Normal));
        assert!(is_subnormal(x) == (cat == FpCategory::Subnormal));
        assert!(is_nan(x) == (cat == FpCategory::Nan));
        assert!(is_zero(x) == (cat == FpCategory::Zero));
        assert!(is_infinite(x) == (cat == FpCategory::Infinite));

        let cat_finite = cat != FpCategory::Infinite && cat != FpCategory::Nan;
        assert!(is_finite(x) == cat_finite);
    }

    #[test]
    fn test_class() {
        check_class(d64::from(-1.0) + d64::EPSILON, FpCategory::Normal);
        check_class(d64::EPSILON, FpCategory::Normal);

        // Check min
        check_class(d64::MIN, FpCategory::Normal);
        check_class((1.0 + d64::EPSILON) * d64::MIN, FpCategory::Infinite);
        check_class((1.0 + d64::EPSILON/8.0) * d64::MIN, FpCategory::Normal);

        // Check min exp
        check_class(checks::ldexp(d64::from(1.1), d64::MIN_EXP), FpCategory::Normal);
        check_class(checks::ldexp(d64::from(0.9), d64::MIN_EXP), FpCategory::Subnormal);

        // Check max
        check_class(d64::MAX, FpCategory::Normal);
        check_class((1.0 + d64::EPSILON) * d64::MAX, FpCategory::Infinite);
        check_class((1.0 + d64::EPSILON/8.0) * d64::MAX, FpCategory::Normal);

        // Check max exp
        check_class(checks::ldexp(d64::from(0.9), d64::MAX_EXP), FpCategory::Normal);
        check_class(checks::ldexp(d64::from(1.0), d64::MAX_EXP), FpCategory::Infinite);

        // Check min positive
        check_class(d64::MIN_POSITIVE, FpCategory::Normal);
        check_class((1.0 + f64::EPSILON) * d64::MIN_POSITIVE, FpCategory::Normal);
        check_class((1.0 - f64::EPSILON) * d64::MIN_POSITIVE, FpCategory::Subnormal);

        // Check nan
        check_class(d64::NAN, FpCategory::Nan);
        check_class(-d64::NAN, FpCategory::Nan);
        check_class(d64::NAN / d64::NAN, FpCategory::Nan);

        // check zero
        check_class(d64::from(0.0), FpCategory::Zero);
        check_class(d64::from(-0.0), FpCategory::Zero);
    }

}