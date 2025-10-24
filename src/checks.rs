use std::num::FpCategory;
use approx;
use crate::d64;
use crate::arith;

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

#[inline]
pub fn is_sign_negative(a: d64) -> bool
{
    return a.hi.is_sign_negative();
}

/// Checks that two d64 numbers are close.
///
/// Given two numbers `a` and `b`, returns true if they close together in
/// at least one of two ways:
///
///  - by absolute distance: `|a - b| <= atol`
///  - by relative distance: `|a - b| <= rtol * max(|a|, |b|)`
///
pub fn isclose_qq(a: d64, b: d64, atol: f64, rtol: f64) -> bool
{
    if a.hi.abs() > b.hi.abs() {
        let threshold = atol.max(rtol * a.hi.abs());
        let diff = arith::subfast_qq(b, a).hi;
        return diff.abs() <= threshold;
    } else {
        let threshold = atol.max(rtol * b.hi.abs());
        let diff = arith::subfast_qq(a, b).hi;
        return diff.abs() <= threshold;
    }
}

impl approx::AbsDiffEq for d64 {
    type Epsilon = f64;

    #[inline(always)]
    fn default_epsilon() -> Self::Epsilon {
        // A useful default absolute tolerance is one at the floor of the
        // double range, since otherwise it is not clear what the scale is.
        // We also ignore denormal numbers.
        return d64::MIN_POSITIVE.hi;
    }

    #[inline(always)]
    fn abs_diff_eq(&self, other: &Self, epsilon: f64) -> bool {
        return isclose_qq(*self, *other, epsilon, 0.0);
    }
}

impl approx::RelativeEq for d64 {
    #[inline(always)]
    fn default_max_relative() -> Self::Epsilon {
        // A small multiple of the machine epsilon is the right default here.
        // We scale this by 3 because this is the largest error we observe from
        // any of the arithmetic operations.
        return 3.0 * d64::EPSILON.hi;
    }

    #[inline(always)]
    fn relative_eq(&self, other: &Self, epsilon: f64, max_relative: f64) -> bool {
        return isclose_qq(*self, *other, epsilon, max_relative);
    }
}

impl approx::UlpsEq for d64 {
    #[inline(always)]
    fn default_max_ulps() -> u32 {
        // We use 3 because this is the largest error we observe from
        // any of the arithmetic operations.
        return 3;
    }

    #[inline(always)]
    fn ulps_eq(&self, other: &Self, epsilon: f64, max_ulps: u32) -> bool {
        let rtol = max_ulps as f64 * d64::EPSILON.hi;
        return isclose_qq(*self, *other, epsilon, rtol);
    }
}

#[cfg(test)]
mod test {
    use approx::assert_relative_eq;
    use super::*;
    use crate::funcs;

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
        check_class(funcs::ldexp(d64::from(1.1), d64::MIN_EXP), FpCategory::Normal);
        check_class(funcs::ldexp(d64::from(0.9), d64::MIN_EXP), FpCategory::Subnormal);

        // Check max
        check_class(d64::MAX, FpCategory::Normal);
        check_class((1.0 + d64::EPSILON) * d64::MAX, FpCategory::Infinite);
        check_class((1.0 + d64::EPSILON/8.0) * d64::MAX, FpCategory::Normal);

        // Check max exp
        check_class(funcs::ldexp(d64::from(0.9), d64::MAX_EXP), FpCategory::Normal);
        check_class(funcs::ldexp(d64::from(1.0), d64::MAX_EXP), FpCategory::Infinite);

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

    #[test]
    fn test_isclose()
    {
        let zero = d64::from(0.0);
        let one = d64::from(1.0);
        assert!(isclose_qq(zero, zero, 0.0, 0.0));
        assert!(isclose_qq(one, one + 1e-30, 1e-29, 0.0));
        assert!(isclose_qq(one, one + 1e-30, 0.0, 1e-29));
        assert!(!isclose_qq(one, one - 1e-30, 1e-31, 0.0));
        assert!(!isclose_qq(one, one - 1e-30, 0.0, 1e-31));

        assert_relative_eq!(one, one + 3e-32);
    }

}
