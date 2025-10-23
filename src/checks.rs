
use libm;
use isclose;
use super::d64;
use super::arith;
use super::consts;

#[inline]
pub fn ldexp(a: d64, n: i32) -> d64
{
    return d64 {hi: libm::ldexp(a.hi, n), lo: libm::ldexp(a.lo, n)};
}

#[inline]
pub fn scalbn(a: d64, n: i32) -> d64
{
    return d64 {hi: libm::scalbn(a.hi, n), lo: libm::scalbn(a.lo, n)};
}

#[inline]
pub fn ilogb(x: d64) -> i32
{
    return libm::ilogb(x.hi);
}

#[inline]
pub fn is_sign_negative(a: d64) -> bool
{
    return a.hi.is_sign_negative();
}

#[inline]
pub fn copysign(mag: d64, sgn: d64) -> d64
{
    // The sign is determined by the hi part, however, the sign of hi and lo
    // need not be the same, so we cannot merely broadcast copysign to both
    // parts.
    if is_sign_negative(mag) != is_sign_negative(sgn) {
        arith::neg_q(mag)
    } else {
        mag
    }
}

#[inline]
pub fn abs(x: d64) -> d64
{
    if x.hi.is_sign_negative() {
        arith::neg_q(x)
    } else {
        x
    }
}

#[inline]
pub fn min(a: d64, b: d64) -> d64
{
    // fmin considers NaN to be the largest number. (a <= b) is false with
    // either element being NaN, if a is NaN, then it is okay to return b;
    // but if b is NaN, we have to return a
    if a <= b || consts::is_nan(b) {
        a
    } else {
        b
    }
}

#[inline]
pub fn max(a: d64, b: d64) -> d64
{
    if a <= b || consts::is_nan(a) {
        b
    } else {
        a
    }
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

impl isclose::IsClose for d64 {
    type Tolerance = f64;
    const ZERO_TOL: f64 = 0.0;

    // A useful default absolute tolerance is one at the floor of the double
    // range. We also ignore denormal numbers.
    const ABS_TOL: f64 = d64::MIN_POSITIVE.hi;

    // A small multiple of the machine epsilon is the right default here. We
    // scale this by 3 because this is the largest error we observe from any
    // of the arithmetic operations.
    const REL_TOL: f64 = 3.0 * d64::EPSILON.hi;

    #[inline(always)]
    fn is_close_tol(&self, rhs: &d64, rel_tol: &f64, abs_tol: &f64) -> bool {
        return isclose_qq(*self, *rhs, *abs_tol, *rel_tol);
    }
}

#[cfg(test)]
mod test {
    use isclose::assert_is_close;

    use super::*;

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

        assert_is_close!(one, one + 3e-32);
    }

}
