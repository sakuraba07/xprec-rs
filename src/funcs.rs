use libm;
use crate::d64;
use crate::{arith, checks};

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
pub fn copysign(mag: d64, sgn: d64) -> d64
{
    // The sign is determined by the hi part, however, the sign of hi and lo
    // need not be the same, so we cannot merely broadcast copysign to both
    // parts.
    if checks::is_sign_negative(mag) != checks::is_sign_negative(sgn) {
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
    if a <= b || checks::is_nan(b) {
        a
    } else {
        b
    }
}

#[inline]
pub fn max(a: d64, b: d64) -> d64
{
    if a <= b || checks::is_nan(a) {
        b
    } else {
        a
    }
}
