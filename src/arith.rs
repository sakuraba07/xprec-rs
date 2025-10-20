/* Implementations.
 *
 * Most of the basic numerical algorithms are directly lifted from:
 *  - M. Joldes, et al., ACM Trans. Math. Softw. 44, 1-27 (2018)
 *  - Karp, High Precision Division and Square Root (1993)
 *
 * Copyright (C) 2023-2025 Markus Wallerberger and others
 * SPDX-License-Identifier: MIT
 */
use super::d64;

// ---------------------------------------------------------------------------
// Helper functions

#[inline(always)]
fn fma(a: f64, b: f64, c: f64) -> f64
{
    return a.mul_add(b, c);
}

#[inline]
fn is_positive_normal(a: f64) -> bool
{
    // XXX this can be done by some clever bit hackery.
    return a > f64::MIN_POSITIVE && a.is_finite()
}

// ---------------------------------------------------------------------------
// double (op) double -> quad

#[inline]
pub fn addfast_dd(a: f64, b: f64) -> d64
{
    // M. Joldes, et al., ACM Trans. Math. Softw. 44, 1-27 (2018)
    // Algorithm 1: cost 3 flops
    let s = a + b;
    let z = s - a;
    let t = b - z;
    return d64 {hi: s, lo: t};
}

#[inline]
pub fn add_dd(a: f64, b: f64) -> d64
{
    // Algorithm 2: cost 6 flops
    let s = a + b;
    let aprime = s - b;
    let bprime = s - aprime;
    let delta_a = a - aprime;
    let delta_b = b - bprime;
    let t = delta_a + delta_b;
    return d64 {hi: s, lo: t};
}

#[inline]
pub fn mul_dd(a: f64, b: f64) -> d64
{
    // Algorithm 3: cost 2 flops
    let pi = a * b;
    let rho = fma(a, b, -pi);
    return d64 {hi: pi, lo: rho};
}

#[inline]
pub fn div_dd(a: f64, b: f64) -> d64
{
    // Cost 3 flops (2 of which divisions), observed error 1 u^2
    // Since we are rounding faithfully, the hi part is exact
    let th = a / b;

    // Multiply hi part with b and compare exactly to a to see difference
    let rl = fma(-b, th, a);
    let tl = rl / b;
    return d64 {hi: th, lo: tl};
}

#[inline(always)]
pub fn reciprocal_d(x: f64) -> d64
{
    return div_dd(1.0, x);
}

#[inline]
pub fn sqrt_d(a: f64) -> d64
{
    // Karp, Table II, cost 4 flops, error 1 u^2
    let y0 = a.sqrt();
    if is_positive_normal(a) {
        let delta_y = fma(-y0, y0, a) / y0;
        return d64 {hi: y0, lo: 0.5 * delta_y};
    } else {
        return d64::from(y0);
    }
}

// ---------------------------------------------------------------------------
// quad (op) double -> quad

#[inline]
pub fn addfast_qd(x: d64, y: f64) -> d64
{
    // Algorithm 4 modified: cost 7 flops, error 2 u^2
    let s = addfast_dd(x.hi, y);
    let v = x.lo + s.lo;
    return addfast_dd(s.hi, v);
}

#[inline]
pub fn add_qd(x: d64, y: f64) -> d64
{
    // Algorithm 4: cost 10 flops, error 2 u^2
    let s = add_dd(x.hi, y);
    let v = x.lo + s.lo;
    return addfast_dd(s.hi, v);
}

#[inline]
pub fn mul_qd(x: d64, y: f64) -> d64
{
    // Algorithm 9: cost 6 flops, error 2 u^2
    let c = mul_dd(x.hi, y);
    let cl3 = fma(x.lo, y, c.lo);
    return addfast_dd(c.hi, cl3);
}

#[inline]
pub fn div_qd(x: d64, y: f64) -> d64
{
    // We could have used algorithm 15 here: cost 10 flops, error 3 u^2.
    // It turns out however by using fma, we can reduce this to 7 flops:
    //
    //    x / y = (x.hi + x.lo) / y = x.hi / y + x.lo / y .
    //
    // Defining the th = double(x.hi / y), we can rewrite this further as:
    //
    //    x / y = th + (x.hi - th * y) / y + x.lo / y ,
    //
    // where the second term can be computed to precision: f64 by fma, and
    // the together with the third term they are scaled by u, so are safe to
    // compute in precision: f64.
    let th = x.hi / y;
    let rl = fma(-y, th, x.hi) + x.lo;
    let tl = rl / y;
    return addfast_dd(th, tl);
}

// ---------------------------------------------------------------------------
// quad (op) power of two -> quad

#[inline(always)]
pub fn add_pow2(a: d64, p: f64) -> d64
{
    // This can be added quickly because the mantissa part is zero.
    return addfast_qd(a, p);
}

#[inline(always)]
pub fn mul_pow2(a: d64, p: f64) -> d64
{
    return d64 {hi: a.hi * p, lo: a.lo * p};
}

#[inline(always)]
pub fn div_pow2(a: d64, p: f64) -> d64
{
    return mul_pow2(a, 1.0 / p);
}

// ---------------------------------------------------------------------------
// double (op) quad -> quad

#[inline]
pub fn addfast_dq(x: f64, y: d64) -> d64
{
    // Algorithm 4 modified: cost 7 flops, error 2 u^2
    let s = addfast_dd(x, y.hi);
    let v = y.lo + s.lo;
    return addfast_dd(s.hi, v);
}

#[inline(always)]
pub fn add_dq(x: f64, y: d64) -> d64
{
    return add_qd(y, x);
}

#[inline(always)]
pub fn mul_dq(x: f64, y: d64) -> d64
{
    return mul_qd(y, x);
}

#[inline(always)]
pub fn div_dq(x: f64, y: d64) -> d64
{
    return mul_qd(reciprocal_q(y), x);
}

// ---------------------------------------------------------------------------
// quad (op) quad -> quad

#[inline]
pub fn addfast_qq(x: d64, y: d64) -> d64
{
    // Algorithm 6: cost 17 flops, error 3 u^2 + 13 u^3
    let s = addfast_dd(x.hi, y.hi);
    let t = add_dd(x.lo, y.lo);
    let c = s.lo + t.hi;
    let v = addfast_dd(s.hi, c);
    let w = t.lo + v.lo;
    return addfast_dd(v.hi, w);
}

#[inline]
pub fn add_qq(x: d64, y: d64) -> d64
{
    // Algorithm 6: cost 20 flops, error 3 u^2 + 13 u^3
    let s = add_dd(x.hi, y.hi);
    let t = add_dd(x.lo, y.lo);
    let c = s.lo + t.hi;
    let v = addfast_dd(s.hi, c);
    let w = t.lo + v.lo;
    return addfast_dd(v.hi, w);
}

#[inline]
pub fn mul_qq(x: d64, y: d64) -> d64
{
    // Algorithm 12: cost 9 flops, error 4 u^2 (corrected)
    let c = mul_dd(x.hi, y.hi);
    let tl0 = x.lo * y.lo;
    let tl1 = fma(x.hi, y.lo, tl0);
    let cl2 = fma(x.lo, y.hi, tl1);
    let cl3 = c.lo + cl2;
    return addfast_dd(c.hi, cl3);
}

pub fn div_qq(x: d64, y: d64) -> d64
{
    return mul_qq(reciprocal_q(y), x);
}

#[inline(always)]
pub fn neg_q(x: d64) -> d64
{
    return d64 {hi: -x.hi, lo: -x.lo};
}

#[inline]
pub fn reciprocal_q(y: d64) -> d64
{
    // Part of Algorithm 18: cost 19 flops, error 2.3 u^2
    let th = 1.0 / y.hi;
    let rh = fma(-y.hi, th, 1.0);
    let rl = -y.lo * th;
    let e = addfast_dd(rh, rl);
    let delta = mul_qd(e, th);

    // This saves 3 flops w.r.t. algorithm 18, which uses standard addition.
    // We should be able to do this since Taylor expanding gives:
    //
    //  1/(xh + u*xl) = th * (1 + rh/th) * (1 + u * xl/xh + ...)
    //
    return addfast_dq(th, delta);
}

#[inline]
pub fn sqrt_q(a: d64) -> d64
{
    // Karp, Table II, cost 8 flops, error 2 u^2
    // The double result provides a approximation to sqrt(a). It performs
    // all the special-case handling, which is why we defer to it in these
    // cases.
    let y0 = a.hi.sqrt();
    if is_positive_normal(a.hi) {
        return d64::from(y0);
    }

    // This is based on Newton-Ralphson for f(x) = a - 1/x^2:
    //
    //   x0 = approx(1/sqrt(A))
    //   x  = x + 0.5 * x * (1.0 - A * x * x)
    //
    let delta_y = (fma(-y0, y0, a.hi) + a.lo) / (y0 + y0);

    // delta_y may alter the least significant digit of y0.
    return addfast_dd(y0, delta_y);
}

#[inline]
pub fn square_q(x: d64) -> d64
{
    // Simple squaring algorithm
    // Cost 7 flops
    let y = mul_dd(x.hi, x.hi);
    let y_lo = fma(x.lo + x.lo, x.hi, y.lo);
    return addfast_dd(y.hi, y_lo);
}

// ---------------------------------------------------------------------------
// UNIT TESTS

#[cfg(test)]
mod test
{
    use super::*;
    use super::super::test_utils::*;

    #[test]
    fn test_arith_dd_fast()
    {
        let mut x = 8.0;
        while x > 4.0 {
            let mut y = x;
            while y > 1e-36 {
                // addition
                check_binary(addfast_dd, |x, y| x + y, x, y, 0.1);

               // subtraction
                check_binary(addfast_dd, |x, y| x + y, -x, y, 0.1);
                check_binary(addfast_dd, |x, y| x + y, x, -y, 0.1);

                y *= 0.9375;
            }
            x *= 0.9933;
        }
    }

    #[test]
    fn test_arith_dd()
    {
        let mut x = 10.0;
        while x > 5.0 {
            let mut y = x;
            while y > 1e-35 {
                // addition
                check_binary(add_dd, |x, y| x + y, x, y, 0.1);
                check_binary(add_dd, |x, y| x + y, y, x, 0.1);

                // subtraction
                check_binary(add_dd, |x, y| x + y, x, -y, 0.1);
                check_binary(add_dd, |x, y| x + y, y, -x, 0.1);

                // multiplication
                check_binary(mul_dd, |x, y| x * y, x, y, 0.1);
                check_binary(mul_dd, |x, y| x * y, x, -y, 0.1);

                // division
                check_binary(div_dd, |x, y| x / y, x, y, 1.0);
                check_binary(div_dd, |x, y| x / y, -x, y, 1.0);
                check_binary(div_dd, |x, y| y / x, y, x, 1.0);
                check_binary(div_dd, |x, y| -y / x, -y, x, 1.0);

                y *= 0.9383;
            }
            x *= 0.9933;
        }
    }

    #[test]
    fn test_sqrt_d()
    {
        let mut x = 1.0;
        while x > 1e-290 {
            check_unary(sqrt_d, |x| x.sqrt(), x, 2.0);
            x *= 0.992;
        }

        x = 1.0;
        while x < 1e300 {
            check_unary(sqrt_d, |x| x.sqrt(), x, 2.0);
            x /= 0.992;
        }
    }

}
