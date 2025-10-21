use super::d64;
use super::arith::*;
use super::checks::*;
use super::consts::*;
use super::roots::*;
use super::exp::*;

pub const COSH_MAX: f64 = 710.4758600739439;

pub fn cosh(x: d64) -> d64
{
    let xx = abs(x);
    if !(xx.hi <= COSH_MAX) {
        if is_nan(x) {
            return d64::NAN;
        } else {
            return d64::INFINITY;
        }
    }

    // simply use the definition:
    //
    //    cosh(x) = exp(x)/2 + exp(-x)/2 = exp(x)/2 + 1/4 * 2/exp(x)
    //
    let (m, y) = exp_split(xx);
    let exp_x_half = ldexp(addfast_dq(1.0, y), m - 1);

    // For large numbers, that's it.
    if !(xx.hi <= 36.5) {
        return exp_x_half;
    }

    let exp_mx_half = mul_pow2(reciprocal_q(exp_x_half), 0.25);
    return addfast_qq(exp_x_half, exp_mx_half);
}

pub fn sinh(x: d64) -> d64
{
    let xx = abs(x);
    if !(xx.hi <= COSH_MAX) {
        if is_nan(x) {
            return d64::NAN;
        } else {
            return copysign(d64::INFINITY, x);
        }
    }

    let (m, y) = exp_split(xx);

    // For small x, corresponding to exponent m = 0 of exp(x), naive
    // subtraction leads to cancellation. Let's instead rewrite. Let
    // exp(x) = 1 + y, then:
    //
    //    sinh(x) = y * (1 + 0.5 * y) / (1 + y)
    //
    if m == 0 {
        let dn = addfast_dq(1.0, mul_pow2(y, 0.5));
        let en = addfast_dq(1.0, y);
        return copysign(y * dn / en, x);
    }

    // Otherwise simply use the definition:
    //
    //    sinh(x) = exp(x)/2 - exp(-x)/2 = exp(x)/2 - 1/4 * 2/exp(x)
    //
    let exp_x_half = ldexp(addfast_dq(1.0, y), m - 1);

    // For large numbers, that's it.
    if !(xx.hi <= 36.5) {
        return copysign(exp_x_half, x);
    }

    let exp_mx_half = mul_pow2(reciprocal_q(exp_x_half), 0.25);
    return copysign(subfast_qq(exp_x_half, exp_mx_half), x);
}

pub fn tanh(x: d64) -> d64
{
    let xx = abs(x);
    if !(xx.hi <= 36.5) {
        if is_nan(x) {
            return d64::NAN;
        } else {
            // Asymptotically, we have +- 1
            return copysign(d64::from(1.0), x);
        }
    }

    // For small x, the naive tanh(x) leads to cancellation. Again, we rewrite
    // it. Let exp(x) = 1 + y, then:
    //
    //    tanh(x) = z / (2 + z)  with  z = y * (2 + y)
    //
    // This is stable along the entire range.
    let y = expm1(xx);
    let z = y * (2.0 + y);
    return copysign(z / (2.0 + z), x);
}

pub fn asinh(x: d64) -> d64
{
    // Special values: +Inf, -Inf are all preserved
    if !is_finite(x) {
        return x;
    }

    // For small values, use Taylor expansion around the double result,
    // because the bottom expression is log(1 + 2x/3 + ...), subject to
    // cancellation.
    if x.hi.abs() < 1.0 {
        let y0 = x.hi.asinh();
        let x0 = sinh(d64::from(y0));

        let delta_y = (x - x0) * inv_sqrt(1.0 + square_q(x0));
        return addfast_dq(y0, delta_y)
    }

    // Use the definition:
    //
    //     asinh(x) = log(sqrt(1 + x^2) + x)
    //
    let xx = abs(x);
    let arg = addfast_qq(hypot(d64::from(1.0), xx), xx);
    return copysign(log(arg), x);
}

pub fn acosh(x: d64) -> d64
{
    // Special values: +Inf, -Inf are all preserved
    if !is_finite(x) {
        return x;
    }

    // Use the definition:
    //
    //     acosh(x) = log(x + sqrt(x^2 - 1))
    //
    // but be careful of overflows.
    let arg = if x.hi < 1.0 / f64::EPSILON {
        addfast_qq(x, sqrt_q(subfast_qd(square_q(x), 1.0)))
    } else {
        mul_pow2(x, 2.0)
    };
    return log(arg);
}

pub fn atanh(x: d64) -> d64
{
    if is_nan(x) {
        return x;
    }

    // Special value
    let xx = abs(x);
    if xx == d64::from(1.0) {
        return copysign(d64::INFINITY, x);
    }

    // Use the definition, but be wary of cancellation around 0.
    //
    //   atanh(x) = 1/2 log((1 + x)/(1 - x)) = 1/2 log(1 + 2x/(1 - x))
    //
    let twox = mul_pow2(xx, 2.0);
    let one_minus_x = subfast_dq(1.0, xx);
    return copysign(mul_pow2(log1p(twox / one_minus_x), 0.5), x);
}

#[cfg(test)]
mod test{
    use super::*;
    use super::super::test_utils::*;

    #[test]
    fn test_cosh()
    {
        // special values
        assert!(is_infinite(cosh(d64::from(1000.0))));
        assert!(is_infinite(cosh(d64::INFINITY)));
        assert!(is_infinite(cosh(d64::NEG_INFINITY)));
        assert!(is_nan(cosh(d64::NAN)));

        // simple vals
        check_unary(cosh, |x| x.cosh(), d64::from(0.0), 1.0);

        // small values
        let mut x = d64::from(1.0);
        while x.hi > 1e-290 {
            check_unary(cosh, |x| x.cosh(), x, 1.0);
            check_unary(cosh, |x| x.cosh(), -x, 1.0);
            x *= 0.947;
        }

        // large values
        x = d64::from(1.0);
        while x.hi < COSH_MAX {
            check_unary(cosh, |x| x.cosh(), x, 1.0);
            check_unary(cosh, |x| x.cosh(), -x, 1.0);
            x *= 1.0041;
        }

        assert!(is_finite(cosh(d64::from(COSH_MAX))));
    }

    #[test]
    fn test_sinh()
    {
        // special values
        assert!(is_infinite(sinh(d64::from(1000.0))));
        assert!(is_infinite(sinh(d64::INFINITY)));
        assert!(is_infinite(sinh(d64::NEG_INFINITY)));
        assert!(sinh(d64::NEG_INFINITY) < d64::from(0.0));
        assert!(is_nan(sinh(d64::NAN)));

        // simple vals
        check_unary(sinh, |x| x.sinh(), d64::from(0.0), 1.0);

        // small values
        let mut x = d64::from(1.0);
        while x.hi > 1e-290 {
            check_unary(sinh, |x| x.sinh(), x, 2.0);
            check_unary(sinh, |x| x.sinh(), -x, 2.0);
            x *= 0.947;
        }

        // large values
        x = d64::from(1.0);
        while x.hi < COSH_MAX {
            check_unary(sinh, |x| x.sinh(), x, 1.0);
            check_unary(sinh, |x| x.sinh(), -x, 1.0);
            x *= 1.0041;
        }

        assert!(is_finite(sinh(d64::from(COSH_MAX))));
    }

    #[test]
    fn test_tanh()
    {
        // special values
        assert!(tanh(d64::from(1000.0)) == d64::from(1.0));
        assert!(tanh(d64::from(-1000.0)) == d64::from(-1.0));
        assert!(tanh(d64::INFINITY) == d64::from(1.0));
        assert!(tanh(-d64::INFINITY) == d64::from(-1.0));
        assert!(is_nan(tanh(d64::NAN)));

        // simple vals
        check_unary(tanh, |x| x.tanh(), d64::from(0.0), 1.0);

        // small values
        let mut x = d64::from(1.0);
        while x.hi > 1e-290 {
            check_unary(tanh, |x| x.tanh(), x, 2.0);
            check_unary(tanh, |x| x.tanh(), -x, 2.0);
            x *= 0.947;
        }

        // large values
        x = d64::from(1.0);
        while x.hi < 10.0 * COSH_MAX {
            check_unary(tanh, |x| x.tanh(), x, 2.0);
            check_unary(tanh, |x| x.tanh(), -x, 2.0);
            x *= 1.0041;
        }
    }

        #[test]
    fn test_arc()
    {
        // small values
        let mut x = d64::from(1.0);
        while x.hi > 1e-290 {
            check_unary(asinh, |x| x.asinh(), x, 2.0);
            check_unary(asinh, |x| x.asinh(), -x, 2.0);
            if x < d64::from(1.0) {
                check_unary(atanh, |x| x.atanh(), x, 2.5);
                check_unary(atanh, |x| x.atanh(), -x, 2.5);
            }
            x *= 0.91;
        }

        // large values
        // XXX not entire range covered
        x = d64::from(1.0);
        while x.hi < f64::MAX / 4.0 {
            check_unary(asinh, |x| x.asinh(), x, 2.0);
            check_unary(asinh, |x| x.asinh(), -x, 2.0);
            check_unary(acosh, |x| x.acosh(), x, 2.0);
            x *= 1.13;
        }
    }
}