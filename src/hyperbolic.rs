use super::d64;
use super::arith::*;
use super::checks::*;
use super::consts::*;
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
}