use super::d64;
use super::arith::*;

pub fn hypot(x: d64, y: d64) -> d64
{
    // XXX unfortunately, rust has no const floats expressions, so we need
    //     to hard-code this here
    const LARGE: f64 = 1.3407807929942597e154;
    const SMALL: f64 = 1.0 / LARGE;

    let x_magn = x.hi.abs();
    let y_magn = y.hi.abs();

    if x_magn >= y_magn {
        if x_magn >= LARGE {
            if x_magn.is_infinite() {
                return d64::INFINITY;
            }
            mul_pow2(_hypot(mul_pow2(x, SMALL), mul_pow2(y, SMALL)), LARGE)
        } else if x_magn < SMALL {
            mul_pow2(_hypot(mul_pow2(x, LARGE), mul_pow2(y, LARGE)), SMALL)
        } else {
            _hypot(x, y)
        }
    } else {
        if y_magn >= LARGE {
            if y_magn.is_infinite() {
                return d64::INFINITY;
            }
            mul_pow2(_hypot(mul_pow2(y, SMALL), mul_pow2(x, SMALL)), LARGE)
        } else if y_magn < SMALL {
            mul_pow2(_hypot(mul_pow2(y, LARGE), mul_pow2(x, LARGE)), SMALL)
        } else {
            _hypot(y, x)
        }
    }
}

#[inline]
fn _hypot(x: d64, y: d64) -> d64
{
    let x2 = square_q(x);
    let y2 = square_q(y);
    return sqrt_q(addfast_qq(x2, y2));
}

pub fn inv_sqrt(x: d64) -> d64
{
    // Use strategy similar to Karp to compute 1/sqrt(x)
    // cost 12 flops (3 of which divisions), observed error 2 u^2

    // First, give an approximation to sqrt(x)
    let sqrt_x0 = x.hi.sqrt();
    if x.hi <= 0.0 || !x.hi.is_finite() {
        return d64::from(1.0 / sqrt_x0);
    }

    // The correction term is then given by the lo part and the difference
    // to the exact sqrt
    let delta_x = (-sqrt_x0).mul_add(sqrt_x0, x.hi) + x.lo;

    // Compute 1/sqrt_x0 to quad precision
    let y0 = reciprocal_d(sqrt_x0);

    // Correct using first-order expansion
    //
    //  1/sqrt(x0 + delta_x) = 1/sqrt(x0) - delta_x / (2 * sqrt(x0)**3) + ...
    //
    // The correction term can be computed in double precision, but it is
    // important to use fma, as sqrt(x0)**3 may overflow.
    let y_lo = (-delta_x / (x.hi + x.hi)).mul_add(y0.hi, y0.lo);
    return addfast_dd(y0.hi, y_lo);
}

#[cfg(test)]
mod test {
    use super::*;
    use super::super::consts::*;
    use super::super::test_utils::*;

    #[test]
    fn test_hypot()
    {
        assert!(is_infinite(hypot(d64::INFINITY, d64::EPSILON)));
        assert!(is_infinite(hypot(d64::EPSILON, -d64::INFINITY)));
        assert!(is_infinite(hypot(d64::INFINITY, d64::INFINITY)));

        check_binary(hypot, |x,y| x.hypot(&y),
                     d64::from(1.0), d64::from(1.0), 1.5);
        check_binary(hypot, |x,y| x.hypot(&y),
                     d64::from(3.0), d64::from(-10000.0), 1.5);
        check_binary(hypot, |x,y| x.hypot(&y),
                     d64::from(1e249), d64::from(1e241), 1.5);
        check_binary(hypot, |x,y| x.hypot(&y),
                     d64::from(1e241), d64::from(-1e249), 1.5);
        check_binary(hypot, |x,y| x.hypot(&y),
                     d64::from(1e-251), d64::from(1e-259), 1.5);
        check_binary(hypot, |x,y| x.hypot(&y),
                     d64::from(-1e-259), d64::from(1e-248), 1.5);

        let mut x = d64::from(10.0);
        while x > d64::from(5.0) {
            let mut y = x;
            while y > d64::from(1e-35) {
                // fast addition
                check_binary(hypot, |x, y| x.hypot(&y), x, y, 1.5);
                y = mul_qd(y, 0.9383);
            }
            x = mul_qd(x, 0.9933);
        }
    }

    #[test]
    fn test_roots_q()
    {
        let mut x = d64::from(1.0);
        while x > d64::from(1e-290) {
            check_unary(inv_sqrt, |x| 1.0 / x.sqrt(), x, 2.0);
            x = mul_qd(x, 0.992);
        }

        x = d64::from(1.0);
        while x < d64::from(1e290) {
            check_unary(inv_sqrt, |x| 1.0 / x.sqrt(), x, 2.0);
            x = div_qd(x, 0.992);
        }
    }

}
