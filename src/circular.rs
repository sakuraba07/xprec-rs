use std::f64;
use super::utils::reciprocal_factorial;
use super::d64;
use super::arith::*;

pub fn sin(x: d64) -> d64
{
    let (sector, z) = reduce_mod_pi2(x);
    match sector {
        0 => sin_kernel(z, 7, 13),
        1 => cos_kernel(z, 8, 13),      // sin(x) = cos(x - pi/2)
        2 => -sin_kernel(z, 7, 13),     // sin(x) = -sin(x - pi)
        3 => -cos_kernel(z, 8, 13),     // sin(x) = -cos(x + pi/2)
        _ => panic!("illegal sector")
    }
}

pub fn cos(x: d64) -> d64
{
    let (sector, z) = reduce_mod_pi2(x);
    match sector {
        0 => cos_kernel(z, 8, 13),
        1 => -sin_kernel(z, 7, 13),
        2 => -cos_kernel(z, 8, 13),
        3 => sin_kernel(z, 7, 13),
        _ => panic!("illegal sector")
    }
}

pub fn sincos(x: d64) -> (d64, d64)
{
    let (sector, z) = reduce_mod_pi2(x);
    match sector {
        0 => (sin_kernel(z, 7, 13),   cos_kernel(z, 8, 13)),
        1 => (cos_kernel(z, 8, 13),  -sin_kernel(z, 7, 13)),
        2 => (-sin_kernel(z, 7, 13), -cos_kernel(z, 8, 13)),
        3 => (-cos_kernel(z, 8, 13),  sin_kernel(z, 7, 13)),
        _ => panic!("illegal sector")
    }
}

pub fn tan(x: d64) -> d64
{
    let (s, c) = sincos(x);
    return s / c;
}

fn reduce_mod_pi2(x: d64) -> (i32, d64)
{
    const INV_PI_HALF: f64 = 2.0 / f64::consts::PI;
    const PI_HALF: f64 = 1.5707963267948966;
    const PI_HALF_CORR: d64 =
            d64 {hi: 6.123233995736766e-17, lo: -1.4973849048591698e-33};

    // Approximate reduction
    let n = (INV_PI_HALF * x.hi).round();
    let sector = (n as i32) & 0b11;

    // Do not perform reduction if no-op
    if n == 0.0 {
        return (0, x);
    }

    // The reduction is done in sextuple precision. This is slow but relatively
    // accurate.
    let z0 = subfast_qq(x, mul_dd(n, PI_HALF));
    let z = subfast_qq(z0, n * PI_HALF_CORR);
    return (sector, z);
}

fn sin_kernel(x: d64, nquad: i32, n: i32) -> d64
{
    // Taylor series of the sin around 0
    assert!(x.hi.abs() <= 1.0);
    let xsq = -square_q(x);

    // r = x
    let mut r = x;
    let mut xpow = x;

    // r += x * (-x*x)**(i) / (2i+1)!
    for i in 1..nquad+1 {
        xpow *= xsq;
        r = addfast_qq(r, reciprocal_factorial(2*i + 1) * xpow);
    }

    // Here the terms are so small that they only affect the lo part, so
    // we can get away with double arithmetic.
    let xsq_d = xsq.hi;
    let mut xpow_d = xpow.hi;
    let mut r_d = 0.0;
    for i in nquad+1..n+1 {
        xpow_d *= xsq_d;
        r_d += reciprocal_factorial(2*i + 1).hi * xpow_d;
    }

    // Add results together
    r = addfast_qd(r, r_d);
    return r;
}

fn cos_kernel(x: d64, nquad: i32, n: i32) -> d64
{
    // Taylor series of the sin around 0
    assert!(x.hi.abs() <= 1.0);
    let xsq = -square_q(x);

    // r = 1 - x*x / 2
    let mut r = addfast_dq(1.0, mul_pow2(xsq, 0.5));
    let mut xpow = xsq;

    // r += (-x*x)**(i+1) / (2i)!
    for i in 2..nquad+1 {
        xpow *= xsq;
        r = addfast_qq(r, reciprocal_factorial(2*i) * xpow);
    }

    // Here the terms are so small that they only affect the lo part, so
    // we can get away with double arithmetic.
    let xsq_d = xsq.hi;
    let mut xpow_d = xpow.hi;
    let mut r_d = 0.0;
    for i in nquad+1..n+1 {
        xpow_d *= xsq_d;
        r_d += reciprocal_factorial(2*i).hi * xpow_d;
    }

    // Add results together
    r = addfast_qd(r, r_d);
    return r;
}

// fn _asin(x: d64) -> d64
// {
//     // Compute a approximation to double precision
//     let y0 = std::asin(x.hi());
//     if (!isfinite(y0))
//         return y0;

//     // This is where Taylor fails
//     if (fabs(x) == 1.0) {
//         return copysign(xprec::numbers::pi_half, x);
//     }

//     // Perform Taylor expansion:
//     //
//     //    asin(x) = asin(x0) + (x - x0) / sqrt(1 - x0**2)
//     //            = y0 + (x - sin(y0)) / cos(y0)
//     //
//     x0: d64, w;
//     sincos(y0, x0, w);

//     let y = y0 + (x - x0) / w;
//     return y;
// }

// fn _acos(x: d64) -> d64
// {
//     // Compute a approximation to double precision
//     let y0 = std::acos(x.hi());
//     if (!isfinite(y0))
//         return y0;

//     // This is where Taylor fails
//     if (x == 1.0)
//         return 0.0;
//     if (x == -1.0)
//         return xprec::numbers::pi;

//     // Perform Taylor expansion:
//     //
//     //    acos(x) = acos(x0) - (x - x0) / sqrt(1 - x0**2)
//     //            = y0 - (x - cos(y0)) / sin(y0)
//     //
//     x0: d64, w, diff;

//     sincos(y0, w, x0);
//     diff = (x0 - x) / w;
//     y0 += diff;
//     return y0;
// }

// fn _atan(x: d64) -> d64
// {
//     // For large values, use reflection formula
//     if (std::fabs(x.hi()) > 1.0) {
//         let y = copysign(xprec::numbers::pi_half, x);
//         if (isfinite(x))
//             y -= atan(reciprocal(x));
//         return y;
//     }

//     // Again use Taylor expansion
//     let y0 = std::atan(x.hi());
//     if (!isfinite(y0))
//         return y0;

//     s: d64, c, x0;

//     sincos(y0, s, c);
//     x0 = s / c;
//     y0 += (x - x0) * square(c);

//     return y0;
// }

// fn _atan2(y: d64, x: d64) -> d64
// {
//     using xprec::numbers::pi;
//     using xprec::numbers::pi_half;

//     // Special values
//     if (isnan(x) || isnan(y))
//         return NAN;
//     if (iszero(y))
//         return x.hi() >= 0 ? 0.0 : pi;
//     if (iszero(x))
//         return copysign(pi_half, y);

//     let res = atan(y / x);
//     if (x.hi() < 0)
//         res = copysign(pi, y).add_small(res);
//     return res;
// }

// extern "C" XPREC_API_EXPORT
// xprec_ddouble xprec_atan2(xprec_ddouble x, xprec_ddouble y)
// {
//     return _atan2(x, y);
// }

#[cfg(test)]
mod test {
    use super::*;
    use super::super::test_utils::*;

    #[test]
    fn test_kernels()
    {
        // small values, start from PI/4
        let mut x = d64::from(f64::consts::PI / 4.0);
        while x.hi > 1e-290 {
            check_unary(|x| sin_kernel(x, 7, 13), |x| x.sin(), x, 1.1);
            check_unary(|x| sin_kernel(x, 7, 13), |x| x.sin(), -x, 1.1);
            check_unary(|x| cos_kernel(x, 8, 13), |x| x.cos(), x, 1.1);
            check_unary(|x| cos_kernel(x, 8, 13), |x| x.cos(), -x, 1.1);
            x *= 0.947;
        }
    }

    #[test]
    fn test_circ()
    {
        // small values, start from PI/4
        let mut x = d64::from(f64::consts::PI / 4.0);
        while x.hi > 1e-290 {
            check_unary(sin, |x| x.sin(),  x, 1.1);
            check_unary(sin, |x| x.sin(), -x, 1.1);
            check_unary(cos, |x| x.cos(),  x, 1.1);
            check_unary(cos, |x| x.cos(), -x, 1.1);
            check_unary(tan, |x| x.tan(),  x, 2.0);
            check_unary(tan, |x| x.tan(), -x, 2.0);
            x *= 0.947;
        }

        // larger values
        x = d64::from(f64::consts::PI / 4.0);
        while x.hi < 100.0 {
            let magn = x.hi.abs().max(1.0);
            check_unary(sin, |x| x.sin(),  x, 1.5 * magn);
            check_unary(sin, |x| x.sin(), -x, 1.5 * magn);
            check_unary(cos, |x| x.cos(),  x, 1.5 * magn);
            check_unary(cos, |x| x.cos(), -x, 1.5 * magn);
            check_unary(tan, |x| x.tan(),  x, 2.5 * magn);
            check_unary(tan, |x| x.tan(), -x, 2.5 * magn);
            x /= 0.947;
        }
    }
}