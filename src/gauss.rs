//! Gauss quadrature rules
//!
//! Algorithms are directly lifted from:
//!  - SIAM J. SCI. COMPUT., Vol. 35, No. 2, p. A652
//
// Copyright (C) 2023-2025 Markus Wallerberger and others
// SPDX-License-Identifier: MIT
use super::Df64;
use super::arith::*;
use super::circular::*;
use super::consts;

pub fn gauss_legendre(x: &mut [Df64], w: &mut [Df64])
{
    let n = x.len();
    gauss_chebyshev_theta(x);
    for _iter in 0..10 {
        legendre_theta_newton(n as i64, x, w);
    }
    for i in 0..n {
        x[i] = cos(x[i]);
    }
}

fn gauss_chebyshev_theta(θ: &mut [Df64])
{
    let n = θ.len();
    let fact = consts::PI / (n as f64);
    for i in 0..n {
        // goes from (npos-0.5)/n * pi back to 0.5 * pi/n
        θ[i] = ((n - i) as f64 - 0.5) * fact;
    }
}

fn legendre_theta_newton(n: i64, θ: &mut [Df64], w: &mut [Df64])
{
    // Newton iteration for theta rather than x
    // SIAM J. SCI. COMPUT., Vol. 35, No. 2, p. A652
    #[allow(non_snake_case)]
    for i in 0..θ.len() {
        let (s, c) = sincos(θ[i]);
        let (pn_1, pn) = plx(n, c);
        let pn_θ = (n as f64) * (c * pn - pn_1) / s;
        let Δθ = pn / pn_θ;
        θ[i] -= Δθ;
        w[i] = 2.0 / square_q(pn_θ);
    }
}

fn plx(ell: i64, x: Df64) -> (Df64, Df64)
{
    assert!(ell >= 1);

    // P(0, x) = 1, P(1, x) = x
    let mut p0 = Df64::ONE;
    let mut p1 = x;

    // Bonnet's recursion formula
    for k in 1..ell {
        // next term
        let kk = k as f64;
        let p2 = ((2.0 * kk + 1.0) * x * p1 - kk * p0) / (kk + 1.0);

        // shift terms by one
        p0 = p1;
        p1 = p2;
    }
    return (p0, p1);
}

#[cfg(test)]
mod test {
    use super::*;

    const X5: [Df64; 5] = [
        Df64 {hi: -0.906179845938664,  lo: -2.909730553174891e-17},
        Df64 {hi: -0.5384693101056831, lo: 1.6678154894696646e-17},
        Df64 {hi: 0.0, lo: 0.0},
        Df64 {hi: 0.5384693101056831, lo: -1.6678154894696646e-17},
        Df64 {hi: 0.906179845938664,  lo: 2.909730553174891e-17}
        ];
    const W5: [Df64; 5] = [
        Df64 {hi: 0.23692688505618908, lo: 2.6149055638876413e-18},
        Df64 {hi: 0.47862867049936647, lo: -2.8616217915821202e-18},
        Df64 {hi: 0.5688888888888889, lo: 4.934324553889585e-19},
        Df64 {hi: 0.47862867049936647, lo: -2.8616217915821202e-18},
        Df64 {hi: 0.23692688505618908, lo: 2.6149055638876413e-18},
        ];

    #[test]
    fn test_leg5()
    {
        use approx::assert_abs_diff_eq;

        // check Legendre quad for n = 5
        let mut x: [Df64; 5] = [Df64::ZERO; 5];
        let mut w: [Df64; 5] = [Df64::ZERO; 5];

        gauss_legendre(&mut x, &mut w);
        for i in 0..5 {
            assert_abs_diff_eq!(x[i], X5[i], epsilon=Df64::EPSILON);
            assert_abs_diff_eq!(w[i], W5[i], epsilon=2.0*Df64::EPSILON);
        }
    }

    const X7: [Df64; 7] = [
        Df64 {hi: -0.9491079123427585, lo: -3.82579658786657e-17},
        Df64 {hi: -0.7415311855993945, lo: 2.0220134774069897e-17},
        Df64 {hi: -0.4058451513773972, lo: 1.72492754475471e-17},
        Df64 {hi: 0.0, lo: 0.0},
        Df64 {hi: 0.4058451513773972, lo: -1.72492754475471e-17},
        Df64 {hi: 0.7415311855993945, lo: -2.0220134774069897e-17},
        Df64 {hi: 0.9491079123427585, lo: 3.82579658786657e-17}
    ];

    const W7: [Df64; 7] = [
        Df64 {hi: 0.1294849661688697, lo: -9.625448970284404e-18},
        Df64 {hi: 0.27970539148927664, lo: 2.3267180221717138e-17},
        Df64 {hi: 0.3818300505051189, lo: 2.1862747923824822e-17},
        Df64 {hi: 0.4179591836734694, lo: -1.5497807119257288e-17},
        Df64 {hi: 0.3818300505051189, lo: 2.1862747923824822e-17},
        Df64 {hi: 0.27970539148927664, lo: 2.3267180221717138e-17},
        Df64 {hi: 0.1294849661688697, lo: -9.625448970284404e-18}
    ];

    #[test]
    fn test_leg7()
    {
        use approx::assert_abs_diff_eq;

        let mut x: [Df64; 7] = [Df64::ZERO; 7];
        let mut w: [Df64; 7] = [Df64::ZERO; 7];

        gauss_legendre(&mut x, &mut w);
        for i in 0..7 {
            assert_abs_diff_eq!(x[i], X7[i], epsilon=Df64::from(Df64::EPSILON));
            assert_abs_diff_eq!(w[i], W7[i], epsilon=Df64::from(1.5 * Df64::EPSILON));
        }
    }

    const X16: [Df64; 16] = [
        Df64 {hi: -0.9894009349916499, lo: 5.914095566469922e-18},
        Df64 {hi: -0.9445750230732326, lo: 2.4190068142444825e-17},
        Df64 {hi: -0.8656312023878318, lo: 1.1315677979849837e-17},
        Df64 {hi: -0.755404408355003, lo: -3.5241085894430354e-17},
        Df64 {hi: -0.6178762444026438, lo: 2.2123521973463665e-17},
        Df64 {hi: -0.45801677765722737, lo:  -1.6662404170959257e-17},
        Df64 {hi: -0.2816035507792589, lo: 2.1958791252592132e-18},
        Df64 {hi: -0.09501250983763744, lo: 3.275947755433097e-19},
        Df64 {hi: 0.09501250983763744, lo: -3.275947755433097e-19},
        Df64 {hi: 0.2816035507792589, lo: -2.1958791252592132e-18},
        Df64 {hi: 0.45801677765722737, lo: 1.6662404170959257e-17},
        Df64 {hi: 0.6178762444026438, lo: -2.2123521973463665e-17},
        Df64 {hi: 0.755404408355003, lo: 3.5241085894430354e-17},
        Df64 {hi: 0.8656312023878318, lo: -1.1315677979849837e-17},
        Df64 {hi: 0.9445750230732326, lo: -2.4190068142444825e-17},
        Df64 {hi: 0.9894009349916499, lo: -5.914095566469922e-18}
    ];

    const W16: [Df64; 16] = [
        Df64 {hi: 0.027152459411754096, lo: -1.56154670271636e-18},
        Df64 {hi: 0.062253523938647894, lo: -7.690264522605704e-19},
        Df64 {hi: 0.09515851168249279, lo: -8.783003597087393e-19},
        Df64 {hi: 0.12462897125553388, lo: -4.841529802320495e-18},
        Df64 {hi: 0.14959598881657674, lo: -3.887619883741701e-18},
        Df64 {hi: 0.16915651939500254, lo: 2.323299329564479e-18},
        Df64 {hi: 0.18260341504492358, lo: 5.090226510905207e-18},
        Df64 {hi:0.1894506104550685, lo: -5.883843495582664e-18},
        Df64 {hi:0.1894506104550685, lo: -5.883843495582664e-18},
        Df64 {hi:0.18260341504492358, lo: 5.090226510905207e-18},
        Df64 {hi:0.16915651939500254, lo: 2.323299329564479e-18},
        Df64 {hi:0.14959598881657674, lo: -3.887619883741701e-18},
        Df64 {hi:0.12462897125553388, lo: -4.841529802320495e-18},
        Df64 {hi:0.09515851168249279, lo: -8.783003597087393e-19},
        Df64 {hi:0.062253523938647894, lo: -7.690264522605704e-19},
        Df64 {hi:0.027152459411754096, lo: -1.56154670271636e-18}
    ];

    #[test]
    fn test_leg16()
    {
        use approx::assert_abs_diff_eq;

        let mut x: [Df64; 16] = [Df64::ZERO; 16];
        let mut w: [Df64; 16] = [Df64::ZERO; 16];

        gauss_legendre(&mut x, &mut w);
        for i in 0..16 {
            assert_abs_diff_eq!(x[i], X16[i], epsilon=Df64::from(Df64::EPSILON));
            assert_abs_diff_eq!(w[i], W16[i], epsilon=Df64::from(Df64::EPSILON));
        }
    }
}
