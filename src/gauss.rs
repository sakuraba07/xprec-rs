use super::d64;
use super::arith::*;
use super::circular::*;
use super::consts;

pub fn gauss_legendre(x: &mut [d64], w: &mut [d64])
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

fn gauss_chebyshev_theta(θ: &mut [d64])
{
    let n = θ.len();
    let fact = consts::PI / (n as f64);
    for i in 0..n {
        // goes from (npos-0.5)/n * pi back to 0.5 * pi/n
        θ[i] = ((n - i) as f64 - 0.5) * fact;
    }
}

fn legendre_theta_newton(n: i64, θ: &mut [d64], w: &mut [d64])
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

fn plx(ell: i64, x: d64) -> (d64, d64)
{
    assert!(ell >= 1);

    // P(0, x) = 1, P(1, x) = x
    let mut p0 = d64::from(1.0);
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

    const X5: [d64; 5] = [
        d64 {hi: -0.906179845938664,  lo: -2.909730553174891e-17},
        d64 {hi: -0.5384693101056831, lo: 1.6678154894696646e-17},
        d64 {hi: 0.0, lo: 0.0},
        d64 {hi: 0.5384693101056831, lo: -1.6678154894696646e-17},
        d64 {hi: 0.906179845938664,  lo: 2.909730553174891e-17}
        ];
    const W5: [d64; 5] = [
        d64 {hi: 0.23692688505618908, lo: 2.6149055638876413e-18},
        d64 {hi: 0.47862867049936647, lo: -2.8616217915821202e-18},
        d64 {hi: 0.5688888888888889, lo: 4.934324553889585e-19},
        d64 {hi: 0.47862867049936647, lo: -2.8616217915821202e-18},
        d64 {hi: 0.23692688505618908, lo: 2.6149055638876413e-18},
        ];

    #[test]
    fn test_leg5()
    {
        use isclose::assert_is_close;

        // check Legendre quad for n = 5
        let mut x: [d64; 5] = [d64::from(0.0); 5];
        let mut w: [d64; 5] = [d64::from(0.0); 5];

        gauss_legendre(&mut x, &mut w);
        for i in 0..5 {
            assert_is_close!(x[i], X5[i], abs_tol=d64::EPSILON.hi);
            assert_is_close!(w[i], W5[i], abs_tol=2.0*d64::EPSILON.hi);
        }
    }
}
