use super::d64;
use super::arith::*;
use super::circular::*;
use super::consts;

pub fn gauss_legendre(x: &mut [d64], w: &mut [d64])
{
    let n = x.len();
    gauss_chebyshev_theta(x);
    for _iter in 0..10 {
        legendre_theta_newton(n as i64, x);
    }
    for i in 0..n {
        x[i] = cos(x[i]);
    }
}

fn gauss_chebyshev_theta(x: &mut [d64])
{
    let n = x.len();
    let fact = consts::PI / (n as f64);
    for i in 0..n {
        // goes from (npos-0.5)/n * pi back to 0.5 * pi/n
        let theta = ((n - i) as f64 - 0.5) * fact;
        x[i] = theta;
    }
}

fn legendre_theta_newton(n: i64, x: &mut [d64])
{
    for i in 0..x.len() {
        let (s, c) = sincos(x[i]);
        let (pn_1, pn) = plx(n, c);
        let dn = (n as f64) * (c * pn - pn_1) / s;
        let dx = pn / dn;
        x[i] -= dx;
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
    use super::super::consts::*;

    #[test]
    fn test_leg()
    {
        let mut x: [d64; 5] = [d64::from(0.0); 5];
        let mut w: [d64; 5] = [d64::from(0.0); 5];
        gauss_legendre(&mut x, &mut w);

        //let wsum = w.into_iter().reduce(|acc, t| acc + t).unwrap();
        //println!("{:#?}", wsum);
        println!("{:#?}", x);
        //println!("{:#?}", w);
    }
}
