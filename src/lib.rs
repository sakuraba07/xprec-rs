use std::ops::Add;

#[allow(non_camel_case_types)]
#[derive(PartialEq, PartialOrd, Clone)]
pub struct d64 {
    hi: f64,
    lo: f64
}

pub mod arith;

// Add arith::add_qq arith::add_qd

impl Add for d64 {
    type Output = d64;
    fn add(self, b: d64) -> d64 {
        arith::add_qq(self, b)
    }
}
impl Add<f64> for d64 {
    type Output = d64;
    fn add(self, b: f64) -> d64 {
        arith::add_qd(self, b)
    }
}

// Convert to float
impl From<d64> for f64 {
    fn from(src: d64) -> f64 {
        src.hi
    }
}

// Convert from float
impl From<f64> for d64 {
    fn from(src: f64) -> d64 {
        d64 {hi: src, lo: 0.0}
    }
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = d64::from(4.0);
        assert_eq!(f64::from(result), 4.0);
    }
}

