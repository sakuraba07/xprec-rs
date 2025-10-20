use std::ops::Add;

#[allow(non_camel_case_types)]
#[derive(PartialEq, PartialOrd, Clone, Copy)]
pub struct d64 {
    hi: f64,
    lo: f64
}

#[cfg(test)]
mod test_utils;

pub mod arith;
pub mod round;

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
