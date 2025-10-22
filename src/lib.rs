use core::f64;

#[allow(non_camel_case_types)]
#[derive(PartialEq, PartialOrd, Clone, Copy, Debug)]
pub struct d64 {
    hi: f64,
    lo: f64
}


pub trait AddFast<T = Self> {
    type Output;

    // XXX: should maybe be unsafe?
    fn add_fast(self, small: T) -> Self::Output;
}

pub trait SubFast<T = Self> {
    type Output;

    // XXX: should maybe be unsafe?
    fn sub_fast(self, small: T) -> Self::Output;
}

pub trait InvSqrt {
    type Output;

    fn inv_sqrt(self) -> Self::Output;
}

#[cfg(test)]
mod test_utils;
mod utils;

pub mod arith;
pub mod circular;
pub mod checks;
pub mod consts;
pub mod exp;
pub mod hyperbolic;
pub mod roots;
pub mod round;
pub mod traits;

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
