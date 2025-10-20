#[allow(non_camel_case_types)]
pub struct d64 {
    pub hi: f64,
    pub lo: f64
}

pub mod arith;

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

