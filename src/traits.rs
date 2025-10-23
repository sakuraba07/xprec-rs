/* Traits.
 *
 * Copyright (C) 2023-2025 Markus Wallerberger and others
 * SPDX-License-Identifier: MIT
 */
use super::{d64, AddFast, SubFast, CompensatedArithmetic};
use super::{arith, round};
use std::ops::*;
use num_traits::*;

// ---------------------------------------------------------------------------
// STANDARD TRAITS

/// Macro for implementing binary operation traits
///
/// Implements traits `Trait` for a binary operation `func` for the following
/// combination of types
///
///   - `$op_qq: fn(d64, d64) -> d64` ... `$Trait for d64`
///   - `$op_qd: fn(d64, f64) -> d64` ... `$Trait<f64> for d64`
///   - `$op_dq: fn(f64, d64) -> d64` ... `$Trait<d64> for f64`
///
macro_rules! binary_op
{
    ($Trait:ident, $func:ident, $op_qq:expr, $op_qd:expr, $op_dq:expr) => {
        // Implementation for d64 (op) d64 -> d64
        impl $Trait for d64 {
            type Output = d64;
            fn $func(self, b: d64) -> d64 {
                return $op_qq(self, b);
            }
        }

        // Implementation for d64 (op) f64 -> d64
        impl $Trait<f64> for d64 {
            type Output = d64;
            fn $func(self, b: f64) -> d64 {
                return $op_qd(self, b);
            }
        }

        // Implementation for f64 (op) d64 -> d64
        impl $Trait<d64> for f64 {
            type Output = d64;
            fn $func(self, b: d64) -> d64 {
                return $op_dq(self, b);
            }
        }
    };
}

binary_op!(Add, add, arith::add_qq, arith::add_qd, arith::add_dq);
binary_op!(Sub, sub, arith::sub_qq, arith::sub_qd, arith::sub_dq);
binary_op!(Mul, mul, arith::mul_qq, arith::mul_qd, arith::mul_dq);
binary_op!(Div, div, arith::div_qq, arith::div_qd, arith::div_dq);
binary_op!(Rem, rem, round::mod_qq, round::mod_qd, round::mod_dq);
binary_op!(AddFast, add_fast, arith::addfast_qq, arith::addfast_qd, arith::addfast_dq);
binary_op!(SubFast, sub_fast, arith::subfast_qq, arith::subfast_qd, arith::subfast_dq);

/// Macro for implementing in-place operation traits
///
/// Implements traits `Trait` for a inplace operation `func` for the following
/// combination of types
///
///   - `$op_qq: fn(d64, d64) -> d64` ... `$Trait for d64`
///   - `$op_qd: fn(d64, f64) -> d64` ... `$Trait<f64> for d64`
///
macro_rules! inplace_op
{
    ($Trait:ident, $func:ident, $op_qq:expr, $op_qd:expr) => {
        impl $Trait for d64 {
            fn $func(&mut self, other: d64) {
                *self = $op_qq(*self, other);
            }
        }
        impl $Trait<f64> for d64 {
            fn $func(&mut self, other: f64) {
                *self = $op_qd(*self, other);
            }
        }
    };
}

inplace_op!(AddAssign, add_assign, arith::add_qq, arith::add_qd);
inplace_op!(SubAssign, sub_assign, arith::sub_qq, arith::sub_qd);
inplace_op!(MulAssign, mul_assign, arith::mul_qq, arith::mul_qd);
inplace_op!(DivAssign, div_assign, arith::div_qq, arith::div_qd);
inplace_op!(RemAssign, rem_assign, round::mod_qq, round::mod_qd);

/// Macro for implementing unary operation traits
///
/// Implements traits `Trait` for a inplace operation `func` for the following
/// type:
///
///   - `$op_q: fn(d64) -> d64` ... `$Trait for d64`
///
macro_rules! unary_op
{
    ($Trait:ident, $func:ident, $op_q:expr) => {
        impl $Trait for d64 {
            type Output = d64;
            fn $func(self) -> d64 {
                return $op_q(self);
            }
        }
    }
}

unary_op!(Neg, neg, arith::neg_q);

// ---------------------------------------------------------------------------
// COMPENSATE

impl CompensatedArithmetic<f64> for d64 {
    type Compensate = f64;

    const ZERO: d64 = d64 {hi: 0.0, lo: 0.0};
    const ONE: d64 = d64 {hi: 0.0, lo: 0.0};

    #[inline(always)]
    fn compensated_sum(a: f64, b: f64) -> d64 {
        return arith::add_dd(a, b);
    }

    #[inline(always)]
    fn compensated_diff(a: f64, b: f64) -> d64 {
        return arith::sub_dd(a, b);
    }

    #[inline(always)]
    fn compensated_prod(a: f64, b: f64) -> d64 {
        return arith::mul_dd(a, b);
    }

    #[inline(always)]
    fn compensated_ratio(a: f64, b: f64) -> d64 {
        return arith::div_dd(a, b);
    }

    #[inline(always)]
    fn compensated_sqrt(a: f64) -> d64 {
        return arith::sqrt_d(a);
    }

    #[inline(always)]
    unsafe fn compensated_fast_sum(a: f64, b: f64) -> d64 {
        return arith::addfast_dd(a, b);
    }

    #[inline(always)]
    unsafe fn compensated_fast_diff(a: f64, b: f64) -> d64 {
        return arith::subfast_dd(a, b);
    }

    #[inline(always)]
    fn compensate(self: &d64) -> f64 {
        return self.lo;
    }
}

// ---------------------------------------------------------------------------
// NUMERIC TRAITS

impl Zero for d64 {
    fn zero() -> d64 {
        return d64 {hi: 0.0, lo: 0.0};
    }
    fn is_zero(&self) -> bool {
        return self.hi == 0.0;
    }
}

impl One for d64 {
    fn one() -> d64 {
        return d64 {hi: 1.0, lo: 0.0};
    }
    fn is_one(&self) -> bool {
        return self.hi == 1.0 && self.lo == 0.0;
    }
}

impl Inv for d64 {
    type Output = d64;
    fn inv(self) -> d64 {
        return arith::reciprocal_q(self);
    }
}

// ---------------------------------------------------------------------------
// UNIT TESTS

#[cfg(test)]
mod test
{
    use super::*;

    #[test]
    fn test_traits()
    {
        let x = d64::from(1.0) * 2.0;
        let y = d64::from(1.0) / 4.0;
        assert_eq!(1.0 + x * y - 2.0, d64::from(-0.5));
    }

}