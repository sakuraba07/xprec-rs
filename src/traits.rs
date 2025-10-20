/* Traits.
 *
 * Copyright (C) 2023-2025 Markus Wallerberger and others
 * SPDX-License-Identifier: MIT
 */
use super::d64;
use super::arith;
use std::ops::*;

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