use core::f64;
use std::ops::{Add, Sub};

/// Type for compensated arithmetic.
#[derive(PartialEq, PartialOrd, Clone, Copy, Debug)]
pub struct Compensated<H, L> {
    hi: H,
    lo: L
}

/// Compensated f64 (emulated quad precision) type.
///
/// Emulates quadruple precision with a pair of doubles.  This roughly doubles
/// the mantissa bits (and thus squares the precision of double).  The range
/// is almost the same as double, with a larger area of denormalized numbers.
/// This is also called double-double arithmetic, compensated arithmetic, or
/// Dekker arithmetic.
///
/// The rough cost in floating point operations (fl) and relative error as
/// multiples of u² = 1.32e-32 (round-off error or half the machine epsilon) is
/// as follows:
///
///   | (op)       | f64 f64 | error | Df64 f64 | error | Df64 Df64 | error |
///   |------------|--------:|------:|--------:|------:|--------:|------:|
///   | add_fast   |    3 fl |   0u² |    7 fl |   2u² |   17 fl |   3u² |
///   | + -        |    6 fl |   0u² |   10 fl |   2u² |   20 fl |   3u² |
///   | *          |    2 fl |   0u² |    6 fl |   2u² |    9 fl |   4u² |
///   | /          |   3* fl |   1u² |   7* fl |   3u² |  28* fl |   6u² |
///   | reciprocal |   3* fl |   1u² |         |       |  19* fl | 2.3u² |
///   | sqrt       |   4* fl |   2u² |         |       |   8* fl |   4u² |
///
/// The error bounds are mostly tight analytical bounds (except for
/// divisions).[^1]  An asterisk indicates the need for one or two double
/// divisions, which are about an order of magnitude more expensive than
/// regular flops on a modern CPU.
///
/// The table can be distilled into two rules of thumb: double-double
/// arithmetic roughly doubles the number of significant digits at the cost of
/// a roughly 15x slowdown compared to double arithmetic.
///
/// [^1]: M. Joldes, et al., ACM Trans. Math. Softw. 44, 1-27 (2018) and
///      J.-M. Muller and L. Rideau, ACM Trans. Math. Softw. 48, 1, 9 (2022).
///      The flop count has been reduced by 3 for divisons/reciprocals.
///      In the case of double-double division, the bound is 10u² but largest
///      observed error is 6u². In double by double division, we expect u². We
///      report the largest observed error.
pub type Df64 = Compensated<f64, f64>;

/// Arithmetic with compensated errors.
///
/// This trait marks a type as a compensated arithmetic type. It maintains the
/// result of an arithmetic operation plus a compensate, which is the
/// difference of the operation inside `T` and the exact result (or at least
/// more accurate result.)
///
/// The following operations are required:
///
///  | (op)       | method                      | Also known as | Exact? |
///  |------------|-----------------------------|---------------|--------|
///  | `a + b`    | `::compensated_sum(a, b)`   | 2sum(a, b)    | yes    |
///  | `a - b`    | `::compensated_diff(a, b)`  | 2diff(a, b)   | yes    |
///  | `a * b`    | `::compensated_prod(a, b)`  | 2prod(a, b)   | yes    |
///  | `a / b`    | `::compensated_ratio(a, b)` |               | no     |
///  | `a.sqrt()` | `::compensated_sqrt(a, b)`  |               | no     |
///
/// For sum and difference, there are `unsafe` versions, which may be faster
/// because they may assume that the arguments are ordered by magnitude, i.e,
/// `a.abs() >= b.abs()`. This constraint can be slightly relaxed.[^1]
///
///  | (op)    | unsafe method                   | Also known as     | Exact? |
///  |---------|---------------------------------|-------------------|--------|
///  | `a + b` | `::compensated_fast_sum(a, b)`  | fast2sum(a, b)    | yes*   |
///  | `a - b` | `::compensated_fast_diff(a, b)` | fast2diff(a, b)   | yes*   |
///
/// **Warning**: Compensated arithmetic is not guaranteed to conform to IEEE
/// rules when it comes to infinities. One usually gets NaN in this case.
///
/// [^1]: J.-M. Muller and L. Rideau, ACM Trans. Math. Softw. 48, 1, 9 (2022).
pub trait CompensatedArithmetic<T> : From<T> + Into<T>
{
    /// type of the compensate.
    type Compensate;

    /// Return the compensate (lo part)
    ///
    /// Return the compensate, i.e., the difference of the current value and
    /// its `T` approximation, `self.into<T>()`.
    fn compensate(self: &Self) -> Self::Compensate;

    /// Add `a` and `b` while compensating exactly for the error.
    ///
    /// Adds two values in extended precision, where the result can be
    /// represented exactly. On floating point numbers, this is known as
    /// "2sum" or compensated summation.
    fn compensated_sum(a: T, b: T) -> Self;

    /// Subtract `b` from `a` while compensating exactly for the error.
    ///
    /// Subtracts two values in extended precision, where the result can be
    /// represented exactly.
    fn compensated_diff(a: T, b: T) -> Self;

    /// Multiply `a` with `b` while compensating exactly for the error.
    ///
    /// Multiplies two values in extended precision, where the result can be
    /// represented exactly.
    fn compensated_prod(a: T, b: T) -> Self;

    /// Divides `a` by `b` while compensating approximately for the error.
    ///
    /// Divides two values in extended precision.
    fn compensated_ratio(a: T, b: T) -> Self;

    /// Compensated square root operation
    ///
    /// Takes the square root and maintains correction term.
    fn compensated_sqrt(a: T) -> Self;

    /// Add `large` and `small` exactly, assuming `large.abs() >= small.abs()`.
    ///
    /// Adds a small value `small` to a large value `large` in extended
    /// precision, where the result can be represented exactly if `a` has
    /// larger magnitude than `b`. (For double-double arithmetic, this
    /// condition can be slightly relaxed.). On floating point numbers, this is
    /// known as Kahan summation or "fast2sum".
    ///
    /// **Safety**: you must make sure that large is indeed the larger number.
    unsafe fn compensated_fast_sum(large: T, small: T) -> Self {
        return Self::compensated_sum(large, small);
    }

    /// Subtract `small` from `large` exactly, assuming `large.abs() >= small.abs()`.
    ///
    /// Subtracts a small value `small` from a large value `large` in extended
    /// precision, where the result can be represented exactly if `a` has
    /// larger magnitude than `b`. (For double-double arithmetic, this
    /// condition can be slightly relaxed.).
    ///
    /// **Safety**: you must make sure that large is indeed the larger number.
    unsafe fn compensated_fast_diff(large: T, small: T) -> Self {
        return Self::compensated_diff(large, small);
    }
}
/// Addition under the assumption of ordered arguments.
pub trait AddFast<T = Self> : Add<T> {
    /// Add `small` to `self`, assuming `small.abs() <= self.abs()`.
    ///
    /// Add a small value `small` to `self`, assuming that `small` is
    /// smaller in magnitude than `self`. Under some specific circumstances,
    /// this may lead to more efficient code.
    ///
    /// **Safety**: you must make sure that `small` is indeed the smaller number.
    unsafe fn add_fast(self, small: T) -> Self::Output;
}

/// Subtraction under the assumption of ordered arguments.
pub trait SubFast<T = Self> : Sub<T>{
    /// Subtract `small` from `self`, assuming `small.abs() <= self.abs()`.
    ///
    /// Subtract a small value `small` from `self`, assuming that `small` is
    /// smaller in magnitude than `self`. Under some specific circumstances,
    /// this may lead to more efficient code.
    ///
    /// **Safety**: you must make sure that `small` is indeed the smaller number.
    unsafe fn sub_fast(self, small: T) -> Self::Output;
}

#[cfg(test)]
mod test_utils;
mod utils;

pub mod arith;
pub mod circular;
pub mod checks;
pub mod consts;
pub mod convert;
pub mod exp;
pub mod funcs;
pub mod gauss;
pub mod hyperbolic;
pub mod roots;
pub mod round;
mod traits;

// Convert to float
impl From<Df64> for f64 {
    fn from(src: Df64) -> f64 {
        src.hi
    }
}

// Convert from float
impl From<f64> for Df64 {
    fn from(src: f64) -> Df64 {
        Df64 {hi: src, lo: 0.0}
    }
}
