/* Implementations.
 *
 * Copyright (C) 2023-2025 Markus Wallerberger and others
 * SPDX-License-Identifier: MIT
 */
use super::d64;
use super::arith::*;

pub fn ceil(x: d64) -> d64
{
    // If hi was not an integer, it means that rounding it up/down/towards zero
    // already gives our answer.  This also covers NaN since then x != x.
    // We cannot simply truncate both hi and lo since they may have the same
    // sign
    let hi = x.hi.ceil();
    if hi != x.hi {
        return d64 {hi: hi, lo: 0.0};
    }

    // hi is an integer, so modify lo instead.  This may actually increase the
    // magnitude above the limit, so let's renormalize to be safe.
    let lo = x.lo.ceil();
    return addfast_dd(hi, lo);
}

pub fn floor(x: d64) -> d64
{
    // If hi was not an integer, it means that rounding it up/down/towards zero
    // already gives our answer.  This also covers NaN since then x != x.
    // We cannot simply truncate both hi and lo since they may have the same
    // sign
    let hi = x.hi.floor();
    if hi != x.hi {
        return d64 {hi: hi, lo: 0.0};
    }

    // hi is an integer, so modify lo instead.  This may actually increase the
    // magnitude above the limit, so let's renormalize to be safe.
    let lo = x.lo.floor();
    return addfast_dd(hi, lo);
}

pub fn trunc(x: d64) -> d64
{
    // If hi was not an integer, it means that rounding it up/down/towards zero
    // already gives our answer.  This also covers NaN since then x != x.
    // We cannot simply truncate both hi and lo since they may have opposite
    // signs.
    let hi = x.hi.trunc();
    if hi != x.hi {
        return d64 {hi: hi, lo: 0.0};
    }

    // hi is an integer, so modify lo instead.  Here, one needs to be careful
    // to respect the truncation direction that hi requires, and so we have
    // to round towards -+infinity, for x > 0 and x < 0, repectively.
    //
    // This may actually increase the  magnitude above the limit, so let's
    // renormalize to be safe.
    let lo = if x.hi < 0.0 {
        x.lo.ceil()
    } else {
        x.lo.floor()
    };
    return addfast_dd(hi, lo);
}

pub fn round(x: d64) -> d64
{
    // trunc is fast, so it makes sense to use this as a building block.
    let nudge = (0.5_f64).copysign(x.hi);
    return trunc(add_qd(x, nudge));
}
