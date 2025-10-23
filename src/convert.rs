use crate::d64;
use crate::arith;
use num_traits;

macro_rules! convert {
    ($fname:ident d64 $Dest:ident) => {
        /// Convert d64 to $Dest, if target in range
        ///
        /// Converts `x` to integer type $Dest, if `x` is indeed in the range
        /// `$Dest::MIN <= x < $Dest::MAX + 1`. In converting, any fractional
        /// part is discarded.
        #[inline]
        pub fn $fname(x: d64) -> Option<$Dest>
        {
            const LOW: f64 = $Dest::MIN as f64;
            const HIGH: f64 = ($Dest::MAX as f64) + 1.0;
            if x.hi >= LOW && x.hi < HIGH {
                let xi = unsafe {
                    x.hi.to_int_unchecked::<$Dest>() +
                    x.lo.to_int_unchecked::<$Dest>()
                };
                return Some(xi);
            } else {
                return None;
            }
        }
    };
    ($fname:ident $Src:ident d64) => {
        /// Convert $Src to d64
        ///
        /// Converts the integer `x` of type $Src to compensated float. Takes
        /// care to preserve the least significant bits of i64, which are
        /// typically truncated in f64.
        #[inline]
        pub fn $fname(x: $Src) -> d64
        {
            const SIZEOF_SRC: usize = size_of::<$Src>();
            if SIZEOF_SRC == 8 {
                const HI_HALF: $Src = !(0 as $Src) << (4 * SIZEOF_SRC);
                const LO_HALF: $Src = !HI_HALF;
                return arith::addfast_dd(
                            (x & HI_HALF) as f64, (x & LO_HALF) as f64);
            } else {
                return d64 {hi: x as f64, lo: 0.0};
            }
        }
    };
}

// Instantiate all conversion routines

convert!(try_to_isize d64   isize);
convert!(try_to_i8    d64   i8);
convert!(try_to_i16   d64   i16);
convert!(try_to_i32   d64   i32);
convert!(try_to_i64   d64   i64);
convert!(try_to_i128  d64   i128);

convert!(try_to_usize d64   usize);
convert!(try_to_u8    d64   u8);
convert!(try_to_u16   d64   u16);
convert!(try_to_u32   d64   u32);
convert!(try_to_u64   d64   u64);
convert!(try_to_u128  d64   u128);

// XXX add u128/i128

convert!(from_isize   isize d64);
convert!(from_i8      i8    d64);
convert!(from_i16     i16   d64);
convert!(from_i32     i32   d64);
convert!(from_i64     i64   d64);

convert!(from_usize   usize d64);
convert!(from_u8      u8    d64);
convert!(from_u16     u16   d64);
convert!(from_u32     u32   d64);
convert!(from_u64     u64   d64);

impl num_traits::ToPrimitive for d64 {
    #[inline] fn to_isize(&self) -> Option<isize> { return try_to_isize(*self); }
    #[inline] fn to_i8(&self)    -> Option<i8>    { return try_to_i8(*self); }
    #[inline] fn to_i16(&self)   -> Option<i16>   { return try_to_i16(*self); }
    #[inline] fn to_i32(&self)   -> Option<i32>   { return try_to_i32(*self); }
    #[inline] fn to_i64(&self)   -> Option<i64>   { return try_to_i64(*self); }
    #[inline] fn to_i128(&self)  -> Option<i128>  { return try_to_i128(*self); }

    #[inline] fn to_usize(&self) -> Option<usize> { return try_to_usize(*self); }
    #[inline] fn to_u8(&self)    -> Option<u8>    { return try_to_u8(*self); }
    #[inline] fn to_u16(&self)   -> Option<u16>   { return try_to_u16(*self); }
    #[inline] fn to_u32(&self)   -> Option<u32>   { return try_to_u32(*self); }
    #[inline] fn to_u64(&self)   -> Option<u64>   { return try_to_u64(*self); }
    #[inline] fn to_u128(&self)  -> Option<u128>  { return try_to_u128(*self); }

    #[inline] fn to_f32(&self)   -> Option<f32>   { return Some(self.hi as f32); }
    #[inline] fn to_f64(&self)   -> Option<f64>   { return Some(self.hi); }
}

impl num_traits::FromPrimitive for d64 {
    #[inline] fn from_isize(n: isize) -> Option<d64> { return Some(from_isize(n)); }
    #[inline] fn from_i8(n: i8)       -> Option<d64> { return Some(from_i8(n)); }
    #[inline] fn from_i16(n: i16)     -> Option<d64> { return Some(from_i16(n)); }
    #[inline] fn from_i32(n: i32)     -> Option<d64> { return Some(from_i32(n)); }
    #[inline] fn from_i64(n: i64)     -> Option<d64> { return Some(from_i64(n)); }

    #[inline] fn from_usize(n: usize) -> Option<d64> { return Some(from_usize(n)); }
    #[inline] fn from_u8(n: u8)       -> Option<d64> { return Some(from_u8(n)); }
    #[inline] fn from_u16(n: u16)     -> Option<d64> { return Some(from_u16(n)); }
    #[inline] fn from_u32(n: u32)     -> Option<d64> { return Some(from_u32(n)); }
    #[inline] fn from_u64(n: u64)     -> Option<d64> { return Some(from_u64(n)); }
}

