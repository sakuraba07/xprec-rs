use super::d64;
use super::arith::*;
use super::consts::*;
use libm::ldexp;

pub fn exp(x: d64) -> d64
{
    // The value of MAX.ln().
    const LOG_MAX: f64 = 709.782712893384;

    // Now we perform checks for special values. Using not <= instead of >
    // also catches NaNs.
    if !(x.hi.abs() < LOG_MAX) {
        if is_nan(x) {
            return x;
        } else if x.hi > 0.0 {
            return d64::INFINITY;
        } else {
            return d64::from(0.0);
        }
    }

    let (m, expm1_y) = expm1_split(x);
    let exp_m = ldexp(1.0, m);
    let exp_y = addfast_dq(1.0, expm1_y);
    let exp_x = mul_pow2(exp_y, exp_m);
    return exp_x;
}

pub fn expm1(x: d64) -> d64
{
    // The value of MAX.ln().
    const LOG_MAX: f64 = 709.782712893384;

    // Now we perform checks for special values. Using not <= instead of >
    // also catches NaNs.
    if !(x.hi.abs() < LOG_MAX) {
        if is_nan(x) {
            return x;
        } else if x.hi > 0.0 {
            return d64::INFINITY;
        } else {
            return d64::from(-1.0);
        }
    }

    let (m, expm1_y) = expm1_split(x);

    // If m == 0, then it means we can and should use the expm1 kernel
    // directly, otherwise it is okay to simply subtract 1.0
    if m == 0 {
        return expm1_y;
    } else {
        let exp_m = ldexp(1.0, m);
        let exp_y = addfast_dq(1.0, expm1_y);
        let exp_x = mul_pow2(exp_y, exp_m);

        // XXX dispatch based on magnitude
        return exp_x - 1.0;
    }
}

pub fn expm1_split(x: d64) -> (i32, d64)
{
    // Here is the main strategy. Let α be log(2)/128. Then we first reduce the
    // argument x modulo α, i.e.:
    //
    //     x = k * α + y
    //
    let (k, y) = reduce_mod_alpha(x);

    // We further split k = 128 * m + n, where `n` is between {0, ..., 127}
    // Then we have that:
    //
    //     exp(x) = ldexp(1, m) * exp(n * ALPHA + y)
    //
    let (m, n) = reduce_mod_128(k as i32);

    // Now compute expm1 for the small argument to full precision
    let expm1_y = expm1_small(n, y);
    return (m, expm1_y);
}

pub fn log(x: d64) -> d64
{
    // Start with logarithm of hi part
    let log_x0 = x.hi.ln();
    if !log_x0.is_finite() {
        return d64::from(log_x0);
    }

    // Abramowitz and Stegun give the following series expansion (4.1.30):
    //
    //   log(x) = log(x0) + 2 (x - x0)/(x + x0) + O(x - x0)^3
    //
    let x0 = exp(d64::from(log_x0));
    let corr = mul_pow2(subfast_qq(x, x0) / addfast_qq(x, x0), 2.0);
    let log_x = log_x0 + corr;
    return log_x;
}

pub fn log1p(x: d64) -> d64
{
    // Start with logarithm of hi part
    let log_x0 = x.hi.ln_1p();
    if !log_x0.is_finite() {
        return d64::from(log_x0);
    }

    // Again, we can use the same correction, but log1p <-> expm1
    //
    //   log(1 + x) = log(1 + x0) + 2 (x - x0)/(2 + x + x0) + O(x - x0)^3
    //
    // One need not worry about cancellation in the denominator for
    // x close to -1, since that is where we have an intrinsic loss of
    // precision anyway
    let x0 = expm1(d64::from(log_x0));
    let corr = mul_pow2(subfast_qq(x, x0) / addfast_qq(2.0 + x, x0), 2.0);
    let log_x = log_x0 + corr;
    return log_x;
}

fn reduce_mod_128(k: i32) -> (i32, i32)
{
    let mut m = k >> 7;
    let mut n = k & 0x7F;
    if k & 0x40 != 0 {
        n -= 0x80;
        m += 1;
    }
    return (m, n);
}

fn reduce_mod_alpha(x: d64) -> (f64, d64)
{
    // ALPHA_T is an approximation of log(2)/128 to 90 significant bits -- 17
    // bits fewer than full double-double precision.  Observe then that
    // 128*log(DBL_MAX) is around 91000, which fit comfortably into 17 bits.
    // That means that the reduction of x modulo ALPHA_T:
    //
    //     x = n * ALPHA_T + z
    //
    // is *exact* for any x in the range of the exponential funcion.  We have
    // to correct this expression to at least 124 digits. The correction term
    // only needs to be in double precision
    //
    //     z = n * ALPHA_CORR + y
    //
    const INV_ALPHA: f64 = 184.6649652337873;
    const ALPHA_T: d64 = d64 {hi: 0.0054152123481245725, lo: 1.8117553232937405e-19};
    const ALPHA_CORR: f64 = 2.3681038446414578e-30;

    // maybe use ceil here instead
    let n = (x.hi * INV_ALPHA).round();
    let z_raw = subfast_qq(x, n * ALPHA_T);
    let z = subfast_qd(z_raw, n * ALPHA_CORR);
    return (n, z);
}

fn expm1_small(n: i32, y: d64) -> d64
{
    // Assuming a reduction mod α = log(2)/128:
    //
    //     x = n * α + y,
    //
    // the idea is to use the identity
    //
    //     expm1(x) = expm1(n * α) + exp(n * α) * expm1(y)
    //
    // to reduce the expansion order.
    assert!(2.0 * y.hi.abs() <= 0.0054152123481245725);
    let expm1_n = expm1_alphas(n);
    let exp_n = addfast_dq(1.0, expm1_n);
    let expm1_y = expm1_kernel(y, 6, 10);
    return addfast_qq(expm1_n, expm1_y * exp_n);
}

fn expm1_kernel(x: d64, nquad: i32, n: i32) -> d64
{
    assert!(x.hi.abs() < 1.0);

    // r = x
    let mut r = x;

    // r += x * x / 2
    let mut xpow = square_q(x);
    r = addfast_qq(r, mul_pow2(xpow, 0.5));

    // r += x^k / k!
    for k in 3..nquad+1 {
        xpow = xpow * x;
        r = addfast_qq(r, reciprocal_factorial(k) * xpow);
    }

    // Here the terms are so small that they only affect the lo part, so
    // we can get away with double arithmetic.
    let mut r_d: f64 = 0.0;
    let mut xpow_d = xpow.hi;
    for k in nquad+1..n+1 {
        xpow_d *= x.hi;
        r_d += reciprocal_factorial(k).hi * xpow_d;
    }
    r = addfast_qd(r, r_d);
    return r;
}

#[inline]
const fn reciprocal_factorial(n: i32) -> d64
{
    const RECIPROCAL_FACTORIAL : [d64; 30] = [
        d64 { hi: 1.0, lo: 0.0 },
        d64 { hi: 1.0, lo: 0.0 },
        d64 { hi: 0.5, lo: 0.0 }, // [2] = 1/2!
        d64 { hi: 0.16666666666666666, lo: 9.25185853854297e-18 },
        d64 { hi: 0.041666666666666664, lo: 2.3129646346357427e-18 },
        d64 { hi: 0.008333333333333333, lo: 1.1564823173178714e-19 },
        d64 { hi: 0.001388888888888889, lo: -5.300543954373577e-20 },
        d64 { hi: 0.0001984126984126984, lo: 1.7209558293420705e-22 },
        d64 { hi: 2.48015873015873e-5, lo: 2.1511947866775882e-23 },
        d64 { hi: 2.7557319223985893e-6, lo: -1.858393274046472e-22 },
        d64 { hi: 2.755731922398589e-7, lo: 2.3767714622250297e-23 },
        d64 { hi: 2.505210838544172e-8, lo: -1.448814070935912e-24 },
        d64 { hi: 2.08767569878681e-9, lo: -1.20734505911326e-25 },
        d64 { hi: 1.6059043836821613e-10, lo: 1.2585294588752098e-26 },
        d64 { hi: 1.1470745597729725e-11, lo: 2.0655512752830745e-28 },
        d64 { hi: 7.647163731819816e-13, lo: 7.03872877733453e-30 },
        d64 { hi: 4.779477332387385e-14, lo: 4.399205485834081e-31 },
        d64 { hi: 2.8114572543455206e-15, lo: 1.6508842730861433e-31 },
        d64 { hi: 1.5619206968586225e-16, lo: 1.1910679660273754e-32 },
        d64 { hi: 8.22063524662433e-18, lo: 2.2141894119604265e-34 },
        d64 { hi: 4.110317623312165e-19, lo: 1.4412973378659527e-36 },
        d64 { hi: 1.9572941063391263e-20, lo: -1.3643503830087908e-36 },
        d64 { hi: 8.896791392450574e-22, lo: -7.911402614872376e-38 },
        d64 { hi: 3.868170170630684e-23, lo: -8.843177655482344e-40 },
        d64 { hi: 1.6117375710961184e-24, lo: -3.6846573564509766e-41 },
        d64 { hi: 6.446950284384474e-26, lo: -1.9330404233703465e-42 },
        d64 { hi: 2.4795962632247976e-27, lo: -1.2953730964765229e-43 },
        d64 { hi: 9.183689863795546e-29, lo: 1.4303150396787322e-45 },
        d64 { hi: 3.279889237069838e-30, lo: 1.5117542744029879e-46 },
        d64 { hi: 1.1309962886447716e-31, lo: 1.0498015412959506e-47 }
        ];

    assert!(n >= 0 && n < 30);
    return RECIPROCAL_FACTORIAL[n as usize];
}

#[inline]
const fn expm1_alphas(n: i32) -> d64
{
    // For multiples of alpha = log(2)/128, precompute and store the
    // exponential function in a table, from -64*alpha until 64*alpha
    const EXPM1_ALPHAS : [d64; 128] = [
        d64 { hi: -0.2928932188134525, lo: 7.174684663993261e-18 },
        d64 { hi: -0.2890536989154172, lo: -8.038914457945122e-18 },
        d64 { hi: -0.285193330804015, lo: -6.0158212445268276e-18 },
        d64 { hi: -0.2813120012755088, lo: -2.1020170082337783e-17 },
        d64 { hi: -0.2774095965114767, lo: -1.5118790674969937e-17 },
        d64 { hi: -0.27348600207547374, lo: 2.66114081842773e-17 },
        d64 { hi: -0.26954110290967653, lo: 2.7509265300881745e-17 },
        d64 { hi: -0.265574783331509, lo: -1.318173744858969e-17 },
        d64 { hi: -0.2615869270302503, lo: -1.741997278446398e-17 },
        d64 { hi: -0.25757741706362375, lo: -1.6107174092204261e-18 },
        d64 { hi: -0.2535461358543676, lo: 7.096460077142018e-18 },
        d64 { hi: -0.24949296518678724, lo: -4.31326076332226e-18 },
        d64 { hi: -0.24541778620328863, lo: 4.688384843543075e-18 },
        d64 { hi: -0.24132047940089266, lo: 6.212078255412209e-18 },
        d64 { hi: -0.23720092462773085, lo: 3.8644266954502085e-19 },
        d64 { hi: -0.233059001079522, lo: -1.1135017009065593e-17 },
        d64 { hi: -0.2288945872960296, lo: 1.199359843285919e-17 },
        d64 { hi: -0.2247075611575, lo: -7.300353295344693e-18 },
        d64 { hi: -0.2204977998810815, lo: -8.849540348841276e-18 },
        d64 { hi: -0.21626518001722356, lo: 3.750842387009219e-18 },
        d64 { hi: -0.21200957744605675, lo: -5.068458235639152e-18 },
        d64 { hi: -0.20773086737375313, lo: -9.668858517292851e-18 },
        d64 { hi: -0.20342892432886656, lo: 5.039118519698011e-18 },
        d64 { hi: -0.19910362215865332, lo: -2.5190116520100086e-18 },
        d64 { hi: -0.19475483402537286, lo: 1.2353596284898944e-17 },
        d64 { hi: -0.19038243240256814, lo: 1.0470667077114546e-17 },
        d64 { hi: -0.1859862890713261, lo: -5.809199807906506e-18 },
        d64 { hi: -0.18156627511651777, lo: 1.0736049740970466e-17 },
        d64 { hi: -0.17712226092301758, lo: 4.882751662883964e-18 },
        d64 { hi: -0.1726541161719028, lo: -7.294679715277685e-18 },
        d64 { hi: -0.16816170983663178, lo: 1.699387867936586e-18 },
        d64 { hi: -0.1636449101792017, lo: 3.719957926310978e-19 },
        d64 { hi: -0.15910358474628547, lo: 1.3239474487278572e-17 },
        d64 { hi: -0.15453760036534742, lo: 7.162793859283428e-18 },
        d64 { hi: -0.14994682314073826, lo: -4.01185968519885e-18 },
        d64 { hi: -0.1453311184497686, lo: 6.167253948093172e-18 },
        d64 { hi: -0.14069035093876103, lo: -9.256902091315555e-18 },
        d64 { hi: -0.13602438451908122, lo: 1.7562419252346148e-18 },
        d64 { hi: -0.13133308236314686, lo: -1.1933629119164127e-17 },
        d64 { hi: -0.12661630690041553, lo: 1.749698813720255e-18 },
        d64 { hi: -0.12187391981335026, lo: 9.229156694299104e-19 },
        d64 { hi: -0.1171057820333636, lo: 5.67321166697297e-18 },
        d64 { hi: -0.11231175373673938, lo: 4.393083367153945e-18 },
        d64 { hi: -0.1074916943405325, lo: -6.2125877472988e-18 },
        d64 { hi: -0.1026454624984464, lo: -4.7640585938584126e-18 },
        d64 { hi: -0.09777291609668806, lo: 1.869463571662324e-18 },
        d64 { hi: -0.09287391224980063, lo: 5.66349353665608e-18 },
        d64 { hi: -0.08794830729647335, lo: 4.713011919872412e-18 },
        d64 { hi: -0.08299595679532877, lo: 2.537748313413679e-18 },
        d64 { hi: -0.07801671552068704, lo: -1.94313451912091e-18 },
        d64 { hi: -0.07301043745830721, lo: -6.701713777619857e-18 },
        d64 { hi: -0.06797697580110548, lo: 4.948987787473942e-18 },
        d64 { hi: -0.06291618294485005, lo: -2.8582414493917966e-18 },
        d64 { hi: -0.057827910483832776, lo: 5.00397795774813e-19 },
        d64 { hi: -0.05271200920651718, lo: 3.1392298682681924e-18 },
        d64 { hi: -0.047568329091162896, lo: -2.025181945944751e-18 },
        d64 { hi: -0.042396719301426355, lo: 2.4114209502780123e-18 },
        d64 { hi: -0.037197028181937535, lo: -1.0025615211181075e-18 },
        d64 { hi: -0.03196910325385278, lo: 3.089672476031033e-18 },
        d64 { hi: -0.026712791210383356, lo: -6.393577718667539e-19 },
        d64 { hi: -0.021427937912299865, lo: -2.989714202136461e-19 },
        d64 { hi: -0.01611438838341211, lo: 4.670642216485574e-19 },
        d64 { hi: -0.010771986806024515, lo: -6.223051570826017e-19 },
        d64 { hi: -0.005400576516366824, lo: -2.342423707574178e-19 },
        d64 { hi: 0.0, lo: 0.0 },
        d64 { hi: 0.005429901112802822, lo: -4.1792582417406993e-19 },
        d64 { hi: 0.01088928605170046, lo: 3.7773268042268547e-19 },
        d64 { hi: 0.016378314910953037, lo: 1.2588974512148405e-18 },
        d64 { hi: 0.02189714865411668, lo: -9.494539895697731e-19 },
        d64 { hi: 0.027445949118763698, lo: -9.884844191031042e-19 },
        d64 { hi: 0.03302487902122842, lo: 6.619449701198605e-19 },
        d64 { hi: 0.03863410196137879, lo: -2.487307246639953e-18 },
        d64 { hi: 0.04427378242741384, lo: 2.252170208492904e-18 },
        d64 { hi: 0.049944085800687266, lo: 4.182272500122047e-19 },
        d64 { hi: 0.05564517836055716, lo: 1.759325738772092e-18 },
        d64 { hi: 0.06137722728926208, lo: 1.9042507224487988e-18 },
        d64 { hi: 0.06714040067682361, lo: 4.268187178470922e-18 },
        d64 { hi: 0.07293486752597556, lo: -3.839668843358824e-18 },
        d64 { hi: 0.07876079775711979, lo: 2.8223346785063543e-18 },
        d64 { hi: 0.08461836221330923, lo: 3.905952842534547e-18 },
        d64 { hi: 0.09050773266525766, lo: -2.712245182495796e-18 },
        d64 { hi: 0.09642908181637683, lo: -3.6881836132353304e-18 },
        d64 { hi: 0.10238258330784095, lo: -2.8507825155508824e-18 },
        d64 { hi: 0.10836841172367864, lo: -4.601411604918528e-18 },
        d64 { hi: 0.11438674259589254, lo: -6.919517894059943e-18 },
        d64 { hi: 0.12043775240960669, lo: -6.499707834283954e-18 },
        d64 { hi: 0.1265216186082419, lo: -3.8525836433032604e-18 },
        d64 { hi: 0.13263851959871922, lo: 4.617986051751087e-18 },
        d64 { hi: 0.13878863475669165, lo: 5.861399913367335e-18 },
        d64 { hi: 0.14497214443180423, lo: -9.09825230955772e-18 },
        d64 { hi: 0.1511892299529827, lo: 4.751526573009359e-18 },
        d64 { hi: 0.15744007363375104, lo: -7.971985464457258e-18 },
        d64 { hi: 0.1637248587775775, lo: 1.0536472753612021e-17 },
        d64 { hi: 0.1700437696832502, lo: -1.8477442017900047e-18 },
        d64 { hi: 0.17639699165028128, lo: 3.088131092296112e-20 },
        d64 { hi: 0.18278471098434104, lo: -1.2325821314838153e-17 },
        d64 { hi: 0.18920711500272105, lo: 1.2064576699027549e-17 },
        d64 { hi: 0.19566439203982738, lo: -9.345114526443012e-18 },
        d64 { hi: 0.20215673145270313, lo: 1.0938663761265181e-17 },
        d64 { hi: 0.20868432362658157, lo: 8.043891778967983e-18 },
        d64 { hi: 0.21524735998046887, lo: 6.140419920071864e-18 },
        d64 { hi: 0.2218460329727575, lo: 4.912090348488744e-18 },
        d64 { hi: 0.22848053610687, lo: 8.767759302603614e-18 },
        d64 { hi: 0.2351510639369333, lo: 3.469859019437239e-18 },
        d64 { hi: 0.24185781207348406, lo: -8.930875312888462e-18 },
        d64 { hi: 0.24860097718920474, lo: 6.4861685666710185e-19 },
        d64 { hi: 0.2553807570246911, lo: -6.7113898212968784e-18 },
        d64 { hi: 0.2621973503942507, lo: 2.4666502356519365e-17 },
        d64 { hi: 0.2690509571917332, lo: 2.667932131342186e-18 },
        d64 { hi: 0.2759417783963921, lo: -1.1868000020372746e-17 },
        d64 { hi: 0.28287001607877826, lo: 1.713594918243561e-17 },
        d64 { hi: 0.28983587340666583, lo: -2.1529727153539737e-17 },
        d64 { hi: 0.29683955465100964, lo: 2.5382502794888315e-17 },
        d64 { hi: 0.3038812651919359, lo: -2.4545546479836942e-17 },
        d64 { hi: 0.31096121152476436, lo: -1.6304210123936712e-17 },
        d64 { hi: 0.318079601266064, lo: 9.315929597662924e-19 },
        d64 { hi: 0.32523664315974127, lo: 2.6923839130869213e-17 },
        d64 { hi: 0.33243254708316144, lo: 4.495284922090389e-18 },
        d64 { hi: 0.339667524053303, lo: -2.1749476514198334e-17 },
        d64 { hi: 0.34694178623294586, lo: -2.3270500218711038e-17 },
        d64 { hi: 0.3542555469368927, lo: 2.1498332566772065e-17 },
        d64 { hi: 0.36160902063822475, lo: 1.533787661270668e-18 },
        d64 { hi: 0.3690024229745906, lo: -1.5084323271327172e-17 },
        d64 { hi: 0.3764359707545301, lo: -1.3474738127460185e-17 },
        d64 { hi: 0.38390988196383197, lo: -1.2193965356690036e-17 },
        d64 { hi: 0.3914243757719262, lo: 6.4494025783679345e-18 },
        d64 { hi: 0.3989796725383111, lo: 1.4880170372002426e-17 },
        d64 { hi: 0.40657599381901544, lo: 7.034914812136422e-18 }
        ];

    assert!(n >= -64 && n < 64);
    return EXPM1_ALPHAS[(n + 64) as usize];
}

#[cfg(test)]
mod test {
    use super::*;
    use super::super::test_utils::*;

    #[test]
    fn test_expm1_kernel()
    {
        // small values, start from ALPHA/2
        let mut x = d64::from(0.0025);
        while x.hi > 1e-290 {
            check_unary(|x| expm1_kernel(x, 6, 10), |x| x.exp_m1(), x, 1.1);
            check_unary(|x| expm1_kernel(x, 6, 10), |x| x.exp_m1(), -x, 1.1);
            x *= 0.947;
        }
    }

    #[test]
    fn test_exp()
    {
        // special values
        assert!(is_infinite(exp(d64::from(1000.0))));
        assert!(is_infinite(exp(d64::INFINITY)));
        assert!(is_zero(exp(d64::from(-1000.0))));
        assert!(is_zero(exp(-d64::INFINITY)));
        assert!(is_nan(exp(d64::NAN)));

        // simple vals
        check_unary(exp, |x| x.exp(), d64::from(0.0), 1.0);

        // small values
        let mut x = d64::from(0.25);
        while x.hi > 1e-290 {
            check_unary(exp, |x| x.exp(), x, 1.0);
            check_unary(exp, |x| x.exp(), -x, 1.0);
            x *= 0.947;
        }

        check_unary(exp, |x| x.exp(), d64::from(1.0), 1.0);

        // large values
        x = d64::from(0.25);
        while x.hi < 708.0 {
            check_unary(exp, |x| x.exp(), x, 1.0);
            if x.hi < 670.0 {
                check_unary(exp, |x| x.exp(), -x, 1.0);
            }
            x *= 1.0041;
        }
    }

    #[test]
    fn test_expm1()
    {
        // special values
        assert!(is_infinite(expm1(d64::from(1000.0))));
        assert!(is_infinite(expm1(d64::INFINITY)));
        assert!(expm1(d64::from(-1000.0)) == d64::from(-1.0));
        assert!(expm1(-d64::INFINITY) == d64::from(-1.0));
        assert!(is_nan(expm1(d64::NAN)));

        // simple vals
        check_unary(expm1, |x| x.exp_m1(), d64::from(0.0), 1.0);
        check_unary(expm1, |x| x.exp_m1(), d64::from(1.0), 1.0);

        // small values
        // XXX here we have to work on the kernel
        let mut x = d64::from(0.5);
        while x.hi > 1e-290 {
            check_unary(expm1, |x| x.exp_m1(), x, 1.5);
            check_unary(expm1, |x| x.exp_m1(), -x, 1.5);
            x *= 0.947;
        }

        // large values
        x = d64::from(0.5);
        while x.hi < 708.0 {
            check_unary(expm1, |x| x.exp_m1(), x, 1.0);
            if x.hi < 670.0 {
                check_unary(expm1, |x| x.exp_m1(), -x, 1.0);
            }
            x *= 1.0041;
        }
    }

    #[test]
    fn test_log()
    {
        // special values
        assert!(is_infinite(log(d64::INFINITY)));
        assert!(is_infinite(log(d64::from(0.0))));
        assert!(is_nan(log(d64::from(-0.1))));
        assert!(is_nan(log(d64::NEG_INFINITY)));
        assert!(is_nan(log(d64::NAN)));

        // simple vals
        check_unary(log, |x| x.ln(), d64::from(1.0), 1.0);
        check_unary(log, |x| x.ln(), d64::from(3.0), 1.0);

        // small values
        let mut x = d64::from(1.0);
        while x.hi > 1e-290 {
            check_unary(log, |x| x.ln(), x, 1.0);
            x *= 0.947;
        }

        // large values
        x = d64::from(1.0);
        while x.hi < 1e300 {
            check_unary(log, |x| x.ln(), x, 1.0);
            x *= 1.13;
        }
    }

    #[test]
    fn test_log1p()
    {
        // special values
        assert!(is_infinite(log1p(d64::INFINITY)));
        assert!(is_infinite(log1p(d64::from(-1.0))));
        assert!(is_nan(log1p(d64::from(-1.1))));
        assert!(is_nan(log1p(d64::NEG_INFINITY)));
        assert!(is_nan(log1p(d64::NAN)));

        // simple vals
        check_unary(log1p, |x| x.ln_1p(), d64::from(0.0), 0.0);
        check_unary(log1p, |x| x.ln_1p(), d64::from(1.0), 1.5);

        // small values
        let mut x = d64::from(0.99);
        while x.hi > 1e-290 {
            check_unary(log1p, |x| x.ln_1p(), x, 1.5);
            check_unary(log1p, |x| x.ln_1p(), -x, 1.5);
            x *= 0.947;
        }

        // large values
        x = d64::from(1.0);
        while x.hi < 1e300 {
            check_unary(log1p, |x| x.ln_1p(), x, 1.0);
            x *= 1.13;
        }
    }
}
