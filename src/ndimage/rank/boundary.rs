//! Boundary extension reproducing SciPy's `NI_ExtendLine` (1-D line buffers)
//! and `NI_InitFilterOffsets` (n-d filters), and padded-buffer construction.

use crate::ndimage::common::Mode;

/// Map coordinate `cc` of an axis of length `len` to the index SciPy reads,
/// or None for the constant value. Both `NI_ExtendLine` and
/// `NI_InitFilterOffsets` extend periodically with these rules (apart from the
/// reflect quirk handled by `reflect_quirk`).
pub fn map_coord(mode: Mode, cc: i64, len: i64) -> Option<i64> {
    if cc >= 0 && cc < len {
        return Some(cc);
    }
    match mode {
        Mode::Constant | Mode::GridConstant => None,
        Mode::Nearest => Some(if cc < 0 { 0 } else { len - 1 }),
        _ if len <= 1 => Some(0),
        Mode::Wrap | Mode::GridWrap => Some(cc.rem_euclid(len)),
        Mode::Reflect => {
            let p = 2 * len;
            let m = cc.rem_euclid(p);
            Some(if m < len { m } else { p - 1 - m })
        }
        Mode::Mirror => {
            let p = 2 * len - 2;
            let m = cc.rem_euclid(p);
            Some(if m < len { m } else { p - m })
        }
    }
}

/// `NI_InitFilterOffsets` maps reflect coordinates `cc <= -4 * len` that are
/// multiples of `2 * len` to -1 (an out-of-bounds read). Returns true when a
/// footprint reaching `reach_before` elements before the array may hit that.
pub fn reflect_quirk(mode: Mode, reach_before: i64, len: i64) -> bool {
    mode == Mode::Reflect && len > 1 && reach_before >= 4 * len
}

/// Per-axis index table for a padded axis: entry q is the source index for
/// padded coordinate q (original coordinate q - before), or -1 for cval.
pub fn axis_table(mode: Mode, len: usize, before: usize, after: usize) -> Vec<i64> {
    let total = len + before + after;
    (0..total)
        .map(|q| map_coord(mode, q as i64 - before as i64, len as i64).unwrap_or(-1))
        .collect()
}
