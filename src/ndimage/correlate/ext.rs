//! Boundary handling, ported from SciPy's ni_support.c.
//!
//! Instead of extending every line, the kernels precompute for each position
//! of an extended line which input element it holds (or `CONST` for cval).

use crate::ndimage::common::Mode;

/// Marker for "use cval" in a position map.
pub const CONST: isize = -1;

/// Source index of every position of a line extended by `before` and `after`
/// elements, exactly as `NI_ExtendLine` fills its buffer (including windows
/// longer than the line, where the C code keeps copying from the already
/// extended part).
pub fn extend_map(len: usize, before: usize, after: usize, mode: Mode) -> Vec<isize> {
    let total = before + len + after;
    let mut b = vec![CONST; total];
    for i in 0..len {
        b[before + i] = i as isize;
    }
    let first = before as isize;
    let last = (before + len) as isize;
    let mut mode = mode;
    if len == 1 && mode == Mode::Mirror {
        mode = Mode::Nearest;
    }
    let set = |b: &mut Vec<isize>, dst: isize, src: isize| {
        b[dst as usize] = b[src as usize];
    };
    match mode {
        Mode::Nearest => {
            for d in 0..first {
                set(&mut b, d, first);
            }
            for d in last..total as isize {
                set(&mut b, d, last - 1);
            }
        }
        Mode::Wrap | Mode::GridWrap => {
            let (mut src, mut dst) = (last - 1, first - 1);
            for _ in 0..before {
                set(&mut b, dst, src);
                dst -= 1;
                src -= 1;
            }
            let (mut src, mut dst) = (first, last);
            for _ in 0..after {
                set(&mut b, dst, src);
                dst += 1;
                src += 1;
            }
        }
        Mode::Reflect => {
            let (mut src, mut dst, mut n) = (first, first - 1, before);
            while n > 0 && src < last {
                set(&mut b, dst, src);
                dst -= 1;
                src += 1;
                n -= 1;
            }
            let mut src = last - 1;
            while n > 0 {
                set(&mut b, dst, src);
                dst -= 1;
                src -= 1;
                n -= 1;
            }
            let (mut src, mut dst, mut n) = (last - 1, last, after);
            while n > 0 && src >= first {
                set(&mut b, dst, src);
                dst += 1;
                src -= 1;
                n -= 1;
            }
            let mut src = first;
            while n > 0 {
                set(&mut b, dst, src);
                dst += 1;
                src += 1;
                n -= 1;
            }
        }
        Mode::Mirror => {
            let (mut src, mut dst, mut n) = (first + 1, first - 1, before);
            while n > 0 && src < last {
                set(&mut b, dst, src);
                dst -= 1;
                src += 1;
                n -= 1;
            }
            let mut src = last - 2;
            while n > 0 {
                set(&mut b, dst, src);
                dst -= 1;
                src -= 1;
                n -= 1;
            }
            let (mut src, mut dst, mut n) = (last - 2, last, after);
            while n > 0 && src >= first {
                set(&mut b, dst, src);
                dst += 1;
                src -= 1;
                n -= 1;
            }
            let mut src = first + 1;
            while n > 0 {
                set(&mut b, dst, src);
                dst += 1;
                src += 1;
                n -= 1;
            }
        }
        Mode::Constant | Mode::GridConstant => {}
    }
    b
}

/// `extend_map` without storing the identity part: extended position `q`
/// holds line element `q - before` for `before <= q < before + len`.
pub struct ExtMap {
    before: usize,
    len: usize,
    head: Vec<isize>,
    tail: Vec<isize>,
}

impl ExtMap {
    pub fn new(len: usize, before: usize, after: usize, mode: Mode) -> ExtMap {
        // NI_ExtendLine only reads the first and last `before + after + 1`
        // elements of a line, so a long line can be emulated by a short one
        // made of its two ends.
        let m = before + after + 1;
        let (head, tail) = if len > 2 * m {
            let b = extend_map(2 * m, before, after, mode);
            let fix = |v: isize| {
                if v == CONST || v < m as isize {
                    v
                } else {
                    v + (len - 2 * m) as isize
                }
            };
            (
                b[..before].iter().map(|&v| fix(v)).collect(),
                b[before + 2 * m..].iter().map(|&v| fix(v)).collect(),
            )
        } else {
            let b = extend_map(len, before, after, mode);
            (b[..before].to_vec(), b[before + len..].to_vec())
        };
        ExtMap {
            before,
            len,
            head,
            tail,
        }
    }

    /// Total extended length.
    pub fn lext(&self) -> usize {
        self.before + self.len + self.tail.len()
    }

    /// Source index (or CONST) of extended position `q`.
    #[inline(always)]
    pub fn get(&self, q: usize) -> isize {
        if q < self.before {
            self.head[q]
        } else if q < self.before + self.len {
            (q - self.before) as isize
        } else {
            self.tail[q - self.before - self.len]
        }
    }
}

/// Map coordinate `cc` along an axis of length `len` into the array, as
/// `NI_InitFilterOffsets` does; `None` means "outside, use cval".
#[inline]
pub fn nd_map(cc: isize, len: isize, mode: Mode) -> Option<isize> {
    if cc >= 0 && cc < len {
        return Some(cc);
    }
    Some(match mode {
        Mode::Mirror => {
            if len <= 1 {
                0
            } else {
                let sz2 = 2 * len - 2;
                if cc < 0 {
                    let cc = sz2 * (-cc / sz2) + cc;
                    if cc <= 1 - len { cc + sz2 } else { -cc }
                } else {
                    let cc = cc - sz2 * (cc / sz2);
                    if cc >= len { sz2 - cc } else { cc }
                }
            }
        }
        Mode::Reflect => {
            if len <= 1 {
                0
            } else {
                let sz2 = 2 * len;
                if cc < 0 {
                    let mut cc = cc;
                    if cc < -sz2 {
                        cc = sz2 * (-cc / sz2) + cc;
                    }
                    if cc < -len { cc + sz2 } else { -cc - 1 }
                } else {
                    let cc = cc - sz2 * (cc / sz2);
                    if cc >= len { sz2 - cc - 1 } else { cc }
                }
            }
        }
        Mode::Wrap | Mode::GridWrap => {
            if len <= 1 {
                0
            } else if cc < 0 {
                let cc = cc + len * (-cc / len);
                if cc < 0 { cc + len } else { cc }
            } else {
                cc - len * (cc / len)
            }
        }
        Mode::Nearest => {
            if cc < 0 {
                0
            } else {
                len - 1
            }
        }
        Mode::Constant | Mode::GridConstant => return None,
    })
}
