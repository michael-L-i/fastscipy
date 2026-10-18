//! Selection networks evaluated on vectors of adjacent output pixels.
//!
//! A network is a list of compare-exchange (and pruned min-only / max-only /
//! copy) operations on "slots"; each slot holds one window element for `L`
//! consecutive output pixels along the last axis. Because min/max are applied
//! lane-wise, the result for every lane is the exact order statistic of that
//! lane's window (inputs are free of NaN and negative zero, so equal values
//! have identical bits).
//!
//! Networks are built from the footprint: elements are grouped into rows
//! (equal leading coordinates), each row is sorted, and rows are combined
//! with Batcher odd-even merges. Two outputs along the second-to-last axis
//! are computed together, sharing the merge of the rows their windows have in
//! common. Operations that do not reach an output are pruned.

pub const CE: u8 = 0;
pub const MIN: u8 = 1;
pub const MAX: u8 = 2;
pub const COPY: u8 = 3;

#[derive(Clone, Copy, Debug)]
pub struct Op {
    pub kind: u8,
    pub a: u16,
    pub b: u16,
}

/// A compiled network: inputs are loaded into slots, ops run, outputs read.
#[derive(Clone, Debug)]
pub struct Net {
    pub ops: Vec<Op>,
    pub nslots: usize,
    /// (slot, offset relative to the window base of output 0)
    pub inputs: Vec<(u16, isize)>,
    /// one slot per output
    pub outputs: Vec<u16>,
}

struct Builder {
    ops: Vec<(u8, usize, usize)>,
    nslots: usize,
}

impl Builder {
    fn ce(&mut self, a: usize, b: usize) {
        self.ops.push((CE, a, b));
    }

    fn new_slot(&mut self) -> usize {
        self.nslots += 1;
        self.nslots - 1
    }

    /// Batcher odd-even merge of two sorted slot lists; returns merged order.
    fn merge(&mut self, a: &[usize], b: &[usize]) -> Vec<usize> {
        if a.is_empty() {
            return b.to_vec();
        }
        if b.is_empty() {
            return a.to_vec();
        }
        if a.len() == 1 && b.len() == 1 {
            self.ce(a[0], b[0]);
            return vec![a[0], b[0]];
        }
        let ae: Vec<usize> = a.iter().copied().step_by(2).collect();
        let be: Vec<usize> = b.iter().copied().step_by(2).collect();
        let ao: Vec<usize> = a.iter().copied().skip(1).step_by(2).collect();
        let bo: Vec<usize> = b.iter().copied().skip(1).step_by(2).collect();
        let c = self.merge(&ae, &be);
        let d = self.merge(&ao, &bo);
        let mut res = Vec::with_capacity(a.len() + b.len());
        res.push(c[0]);
        let mut i = 0;
        while i < d.len() || i + 1 < c.len() {
            if i < d.len() && i + 1 < c.len() {
                self.ce(d[i], c[i + 1]);
                res.push(d[i]);
                res.push(c[i + 1]);
            } else if i < d.len() {
                res.push(d[i]);
            } else {
                res.push(c[i + 1]);
            }
            i += 1;
        }
        res
    }

    /// Merge sorted lists, smallest two first; returns the sorted order.
    fn merge_all(&mut self, mut lists: Vec<Vec<usize>>) -> Vec<usize> {
        lists.retain(|l| !l.is_empty());
        if lists.is_empty() {
            return Vec::new();
        }
        while lists.len() > 1 {
            lists.sort_by_key(|l| std::cmp::Reverse(l.len()));
            let a = lists.pop().unwrap();
            let b = lists.pop().unwrap();
            let m = self.merge(&b, &a);
            lists.push(m);
        }
        lists.pop().unwrap()
    }

    /// Sort an unsorted group of slots (odd-even merge sort).
    fn sort(&mut self, group: &[usize]) -> Vec<usize> {
        self.merge_all(group.iter().map(|&s| vec![s]).collect())
    }
}

/// Build a network for windows given by `rows` (each row: list of padded
/// offsets of the window elements, relative to the window base), for
/// output 0 at base 0 and, when `shift` is Some(s), output 1 at base s,
/// selecting rank `r` of each window.
pub fn build(rows: &[Vec<isize>], shift: Option<isize>, r: usize) -> Net {
    use std::collections::{BTreeSet, HashMap};
    let set_a: BTreeSet<isize> = rows.iter().flatten().copied().collect();
    let k = set_a.len();
    let mut b = Builder { ops: Vec::new(), nslots: 0 };
    let mut slot_of: HashMap<isize, usize> = HashMap::new();
    let mut inputs = Vec::new();
    let mut slot = |b: &mut Builder, off: isize| -> usize {
        *slot_of.entry(off).or_insert_with(|| {
            let s = b.new_slot();
            inputs.push((s, off));
            s
        })
    };
    // split rows into core (in both windows) and extras
    let set_b: BTreeSet<isize> = match shift {
        Some(s) => set_a.iter().map(|&o| o + s).collect(),
        None => BTreeSet::new(),
    };
    let mut core_rows: Vec<Vec<usize>> = Vec::new();
    let mut extra_a: Vec<Vec<usize>> = Vec::new();
    let mut extra_b: Vec<Vec<usize>> = Vec::new();
    for row in rows {
        let (mut c, mut e) = (Vec::new(), Vec::new());
        for &o in row {
            if shift.is_some() && set_b.contains(&o) {
                c.push(slot(&mut b, o));
            } else {
                e.push(slot(&mut b, o));
            }
        }
        core_rows.push(c);
        extra_a.push(e);
        if let Some(s) = shift {
            let eb: Vec<usize> = row
                .iter()
                .map(|&o| o + s)
                .filter(|o| !set_a.contains(o))
                .map(|o| slot(&mut b, o))
                .collect();
            extra_b.push(eb);
        }
    }
    assert!(r < k);
    let mut outputs = Vec::new();
    let sorted: Vec<Vec<usize>> = core_rows.iter().map(|g| b.sort(g)).collect();
    let core = b.merge_all(sorted);
    // output 1 works on copies of the core, made before output 0 merges into
    // the core slots in place
    let core_b: Option<Vec<usize>> = shift.map(|_| {
        core.iter()
            .map(|&s| {
                let d = b.new_slot();
                b.ops.push((COPY, s, d));
                d
            })
            .collect()
    });
    let mut la: Vec<Vec<usize>> = extra_a.iter().map(|g| b.sort(g)).collect();
    la.push(core);
    let full_a = b.merge_all(la);
    outputs.push(full_a[r]);
    if let Some(cb) = core_b {
        let mut lb: Vec<Vec<usize>> = extra_b.iter().map(|g| b.sort(g)).collect();
        lb.push(cb);
        let full_b = b.merge_all(lb);
        outputs.push(full_b[r]);
    }
    prune_and_compact(b, inputs, outputs)
}

fn prune_and_compact(b: Builder, inputs: Vec<(usize, isize)>, outputs: Vec<usize>) -> Net {
    // backward liveness
    let mut live = vec![false; b.nslots];
    for &o in &outputs {
        live[o] = true;
    }
    let mut kept: Vec<(u8, usize, usize)> = Vec::new();
    for &(kind, x, y) in b.ops.iter().rev() {
        match kind {
            CE => {
                let (lx, ly) = (live[x], live[y]);
                if lx && ly {
                    kept.push((CE, x, y));
                } else if lx {
                    kept.push((MIN, x, y));
                    live[y] = true;
                } else if ly {
                    kept.push((MAX, x, y));
                    live[x] = true;
                }
            }
            COPY => {
                // y = x
                if live[y] {
                    kept.push((COPY, x, y));
                    live[y] = false;
                    live[x] = true;
                }
            }
            _ => unreachable!(),
        }
    }
    kept.reverse();
    // renumber slots: inputs first (only live ones), then others
    let mut map = vec![usize::MAX; b.nslots];
    let mut next = 0usize;
    let mut new_inputs = Vec::new();
    for &(s, off) in &inputs {
        if live[s] {
            map[s] = next;
            new_inputs.push((next as u16, off));
            next += 1;
        }
    }
    let mut ops = Vec::with_capacity(kept.len());
    for &(kind, x, y) in &kept {
        for s in [x, y] {
            if map[s] == usize::MAX {
                map[s] = next;
                next += 1;
            }
        }
        ops.push(Op { kind, a: map[x] as u16, b: map[y] as u16 });
    }
    let outputs = outputs
        .iter()
        .map(|&o| {
            if map[o] == usize::MAX {
                map[o] = next;
                next += 1;
            }
            map[o] as u16
        })
        .collect();
    assert!(next < u16::MAX as usize);
    Net { ops, nslots: next, inputs: new_inputs, outputs }
}
