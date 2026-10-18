"""Generate src/rank/nets.rs: straight-line selection networks for common windows.

    python src/rank/gen_networks.py > src/rank/nets.rs

The networks are built exactly like `network::build` in Rust (rows sorted by
odd-even merge sort, combined with Batcher odd-even merges, two output rows
paired so they share the merge of their common rows, then pruned backwards
from the outputs), and verified here on random and 0/1 inputs. Emitting them
as straight-line code lets the compiler keep every window element in a SIMD
register instead of interpreting the network through memory.

Per window there is a network pruned for the median, and "bucket" networks
that keep a range of ranks live (about 1.1x the cost of a single rank) and
store the one requested at run time, so every rank is covered.
"""
import itertools
import random
import sys

CE, MIN, MAX, COPY = 0, 1, 2, 3


class Builder:
    def __init__(self):
        self.ops = []
        self.nslots = 0

    def new_slot(self):
        self.nslots += 1
        return self.nslots - 1

    def ce(self, a, b):
        self.ops.append((CE, a, b))

    def merge(self, a, b):
        if not a:
            return list(b)
        if not b:
            return list(a)
        if len(a) == 1 and len(b) == 1:
            self.ce(a[0], b[0])
            return [a[0], b[0]]
        c = self.merge(a[0::2], b[0::2])
        d = self.merge(a[1::2], b[1::2])
        res = [c[0]]
        i = 0
        while i < len(d) or i + 1 < len(c):
            if i < len(d) and i + 1 < len(c):
                self.ce(d[i], c[i + 1])
                res += [d[i], c[i + 1]]
            elif i < len(d):
                res.append(d[i])
            else:
                res.append(c[i + 1])
            i += 1
        return res

    def merge_all(self, lists):
        lists = [l for l in lists if l]
        if not lists:
            return []
        while len(lists) > 1:
            lists.sort(key=lambda l: -len(l))  # stable, like Rust sort_by_key(Reverse)
            a = lists.pop()
            b = lists.pop()
            lists.append(self.merge(b, a))
        return lists.pop()

    def sort(self, group):
        return self.merge_all([[s] for s in group])


def build(rows, shift, ranks):
    """rows: list of lists of coordinate tuples (window of output 0).
    shift: coordinate delta of output 1, or None. ranks: ranks to keep.
    Returns (ops, nslots, inputs, outs) with outs[w][i] the slot holding
    rank ranks[i] of window w."""
    set_a = [c for row in rows for c in row]
    sa = set(set_a)
    b = Builder()
    slot_of = {}
    inputs = []

    def slot(c):
        if c not in slot_of:
            s = b.new_slot()
            slot_of[c] = s
            inputs.append((s, c))
        return slot_of[c]

    add = lambda c, d: tuple(x + y for x, y in zip(c, d))
    sb = set(add(c, shift) for c in set_a) if shift else set()
    core_rows, extra_a, extra_b = [], [], []
    for row in rows:
        cc, e = [], []
        for c in row:
            if shift and c in sb:
                cc.append(slot(c))
            else:
                e.append(slot(c))
        core_rows.append(cc)
        extra_a.append(e)
        if shift:
            extra_b.append([slot(add(c, shift)) for c in row if add(c, shift) not in sa])
    core = b.merge_all([b.sort(g) for g in core_rows])
    core_b = None
    if shift:
        core_b = []
        for s in core:
            d = b.new_slot()
            b.ops.append((COPY, s, d))
            core_b.append(d)
    full_a = b.merge_all([b.sort(g) for g in extra_a] + [core])
    outs = [[full_a[r] for r in ranks]]
    if shift:
        full_b = b.merge_all([b.sort(g) for g in extra_b] + [core_b])
        outs.append([full_b[r] for r in ranks])
    ops, nslots, new_inputs, flat = prune(b, inputs, [s for w in outs for s in w])
    n = len(ranks)
    return ops, nslots, new_inputs, [flat[i * n:(i + 1) * n] for i in range(len(outs))]


def prune(b, inputs, outputs):
    live = [False] * b.nslots
    for o in outputs:
        live[o] = True
    kept = []
    for kind, x, y in reversed(b.ops):
        if kind == CE:
            lx, ly = live[x], live[y]
            if lx and ly:
                kept.append((CE, x, y))
            elif lx:
                kept.append((MIN, x, y))
                live[y] = True
            elif ly:
                kept.append((MAX, x, y))
                live[x] = True
        elif kind == COPY:
            if live[y]:
                kept.append((COPY, x, y))
                live[y] = False
                live[x] = True
        else:
            raise AssertionError
    kept.reverse()
    m = {}
    nxt = 0
    new_inputs = []
    for s, c in inputs:
        if live[s]:
            m[s] = nxt
            new_inputs.append(c)
            nxt += 1
    ops = []
    for kind, x, y in kept:
        for s in (x, y):
            if s not in m:
                m[s] = nxt
                nxt += 1
        ops.append((kind, m[x], m[y]))
    outs = []
    for o in outputs:
        if o not in m:
            m[o] = nxt
            nxt += 1
        outs.append(m[o])
    return ops, nxt, new_inputs, outs


def evaluate(ops, nslots, vals, slots):
    s = list(vals) + [None] * (nslots - len(vals))
    for kind, a, b in ops:
        if kind == CE:
            s[a], s[b] = min(s[a], s[b]), max(s[a], s[b])
        elif kind == MIN:
            s[a] = min(s[a], s[b])
        elif kind == MAX:
            s[b] = max(s[a], s[b])
        else:
            s[b] = s[a]
    return [s[o] for o in slots]


def verify(net, window, shift, ranks, trials):
    ops, nslots, inputs, outs = net
    rng = random.Random(1)
    add = lambda c, d: tuple(x + y for x, y in zip(c, d))
    windows = [window] + ([[add(c, shift) for c in window]] if shift else [])
    for t in range(trials):
        hi = 1 if t < trials // 2 else 50  # 0/1 inputs, then small integers
        val = {c: rng.randint(0, hi) for w in windows for c in w}
        vals = [val[c] for c in inputs]
        for w, slots in zip(windows, outs):
            got = evaluate(ops, nslots, vals, slots)
            exp = sorted(val[c] for c in w)
            assert got == [exp[r] for r in ranks], (window, shift, ranks)


def box(shape):
    return list(itertools.product(*[range(n) for n in shape]))


def disk(r):
    """skimage.morphology.disk(r): x**2 + y**2 <= r**2, as 0-based (y, x)."""
    return [(y + r, x + r) for y in range(-r, r + 1) for x in range(-r, r + 1) if x * x + y * y <= r * r]


def rows_of(coords):
    rows = {}
    for c in coords:
        rows.setdefault(c[:-1], []).append(c)
    return list(rows.values())


def emit_fn(name, net, ndim, ranks):
    ops, nslots, inputs, outs = net
    ni = len(inputs)
    lines = []
    lines.append("#[inline(always)]")
    lines.append(f"unsafe fn {name}<T, V: Vx<T>>(p: *const T, o: &[isize], out: *mut T, ostride: isize, r: usize) {{")
    lines.append(f"    let o: &[isize; {ni}] = o.try_into().unwrap();")
    lines.append("    unsafe {")
    for i in range(ni):
        lines.append(f"        let s{i} = V::load(p.offset(o[{i}]));")
    for kind, a, b in ops:
        if kind == CE:
            lines.append(f"        let (s{a}, s{b}) = (s{a}.mn(s{b}), s{a}.mx(s{b}));")
        elif kind == MIN:
            lines.append(f"        let s{a} = s{a}.mn(s{b});")
        elif kind == MAX:
            lines.append(f"        let s{b} = s{a}.mx(s{b});")
        else:
            lines.append(f"        let s{b} = s{a};")
    stores = lambda i: " ".join(f"s{w[i]}.store(out.offset({j} * ostride));" for j, w in enumerate(outs))
    if len(ranks) == 1:
        lines.append("        let _ = r;")
        lines.append(f"        {stores(0)}")
    else:
        lines.append("        match r {")
        for i, rk in enumerate(ranks):
            lines.append(f"            {rk} => {{ {stores(i)} }}")
        lines.append("            _ => {}")
        lines.append("        }")
    lines.append("    }")
    lines.append("}")
    coords = ", ".join("[" + ", ".join(str(x) for x in (0,) * (3 - ndim) + c) + "]" for c in inputs)
    lines.append(f"const {name.upper()}_IN: [[u8; 3]; {ni}] = [{coords}];")
    cost = sum(2 if k == CE else (1 if k in (MIN, MAX) else 0) for k, _, _ in ops)
    return lines, cost


def rank_sets(k):
    """The median alone, then buckets covering every rank 1 .. k-2 (ranks 0
    and k-1 go to minimum/maximum filters in SciPy)."""
    sets = [[k // 2]]
    ranks = list(range(1, k - 1))
    if k <= 9:
        sets += [[r] for r in ranks if r != k // 2]
    else:
        q = -(-len(ranks) // 4)
        sets += [ranks[i:i + q] for i in range(0, len(ranks), q)]
    return sets


# (struct name, window coordinates)
WINDOWS = [
    ("Box3x3", box((3, 3))),
    ("Box5x5", box((5, 5))),
    ("Box7x7", box((7, 7))),
    ("Box3x3x3", box((3, 3, 3))),
] + [(f"Disk{r}", disk(r)) for r in range(1, 6)]


def main():
    out = []
    out.append("// @generated by src/rank/gen_networks.py -- do not edit.")
    out.append("//! Straight-line selection networks for common windows (boxes, disks).")
    out.append("//!")
    out.append("//! Each network loads its window elements (coordinates in `*_IN`, relative")
    out.append("//! to the window corner of the first output) as SIMD vectors and stores the")
    out.append("//! order statistic `r` for one output row (`*_s`) or two adjacent rows")
    out.append("//! along the second-to-last axis (`*_p`, outputs `ostride` apart).")
    out.append("#![allow(clippy::all, unused_unsafe)]")
    out.append("")
    out.append("use super::simd::{Vx, X2};")
    out.append("")
    out.append("/// A generated network pair (single row / two rows) for one window and a")
    out.append("/// set of ranks.")
    out.append("pub trait GenNet {")
    out.append("    /// Vector type to run with: two native vectors for small networks.")
    out.append("    type W<T, V: Vx<T>>: Vx<T>;")
    out.append("    const SINGLE_IN: &'static [[u8; 3]];")
    out.append("    const PAIR_IN: &'static [[u8; 3]];")
    out.append("    /// # Safety")
    out.append("    /// Inputs `p + o[i]` readable for `V::LANES` elements; outputs writable;")
    out.append("    /// `r` one of the network's ranks.")
    out.append("    unsafe fn single<T, V: Vx<T>>(p: *const T, o: &[isize], out: *mut T, ostride: isize, r: usize);")
    out.append("    /// # Safety")
    out.append("    /// As for `single`, with two output rows.")
    out.append("    unsafe fn pair<T, V: Vx<T>>(p: *const T, o: &[isize], out: *mut T, ostride: isize, r: usize);")
    out.append("}")
    out.append("")
    out.append("/// Receives the network type chosen by [`dispatch`].")
    out.append("pub trait Visit {")
    out.append("    type Out;")
    out.append("    fn visit<N: GenNet>(self) -> Self::Out;")
    out.append("}")
    out.append("")
    disp = []
    report = []
    total = 0
    for sname, window in WINDOWS:
        ndim = len(window[0])
        k = len(window)
        rows = rows_of(window)
        shift = tuple([0] * (ndim - 2) + [1, 0])
        wname = f"WIN_{sname.upper()}"
        wsorted = sorted((0,) * (3 - ndim) + c for c in window)
        out.append(f"const {wname}: [[u8; 3]; {k}] = [" + ", ".join("[" + ", ".join(map(str, c)) + "]" for c in wsorted) + "];")
        out.append("")
        arms = []
        for ranks in rank_sets(k):
            single = build(rows, None, ranks)
            pair = build(rows, shift, ranks)
            trials = 2000 if k <= 30 else 400
            verify(single, window, None, ranks, trials)
            verify(pair, window, shift, ranks, trials)
            tag = f"r{ranks[0]}" if len(ranks) == 1 else f"r{ranks[0]}_{ranks[-1]}"
            base = f"{sname.lower()}_{tag}"
            l1, c1 = emit_fn(base + "_s", single, ndim, ranks)
            l2, c2 = emit_fn(base + "_p", pair, ndim, ranks)
            total += c1 + c2
            out += l1 + [""] + l2 + [""]
            tname = f"{sname}{tag.upper()}"
            out.append(f"pub struct {tname};")
            out.append(f"impl GenNet for {tname} {{")
            out.append("    type W<T, V: Vx<T>> = " + ("X2<V>;" if len(pair[2]) <= 8 else "V;"))
            out.append(f"    const SINGLE_IN: &'static [[u8; 3]] = &{base.upper()}_S_IN;")
            out.append(f"    const PAIR_IN: &'static [[u8; 3]] = &{base.upper()}_P_IN;")
            out.append("    #[inline(always)]")
            out.append("    unsafe fn single<T, V: Vx<T>>(p: *const T, o: &[isize], out: *mut T, ostride: isize, r: usize) {")
            out.append(f"        unsafe {{ {base}_s::<T, V>(p, o, out, ostride, r) }}")
            out.append("    }")
            out.append("    #[inline(always)]")
            out.append("    unsafe fn pair<T, V: Vx<T>>(p: *const T, o: &[isize], out: *mut T, ostride: isize, r: usize) {")
            out.append(f"        unsafe {{ {base}_p::<T, V>(p, o, out, ostride, r) }}")
            out.append("    }")
            out.append("}")
            out.append("")
            cond = f"{ranks[0]}" if len(ranks) == 1 else f"{ranks[0]}..={ranks[-1]}"
            arms.append((cond, tname))
            report.append(f"{sname} ranks {ranks[0]}..{ranks[-1]}: single {c1} ops, pair {c2} ops ({c2 / 2:.1f}/output)")
        disp.append((k, wname, arms))
    out.append("/// Call `v.visit::<N>()` for the generated network of the window with")
    out.append("/// (lexicographically sorted, `[z, y, x]`) coordinates `win` that computes")
    out.append("/// `rank`, if any.")
    out.append("pub fn dispatch<V: Visit>(win: &[[u8; 3]], rank: usize, v: V) -> Option<V::Out> {")
    for k, wname, arms in disp:
        out.append(f"    if win.len() == {k} && win == &{wname}[..] {{")
        out.append("        return match rank {")
        for cond, tname in arms:
            out.append(f"            {cond} => Some(v.visit::<{tname}>()),")
        out.append("            _ => None,")
        out.append("        };")
        out.append("    }")
    out.append("    None")
    out.append("}")
    sys.stdout.write("\n".join(out) + "\n")
    for line in report:
        print(line, file=sys.stderr)
    print(f"total ops {total}", file=sys.stderr)


if __name__ == "__main__":
    main()
