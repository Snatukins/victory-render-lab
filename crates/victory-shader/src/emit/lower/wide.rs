//! Wave-wide regions, for programs in halves ([`super::halves`]). The two
//! host waves of a console wave trade at group barriers, which every thread
//! of the group must reach, so inside a branch only some lanes take a trade
//! is refused. Such a branch can instead run wave-wide: every thread runs
//! every block of its region (the blocks before the block that
//! post-dominates the branch, its join) in order, and each block acts only
//! in the lanes the console runs there.
//!
//! - A block's lanes are those along its edges in; an edge's are those of
//!   the block it leaves that its branch sends that way.
//! - Each vector instruction's guard ([`Lower::guard`]) takes in its block's
//!   lanes: a VGPR write keeps the old value in the others, memory is
//!   accessed by those lanes only, and the wave operations that read EXEC
//!   (ballots, first lanes, DPP source lanes) see only them.
//! - A phi of a block of the region, or of its join, picks lane by lane the
//!   value along the edge the lane came by.
//! - Every block of the region is one where every thread meets
//!   ([`super::halves::Exchange`]): the halves trade there, and LDS is put
//!   in order there, as anywhere else.
//!
//! Scalar code runs in every lane, as it does in the per-lane form, which
//! keeps each lane's values; a region whose scalar code has effects beyond
//! its values (stores, atomics, messages) does not run wave-wide, nor one
//! with a loop, a way out other than its join, or an edge into it other
//! than the branch's.
//!
//! The lowering takes these regions only where an attempt was refused for
//! want of a meeting ([`Plan`]): a program that needs none is lowered as
//! before, and one that does pays for its regions only.

use std::cell::RefCell;

use super::*;

/// The regions run wave-wide, over the lowering's attempts ([`super::lower`]):
/// an attempt refused where the halves cannot meet asks for the region
/// around each block that wanted a meeting, and the next runs them
/// wave-wide.
#[derive(Default)]
pub(super) struct Plan {
    /// The branches whose regions run wave-wide, by block.
    heads: BTreeSet<usize>,
    /// The blocks the last attempt wanted a meeting at.
    asked: RefCell<BTreeSet<usize>>,
}

impl Plan {
    /// Notes that IR block `b` wanted every thread of the group to meet.
    pub(super) fn ask(&self, b: usize) {
        self.asked.borrow_mut().insert(b);
    }

    /// After an attempt refused for `what`: the regions around the blocks
    /// it asked for join the plan; whether another attempt is worth making
    /// (some region joined).
    pub(super) fn widen(&mut self, x: &Parts, what: Why) -> bool {
        let asked = std::mem::take(self.asked.get_mut());
        if asked.is_empty() || !super::halves::for_want_of_a_meeting(what) {
            return false;
        }
        let regions = regions(x);
        let mut grew = false;
        for b in asked {
            if let Some(r) = regions.iter().find(|r| r.blocks[b]) {
                grew |= self.heads.insert(r.head);
            }
        }
        grew
    }

    /// Forgets what the last attempt asked for.
    pub(super) fn forget(&mut self) {
        self.asked.get_mut().clear();
    }
}

/// A branch where every thread meets whose region can run wave-wide.
struct Region {
    head: usize,
    /// Per IR block: in the region.
    blocks: Vec<bool>,
    /// The block every way out of the region leads to.
    join: usize,
}

/// The regions of a program in halves that can run wave-wide: a per-lane
/// branch where every thread meets, whose region has no loop (each edge
/// goes forward in reverse postorder), only edges from the branch or the
/// region into it, a single way out (the join, entered only from them), no
/// end, and no scalar instruction with effects beyond its values.
fn regions(x: &Parts) -> Vec<Region> {
    let (p, st) = (x.p, x.st);
    let n = st.exits.len();
    let Some(all) = super::halves::lane_regions(x) else {
        return Vec::new();
    };
    let meet = super::halves::meeting(x);
    let mut rpo = vec![usize::MAX; n];
    for (k, b) in crate::ir::reverse_postorder(&p.blocks)
        .into_iter()
        .enumerate()
    {
        rpo[b] = k;
    }
    let reached = |b: usize| rpo[b] != usize::MAX;
    all.into_iter()
        .filter_map(|(head, blocks)| {
            if !meet[head] || !reached(head) {
                return None;
            }
            let within = |b: usize| b == head || blocks[b];
            let mut join = None;
            for u in (0..n).filter(|&u| within(u)) {
                let next = match st.exits[u] {
                    LaneExit::Jump(t) => vec![t],
                    LaneExit::Branch {
                        taken, not_taken, ..
                    } => vec![taken, not_taken],
                    LaneExit::End | LaneExit::Indirect => return None,
                };
                for t in next {
                    if rpo[t] <= rpo[u] || !within(t) && *join.get_or_insert(t) != t {
                        return None;
                    }
                }
                if u != head
                    && (p.blocks[u].preds.iter().any(|&q| reached(q) && !within(q))
                        || p.blocks[u].insts.iter().any(|i| scalar_effect(i.name())))
                {
                    return None;
                }
            }
            let join = join?;
            if p.blocks[join]
                .preds
                .iter()
                .any(|&q| reached(q) && !within(q))
            {
                return None;
            }
            Some(Region { head, blocks, join })
        })
        .collect()
}

/// Whether scalar instruction `name` has an effect beyond the values it
/// writes, which running it in lanes the console does not would repeat.
fn scalar_effect(name: &str) -> bool {
    const EFFECTS: [&str; 9] = [
        "s_store",
        "s_buffer_store",
        "s_scratch_store",
        "s_atomic",
        "s_buffer_atomic",
        "s_sendmsg",
        "s_trap",
        "s_sethalt",
        "s_dcache_wb",
    ];
    EFFECTS.iter().any(|e| name.starts_with(e))
}

/// The [`Plan`]'s regions laid out for one attempt, and their lanes as
/// lowered.
#[derive(Default)]
pub(super) struct Wide {
    /// Per IR block heading a region or in one: the block lowered after it,
    /// in reverse postorder (the join after the region's last).
    next: HashMap<usize, usize>,
    /// Per IR block: in a region run wave-wide.
    pub(super) inside: Vec<bool>,
    /// Per IR block in a region or joining one, once the block lowered
    /// before it is: that block's last DXIL block, its one predecessor.
    from: HashMap<usize, usize>,
    /// Per IR block in a region: its lanes (DXIL i1), from its start.
    lanes: HashMap<usize, V>,
    /// Per IR edge out of a region's branch or block: its lanes.
    edges: HashMap<(usize, usize), V>,
}

impl Wide {
    pub(super) fn new(x: &Parts, plan: &Plan) -> Self {
        let n = x.st.exits.len();
        let mut wide = Wide {
            inside: vec![false; n],
            ..Wide::default()
        };
        if plan.heads.is_empty() {
            return wide;
        }
        let mut rpo = vec![usize::MAX; n];
        for (k, b) in crate::ir::reverse_postorder(&x.p.blocks)
            .into_iter()
            .enumerate()
        {
            rpo[b] = k;
        }
        for r in regions(x)
            .into_iter()
            .filter(|r| plan.heads.contains(&r.head))
        {
            let mut order: Vec<usize> = (0..n).filter(|&b| r.blocks[b]).collect();
            order.sort_by_key(|&b| rpo[b]);
            let mut last = r.head;
            for b in order {
                wide.inside[b] = true;
                wide.next.insert(last, b);
                last = b;
            }
            wide.next.insert(last, r.join);
        }
        wide
    }
}

impl Lower<'_> {
    /// At the start of IR block `b`, after its phis: where `b` runs
    /// wave-wide, its lanes, those of its edges in.
    pub(super) fn wide_enter(&mut self, b: usize) {
        if !self.wide.inside.get(b).copied().unwrap_or(false) {
            return;
        }
        let c = self.cur;
        let mut on = self.s.m.bool(false);
        for &q in &self.x.p.blocks[b].preds {
            if let Some(&e) = self.wide.edges.get(&(q, b)) {
                on = self.s.m.bin(c, Bin::Or, on, e);
            }
        }
        self.wide.lanes.insert(b, on);
    }

    /// `x`, an i1 per lane in DXIL block `at`, in IR block `b`'s lanes only
    /// where it runs wave-wide.
    pub(super) fn wide_and(&mut self, b: usize, x: V, at: usize) -> V {
        match self.wide.lanes.get(&b) {
            Some(&on) => self.s.m.bin(at, Bin::And, on, x),
            None => x,
        }
    }

    /// The guard of an instruction in IR block `b` (`g`, if any), with the
    /// block's lanes where it runs wave-wide.
    pub(super) fn wide_guard(&mut self, b: usize, g: Option<V>) -> Option<V> {
        match (g, self.wide.lanes.get(&b)) {
            (Some(g), Some(_)) => Some(self.wide_and(b, g, self.cur)),
            (None, Some(&on)) => Some(on),
            (g, None) => g,
        }
    }

    /// The exit of IR block `b` where it heads a region run wave-wide or is
    /// in one: on to the block lowered next, the lanes of each edge noted;
    /// `None` elsewhere.
    pub(super) fn wide_exit(&mut self, b: usize) -> Result<Option<Term>, Why> {
        let Some(&next) = self.wide.next.get(&b) else {
            return Ok(None);
        };
        let c = self.cur;
        let on = self.wide.lanes.get(&b).copied();
        let edge = |l: &mut Self, to: usize, e: Option<V>| {
            let e = e.unwrap_or_else(|| l.s.m.bool(true));
            l.wide.edges.insert((b, to), e);
        };
        match self.x.st.exits[b] {
            LaneExit::Jump(t) => edge(self, t, on),
            LaneExit::Branch {
                taken, not_taken, ..
            } if taken == not_taken => edge(self, taken, on),
            LaneExit::Branch {
                when,
                taken,
                not_taken,
            } => {
                // A negated condition goes the other way round (as `exit`).
                let (cond, taken, not_taken) = match self.x.st.bools.node(when) {
                    Some((atom, Bool::TRUE, Bool::FALSE)) => {
                        (self.atom(atom, c)?, not_taken, taken)
                    }
                    _ => (self.boolean(when, c)?, taken, not_taken),
                };
                let t = self.s.m.bool(true);
                let not = self.s.m.bin(c, Bin::Xor, cond, t);
                let (yes, no) = match on {
                    Some(on) => (
                        self.s.m.bin(c, Bin::And, on, cond),
                        self.s.m.bin(c, Bin::And, on, not),
                    ),
                    None => (cond, not),
                };
                edge(self, taken, Some(yes));
                edge(self, not_taken, Some(no));
            }
            LaneExit::End | LaneExit::Indirect => return Ok(None),
        }
        self.wide.from.insert(next, self.cur);
        Ok(Some(Term::Br(self.block[next].expect("reachable"))))
    }

    /// Sets DXIL phi `v` of IR block `b` from `incoming`, the values along
    /// the edges from IR blocks `from`. Where `b` is in a region run
    /// wave-wide or joins one, its one DXIL predecessor brings the value
    /// along the edge each lane came by.
    pub(super) fn wide_phi(&mut self, v: V, b: usize, incoming: Vec<(V, usize)>, from: &[usize]) {
        let Some(&at) = self.wide.from.get(&b) else {
            self.s.m.set_phi(v, incoming);
            return;
        };
        let mut picked: Option<V> = None;
        for (&(x, _), &q) in incoming.iter().zip(from).rev() {
            picked = Some(match picked {
                None => x,
                Some(rest) => {
                    let e = match self.wide.edges.get(&(q, b)) {
                        Some(&e) => e,
                        None => self.s.m.bool(false),
                    };
                    self.s.m.select(at, e, x, rest)
                }
            });
        }
        let incoming = picked.map(|x| vec![(x, at)]).unwrap_or_default();
        self.s.m.set_phi(v, incoming);
    }
}
