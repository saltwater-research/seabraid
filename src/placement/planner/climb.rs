// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! The climb: one hot expert moves up several adjacent boundaries in one round, as a chain of ordinary swaps.
//! OWNER: this file owns "how far one expert may travel in one round"; the margin, helper share cap, cooldown and
//! write budget are the planner's (`Round::displacement`, `Round::at_limit`) and mean here what they mean there.
//! CONTRACT: `Round::run_climb` is the planner's round when `max_hops > 1`. Challengers are taken hottest first across
//! the whole table; each climbs bottom-up (fallback to the last helper, then each helper to the one above), one hop
//! per boundary, and stops at the first boundary without an incumbent that passes. Every hop is a plain same-layer
//! swap, so each is committed, rejected or rolled back on its own, and the climber rests at its last committed rung.
//! Each displaced cell drops exactly one rung and is used once; a climber is never displaced in its own round.
//! WHY a climb and not a rotation: it reaches the target rung in one round with today's transaction semantics; a
//! rotation would save one write per extra hop at the price of a chain outcome and a multi-card rollback.
//! FAILURE: none of its own; it only builds a plan, and the transaction validates the plan before any I/O.
//! SIZE: under 100 lines.

use super::{Incumbents, Round};
use crate::placement::{Promotion, Rung};

impl<'a> Round<'a> {
    pub(super) fn run_climb(mut self) -> Vec<Promotion> {
        let incumbents: Vec<Incumbents> = (0..self.place.fallback)
            .map(|hi| self.incumbents(hi))
            .collect();
        for up in self.climbers() {
            if self.at_limit() {
                break;
            }
            if !self.used[up] && self.heat[up] != 0.0 {
                self.climb(up, &incumbents);
            }
        }
        self.promotions
    }

    /// Every eligible cell below the lead, hottest first (ties by flattened index, as `boundary` breaks them).
    fn climbers(&self) -> Vec<usize> {
        let mut cells: Vec<usize> = self.candidates[1..].iter().flatten().copied().collect();
        cells.sort_unstable_by(|&a, &b| self.heat[b].total_cmp(&self.heat[a]).then(a.cmp(&b)));
        cells
    }

    fn climb(&mut self, up: usize, incumbents: &[Incumbents]) {
        let mut rung = self.place.owner[up];
        for _ in 0..self.config.max_hops {
            if rung == 0 || self.at_limit() || !self.boundary_open(rung - 1) {
                return;
            }
            let hi = rung - 1;
            match self.displacement(hi, up, &incumbents[usize::from(hi)]) {
                Some(down) => self.exchange(hi, up, down),
                None => return,
            }
            rung = hi;
        }
    }

    /// Whether boundary `(hi, hi + 1)` may be planned: the fallback boundary only when enabled (and then never
    /// limited by `ladder_boundaries`), a card boundary only within the top `ladder_boundaries` of them.
    fn boundary_open(&self, hi: Rung) -> bool {
        if hi + 1 == self.place.fallback {
            return self.config.fallback_boundary;
        }
        self.config.ladder_boundaries == 0 || hi < self.config.ladder_boundaries
    }
}
