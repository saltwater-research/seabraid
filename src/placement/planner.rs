// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! The live, paired subset of `strata-ext/src/tier_ladder.cpp` and
//! `src/_retier_ladder.cpp`'s `to_pairs`: top boundary first, hottest challenger
//! first. Invalid input leaves policy state untouched. Fills and host/cache
//! adaptation use other primitives and are never proposed as live swaps.
//! FAILURE: fail-fast `PlacementError` before any state changes (the line above), so a refused plan or record
//! leaves the planner as it was.

mod climb;

use super::heat::layer_totals;
use super::{
    AppliedRound, MoveOutcome, PlacementError, PlacementPlan, PlacementSnapshot, Promotion, Rung,
};
use std::collections::HashMap;

const HEAT_DECAY: f64 = 0.75;
const MAX_MOVES: u16 = 128; // 64 exchanges in the legacy transport; two expert moves per exchange
const MAX_HOPS: u8 = 64;

#[derive(Clone, Copy, Debug)]
pub struct PlannerConfig {
    pub max_moves_per_round: u16,
    pub promote_margin: f64,
    pub cooldown_rounds: u64,
    pub max_helper_layer_share: f64,
    /// Limit to the highest card boundaries. Zero allows every card boundary.
    pub ladder_boundaries: Rung,
    /// Plan swaps between the last helper and the fallback (needs a map that enabled the fallback boundary and two
    /// cards). It never counts against `ladder_boundaries`.
    pub fallback_boundary: bool,
    /// Adjacent swaps one expert may chain in one round (a climb). 1 plans each boundary on its own.
    pub max_hops: u8,
}

impl Default for PlannerConfig {
    fn default() -> Self {
        Self {
            max_moves_per_round: 4,
            promote_margin: 1.25,
            cooldown_rounds: 8,
            max_helper_layer_share: 0.08,
            ladder_boundaries: 0,
            fallback_boundary: false,
            max_hops: 1,
        }
    }
}

/// Policy keeps smoothed observations and cooldowns, never provider residency.
pub struct PlacementPlanner {
    config: PlannerConfig,
    round: u64,
    shape: Option<(u32, u32)>,
    heat: Vec<f64>,
    moved_at: Vec<u64>, // zero means no acknowledged move
}

impl PlacementPlanner {
    pub fn new(config: PlannerConfig) -> Result<Self, PlacementError> {
        if !valid(config) {
            return Err(PlacementError::InvalidPolicy);
        }
        Ok(Self {
            config,
            round: 0,
            shape: None,
            heat: Vec::new(),
            moved_at: Vec::new(),
        })
    }

    /// The engine chooses the cadence (128 decode windows in Strata), and
    /// provides all-rung routing counts in layer-major order.
    pub fn plan(
        &mut self,
        place: &PlacementSnapshot,
        heat: &[f64],
    ) -> Result<PlacementPlan, PlacementError> {
        let shape = place.shape();
        if self.shape.is_some_and(|previous| previous != shape) {
            return Err(PlacementError::PlannerShapeChanged);
        }
        if heat.len() != place.owner.len() || heat.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err(PlacementError::InvalidHeat);
        }
        self.check_fallback_boundary(place)?;
        // WHY the raw window, not only the blend `Round::new` checks: a blend of finite values never overflows when
        // the window did not, but the reverse holds (0.75 * old + 0.25 * new halves a huge window), so only this
        // check makes plan refuse exactly what `fill` refuses, on every window and not just the first.
        layer_totals(heat, place.experts as usize)?;
        let round = self
            .round
            .checked_add(1)
            .ok_or(PlacementError::EpochExhausted)?;
        let heat = self.smooth(heat);
        let eligible = |index| match self.moved_at.get(index).copied().unwrap_or(0) {
            0 => true,
            last => round - last > self.config.cooldown_rounds,
        };
        let promotions = Round::new(place, &heat, self.config, eligible)?.run();
        self.round = round;
        self.shape = Some(shape);
        self.moved_at.resize(heat.len(), 0);
        self.heat = heat;
        Ok(PlacementPlan {
            epoch: place.epoch(),
            round,
            promotions,
        })
    }

    /// WHY refused at plan: a planner that plans fallback swaps for a map that would refuse them at apply is a
    /// mismatch of two settings, and it should fail here, before the round, with no state changed.
    fn check_fallback_boundary(&self, place: &PlacementSnapshot) -> Result<(), PlacementError> {
        if self.config.fallback_boundary && !place.fallback_live {
            return Err(PlacementError::InvalidPolicy);
        }
        Ok(())
    }

    /// Only acknowledged exchanges enter cooldown. A refusal or restoration
    /// can be retried after the next observation; quarantines are in the map.
    pub fn record(&mut self, applied: &AppliedRound) -> Result<(), PlacementError> {
        let (layers, experts) = self.shape.ok_or(PlacementError::PlannerShapeChanged)?;
        if applied.shape != (layers, experts) || applied.round > self.round {
            return Err(PlacementError::PlannerShapeChanged);
        }
        for entry in applied.outcomes() {
            if matches!(entry.outcome, MoveOutcome::Committed { .. }) {
                for cell in entry.promotion.cells() {
                    let index = cell.layer as usize * experts as usize + cell.expert as usize;
                    self.moved_at[index] = self.moved_at[index].max(applied.round);
                }
            }
        }
        Ok(())
    }

    fn smooth(&self, heat: &[f64]) -> Vec<f64> {
        if self.heat.is_empty() {
            return heat.to_vec(); // seed with raw heat to avoid startup bias
        }
        self.heat
            .iter()
            .zip(heat)
            .map(|(old, new)| HEAT_DECAY * old + (1.0 - HEAT_DECAY) * new)
            .collect()
    }
}

fn valid(config: PlannerConfig) -> bool {
    config.max_moves_per_round <= MAX_MOVES
        && config.promote_margin.is_finite()
        && config.promote_margin >= 1.0
        && (0.0..=1.0).contains(&config.max_helper_layer_share)
        && (1..=MAX_HOPS).contains(&config.max_hops)
}

/// Per layer, the candidate cells of one rung, coldest first (ties by index).
type Incumbents = HashMap<usize, Vec<usize>>;
type Find<'a> = fn(&Round<'a>, Rung, usize, &Incumbents) -> Option<usize>;

struct Round<'a> {
    place: &'a PlacementSnapshot,
    heat: &'a [f64],
    config: PlannerConfig,
    width: usize,
    candidates: Vec<Vec<usize>>,
    total: Vec<f64>,
    held: HashMap<(Rung, usize), f64>,
    used: Vec<bool>,
    promotions: Vec<Promotion>,
}

impl<'a> Round<'a> {
    fn new(
        place: &'a PlacementSnapshot,
        heat: &'a [f64],
        config: PlannerConfig,
        eligible: impl Fn(usize) -> bool,
    ) -> Result<Self, PlacementError> {
        let width = place.experts as usize;
        let total = layer_totals(heat, width)?;
        let mut candidates = vec![Vec::new(); place.fallback as usize + 1];
        let mut held = HashMap::new();
        for (i, &rung) in place.owner.iter().enumerate() {
            // A cell owned by a closed card still executes on the fallback but is never a candidate: a rebuild
            // needs its true owner.
            if place.is_quarantined(rung) {
                continue;
            }
            if rung == place.fallback {
                if config.fallback_boundary && eligible(i) {
                    candidates[rung as usize].push(i);
                }
                continue;
            }
            *held.entry((rung, i / width)).or_insert(0.0) += heat[i];
            if eligible(i) {
                candidates[rung as usize].push(i);
            }
        }
        Ok(Self {
            place,
            heat,
            config,
            width,
            candidates,
            total,
            held,
            used: vec![false; heat.len()],
            promotions: Vec::new(),
        })
    }

    fn run(self) -> Vec<Promotion> {
        if self.config.max_hops > 1 {
            return self.run_climb();
        }
        self.run_with(Round::displacement)
    }

    /// WHY the finder is a parameter: the unit tests run the same round with the pre-fix linear scan
    /// (`tests::scan_displacement`) and require identical plans; production always passes `displacement`.
    fn run_with(mut self, find: Find<'a>) -> Vec<Promotion> {
        let cards = self.place.fallback.saturating_sub(1);
        let mut boundaries = cards;
        if self.config.ladder_boundaries != 0 {
            boundaries = boundaries.min(self.config.ladder_boundaries);
        }
        for hi in 0..boundaries {
            if self.at_limit() {
                break;
            }
            self.boundary(hi, find);
        }
        // The fallback boundary goes last, so it gets what budget the promotions toward main left.
        if self.config.fallback_boundary && !self.at_limit() {
            self.boundary(cards, find);
        }
        self.promotions
    }

    fn boundary(&mut self, hi: Rung, find: Find<'a>) {
        let mut challengers = self.candidates[(hi + 1) as usize].clone();
        if challengers.is_empty() {
            return;
        }
        // Flattened index breaks exact heat ties deterministically, as does
        // the upstream stable sort over layer-major indices.
        challengers
            .sort_unstable_by(|&a, &b| self.heat[b].total_cmp(&self.heat[a]).then(a.cmp(&b)));
        let incumbents = self.incumbents(hi);
        for up in challengers {
            if self.at_limit() {
                break;
            }
            if self.used[up] || self.heat[up] == 0.0 {
                continue;
            }
            if let Some(down) = find(self, hi, up, &incumbents) {
                self.exchange(hi, up, down);
            }
        }
    }

    fn incumbents(&self, hi: Rung) -> Incumbents {
        let mut layers = Incumbents::new();
        for &index in &self.candidates[hi as usize] {
            layers.entry(index / self.width).or_default().push(index);
        }
        for cells in layers.values_mut() {
            cells.sort_unstable_by(|&a, &b| self.heat[a].total_cmp(&self.heat[b]).then(a.cmp(&b)));
        }
        layers
    }

    /// The coldest incumbent of `up`'s layer that `up` may displace, or none. Equal to the linear scan it replaced
    /// (kept as the oracle in `tests::scan_displacement`), without visiting every incumbent for every challenger.
    ///
    /// WHY a binary search is exact: `incumbents` is sorted by ascending heat, and `below_ceiling` is monotone in
    /// that order. It is `(before - heat[down]) + heat[up] <= cap`; for fixed `before`, `up` and `cap`, IEEE
    /// subtraction and addition are monotone (a larger `heat[down]` never gives a larger result), and so is the
    /// comparison, so once it holds for a cell it holds for every warmer one. `partition_point` therefore finds the
    /// first cell that passes, and the old `find` could only have rejected the cells before it. The margin test
    /// holds for a prefix of the same order (`take_while`). What is left to step over is cells already used this
    /// round: at most two per accepted exchange, so at most `MAX_MOVES` (128) in all, a constant.
    fn displacement(&self, hi: Rung, up: usize, incumbents: &Incumbents) -> Option<usize> {
        let layer = up / self.width;
        let cells = incumbents.get(&layer)?;
        // Rung 0 has no helper cap, so every cell passes and the search starts at the coldest.
        let first_passing = if hi == 0 {
            0
        } else {
            let before = self.held[&(hi, layer)];
            cells.partition_point(|&down| !self.below_ceiling(before, layer, up, down))
        };
        // A warmer incumbent may raise a helper's share by less than the coldest one; margin is the limit at
        // which none is eligible anymore.
        cells[first_passing..]
            .iter()
            .copied()
            .take_while(|&down| self.heat[up] > self.config.promote_margin * self.heat[down])
            .find(|&down| !self.used[down])
    }

    /// Whether swapping `down` out of a helper that holds `before` of `layer`'s heat, and `up` in, stays within
    /// the helper cap.
    fn below_ceiling(&self, before: f64, layer: usize, up: usize, down: usize) -> bool {
        before - self.heat[down] + self.heat[up]
            <= self.config.max_helper_layer_share * self.total[layer]
    }

    fn exchange(&mut self, hi: Rung, up: usize, down: usize) {
        let layer = up / self.width;
        self.promotions.push(Promotion {
            layer: layer as u32,
            promote: (up % self.width) as u32,
            displace: (down % self.width) as u32,
            upper_rung: hi,
        });
        // A moved cell cannot enter another boundary this round. We can keep
        // candidates tied to the original snapshot and update only its heat totals.
        self.used[up] = true;
        self.used[down] = true;
        *self.held.entry((hi, layer)).or_default() += self.heat[up] - self.heat[down];
        *self.held.entry((hi + 1, layer)).or_default() += self.heat[down] - self.heat[up];
    }

    fn at_limit(&self) -> bool {
        self.promotions.len() >= usize::from(self.config.max_moves_per_round / 2)
    }
}

#[cfg(test)]
mod tests;
