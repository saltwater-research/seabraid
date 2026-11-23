// Copyright 2026 Saltwater Research LLC
// SPDX-License-Identifier: Apache-2.0
//! Validates work geometry before allocating or reading caller buffers, and owns the batch storage.
//!
//! OWNER: the activation formats (element size, word storage) and the geometry (counts, extents) of a batch. It
//! hands the picks to `picks::check` and decides nothing about them: what a pick may be lives in `src/picks.rs`.
//! FAILURE: fail-fast `Code::Invalid`, each reason logged through `fail!`.

use crate::capabilities::{FORMAT_ROUTED_BF16, FORMAT_ROUTED_F16, FORMAT_ROUTED_F32};
use crate::diag::{fail, Code};
use crate::picks::PickRule;
use crate::rows::RowMap;
use crate::submission::Submission;
use crate::Result;

/// The C `seabraid_config_v1` itself, not a Rust-side copy: callers pass it by pointer and the core uses the same
/// value. WHY one type: a second Rust config would be a second definition of the same layout to keep in step; the
/// ABI version header is checked by the ffi before this struct is read (`Config::validate` checks only geometry).
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Config {
    pub abi_version: u32,
    pub hidden: u32,
    pub experts: u32,
    pub max_tokens: u32,
    pub top_k: u32,
}

crate::layout::assert_layout!(
    Config,
    "seabraid_config_v1",
    [abi_version, hidden, experts, max_tokens, top_k]
);

impl Config {
    pub fn validate(self) -> Result<Self> {
        if self.experts == 0 || self.top_k > self.experts {
            return Err(fail!(
                Code::Invalid,
                "config top_k {} must not exceed experts {} (experts >= 1)",
                self.top_k,
                self.experts
            ));
        }
        self.lengths(self.max_tokens)?;
        Ok(self)
    }

    pub fn lengths(self, tokens: u32) -> Result<Lengths> {
        if tokens == 0 || tokens > self.max_tokens {
            return Err(fail!(
                Code::Invalid,
                "tokens {} outside 1..={}",
                tokens,
                self.max_tokens
            ));
        }
        Ok(Lengths {
            rows: element_count(tokens, self.hidden, "hidden")?,
            picks: element_count(tokens, self.top_k, "top_k")?,
        })
    }
}

fn element_count(tokens: u32, width: u32, name: &str) -> Result<usize> {
    // Both wire elements are 32 bits. The byte extent must also fit Rust's slice limit.
    let elements = u64::from(tokens) * u64::from(width);
    if elements == 0 || elements > (isize::MAX as u64 / 4) {
        return Err(fail!(
            Code::Invalid,
            "tokens {} * {} {} is zero or too large to address",
            tokens,
            name,
            width
        ));
    }
    Ok(elements as usize)
}

#[derive(Clone, Copy, Debug)]
pub struct Lengths {
    pub rows: usize,
    pub picks: usize,
}

impl Lengths {
    /// Counts are `[activations, expert_ids, weights]` as the caller supplied them.
    pub fn check(self, tokens: u32, config: Config, counts: [usize; 3]) -> Result<()> {
        let [activations, ids, weights] = counts;
        if activations != self.rows {
            return Err(fail!(
                Code::Invalid,
                "activation_count {} != tokens {} * hidden {}",
                activations,
                tokens,
                config.hidden
            ));
        }
        if ids != self.picks || weights != self.picks {
            return Err(fail!(
                Code::Invalid,
                "expert_id_count {} and weight_count {} must both equal tokens {} * top_k {}",
                ids,
                weights,
                tokens,
                config.top_k
            ));
        }
        Ok(())
    }
}

/// What the provider declared at open. WHY one struct: the format and row rules read the same four bits, and a
/// session must refuse an impossible combination once, at open, not on every submit.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Features {
    pub partial_routes: bool,
    pub compact_rows: bool,
    pub accepts_f16: bool,
    pub accepts_bf16: bool,
}

impl Features {
    /// COMPACT_ROWS drops rows whose picks are all skipped, which only exists for partial-route providers.
    pub(crate) fn validate(self) -> Result<Self> {
        if self.compact_rows && !self.partial_routes {
            return Err(fail!(
                Code::Invalid,
                "COMPACT_ROWS requires PARTIAL_ROUTES: without skip markers no row can be dropped"
            ));
        }
        Ok(self)
    }

    /// Element size in bytes of an activation format this provider accepts; F32 is always accepted.
    fn element_size(self, format: u32) -> Result<usize> {
        let declared = match format {
            FORMAT_ROUTED_F32 => return Ok(4),
            FORMAT_ROUTED_F16 => self.accepts_f16,
            FORMAT_ROUTED_BF16 => self.accepts_bf16,
            _ => {
                return Err(fail!(
                    Code::Invalid,
                    "activation format {} is unknown",
                    format
                ))
            }
        };
        if declared {
            Ok(2)
        } else {
            Err(fail!(
                Code::Invalid,
                "activation format {} was not declared by the provider (ACCEPTS_F16 / ACCEPTS_BF16)",
                format
            ))
        }
    }
}

/// The batch a provider sees. `activations` holds raw elements of `format` packed into u32 words, never bytes,
/// so a provider always receives a 4-aligned pointer; F16 and BF16 are passed through, never converted.
#[derive(Debug, Default)]
pub struct Batch {
    pub tokens: u32,
    pub format: u32,
    pub placement_epoch: u64,
    /// The origins the caller declared for this submission. WHY here: the v3 provider's view reports them, and the
    /// worker hands a provider only the `Batch`, so this is the one place they can travel with the rows.
    pub source_modalities: u32,
    pub activations: Vec<u32>,
    /// Elements (not words, not bytes): `tokens * hidden`.
    pub activation_count: usize,
    pub expert_ids: Vec<i32>,
    pub weights: Vec<f32>,
}

impl Batch {
    /// Validates everything about a submission that needs no copy: format, counts and every pick.
    pub(crate) fn check(config: Config, s: &Submission<'_>, features: Features) -> Result<()> {
        let size = features.element_size(s.format)?;
        let lengths = config.lengths(s.tokens)?;
        if s.activations.len() % size != 0 {
            return Err(fail!(
                Code::Invalid,
                "activation byte count {} is not a multiple of the {}-byte element",
                s.activations.len(),
                size
            ));
        }
        let counts = [
            s.activations.len() / size,
            s.expert_ids.len(),
            s.weights.len(),
        ];
        lengths.check(s.tokens, config, counts)?;
        crate::picks::check(
            (s.expert_ids, s.weights),
            config.top_k as usize,
            PickRule {
                experts: config.experts,
                partial: features.partial_routes,
            },
        )
    }

    /// Copies a checked submission, keeping only the rows of `rows` when it is given. WHY the caller checks
    /// first: the slicing below relies on the counts `check` proved.
    pub(crate) fn load(
        &mut self,
        config: Config,
        s: &Submission<'_>,
        rows: Option<&RowMap>,
    ) -> Result<()> {
        let hidden = config.hidden as usize;
        let row_bytes = s.activations.len() / s.tokens as usize;
        let tokens = match rows {
            None => {
                pack_words(&mut self.activations, s.activations.len(), [s.activations])?;
                copy_into(&mut self.expert_ids, s.expert_ids)?;
                copy_into(&mut self.weights, s.weights)?;
                s.tokens
            }
            Some(map) => {
                let parts = map
                    .rows()
                    .iter()
                    .map(|&row| &s.activations[row as usize * row_bytes..][..row_bytes]);
                let total = map.rows().len() * row_bytes;
                pack_words(&mut self.activations, total, parts)?;
                let top_k = config.top_k as usize;
                map.gather(s.expert_ids, top_k, &mut self.expert_ids)?;
                map.gather(s.weights, top_k, &mut self.weights)?;
                map.kept()
            }
        };
        self.tokens = tokens;
        self.activation_count = tokens as usize * hidden;
        self.format = s.format;
        self.placement_epoch = s.placement_epoch;
        self.source_modalities = s.source_modalities;
        Ok(())
    }

    /// Activation `index` read as F32; only meaningful when `format` is `FORMAT_ROUTED_F32`.
    pub fn f32_at(&self, index: usize) -> f32 {
        f32::from_bits(self.activations[index])
    }
}

/// Packs byte slices back to back into native-endian words; the last word is zero padded.
struct Packer<'a> {
    words: &'a mut Vec<u32>,
    carry: [u8; 4],
    held: usize,
}

impl Packer<'_> {
    fn push(&mut self, mut bytes: &[u8]) {
        if self.held > 0 {
            let take = (4 - self.held).min(bytes.len());
            self.carry[self.held..self.held + take].copy_from_slice(&bytes[..take]);
            self.held += take;
            bytes = &bytes[take..];
            if self.held == 4 {
                self.words.push(u32::from_ne_bytes(self.carry));
                self.held = 0;
            }
        }
        let mut chunks = bytes.chunks_exact(4);
        let whole = chunks
            .by_ref()
            .map(|c| u32::from_ne_bytes([c[0], c[1], c[2], c[3]]));
        self.words.extend(whole);
        let rest = chunks.remainder();
        if !rest.is_empty() {
            self.carry[..rest.len()].copy_from_slice(rest);
            self.held = rest.len();
        }
    }

    fn finish(self) {
        if self.held > 0 {
            let mut last = self.carry;
            last[self.held..].fill(0);
            self.words.push(u32::from_ne_bytes(last));
        }
    }
}

/// `total` is the byte sum of `parts`, so the one reservation covers every push.
fn pack_words<'a>(
    words: &mut Vec<u32>,
    total: usize,
    parts: impl IntoIterator<Item = &'a [u8]>,
) -> Result<()> {
    words.clear();
    words
        .try_reserve_exact(total.div_ceil(4))
        .map_err(|source| {
            fail!(
                Code::Internal,
                "cannot reserve {} activation bytes ({})",
                total,
                source
            )
        })?;
    let mut packer = Packer {
        words,
        carry: [0; 4],
        held: 0,
    };
    parts.into_iter().for_each(|part| packer.push(part));
    packer.finish();
    Ok(())
}

fn copy_into<T: Copy>(storage: &mut Vec<T>, data: &[T]) -> Result<()> {
    storage
        .try_reserve_exact(data.len().saturating_sub(storage.len()))
        .map_err(|source| {
            fail!(
                Code::Internal,
                "cannot reserve {} elements ({})",
                data.len(),
                source
            )
        })?;
    storage.clear();
    storage.extend_from_slice(data);
    Ok(())
}

// This deterministic reference operation exercises weighted routing without model dependencies.
pub fn reference(config: Config, batch: &Batch, output: &mut [f32]) -> Result<()> {
    let hidden = config.hidden as usize;
    let top_k = config.top_k as usize;
    for (token, row) in output.chunks_exact_mut(hidden).enumerate() {
        let picks = token * top_k..(token + 1) * top_k;
        for (col, value) in row.iter_mut().enumerate() {
            let activation = batch.f32_at(token * hidden + col);
            // WHY an explicit fold from -0.0: `Iterator::sum` for f32 starts at -0.0 only since rustc 1.83 (earlier
            // toolchains, and our MSRV 1.79, start at +0.0), so an all-zero result changed sign with the compiler.
            *value = picks
                .clone()
                .map(|pick| {
                    let expert_scale = batch.expert_ids[pick] as f32 + 1.0;
                    batch.weights[pick] * (activation * expert_scale)
                })
                .fold(-0.0_f32, |total, term| total + term);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
