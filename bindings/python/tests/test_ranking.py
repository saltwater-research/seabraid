# Copyright 2026 Saltwater Research LLC
# SPDX-License-Identifier: Apache-2.0
"""seabraid.ranking: input validation, the bandwidth table's lookup order, and force_main.

OWNER   : the checker of seabraid.ranking: its FAILURE contract, bandwidth_gbs, and the rules R1, R2, R3a and R3b and
          force_main (the rule text is in the module docstring of seabraid.ranking, which ships in the wheel).
          The compat add-on's test_plan_rank.py tests how the installer uses it; it is not this module's checker.
"""

from __future__ import annotations

import unittest
from typing import NamedTuple

from seabraid import ranking

GIB = 1 << 30


class Card(NamedTuple):
    key: tuple
    name: str | None
    arch: str
    free: int | None
    total: int | None
    pcie: float


def card(index: int = 0, **changes: object) -> Card:
    base = Card(("cuda", index), "RTX 3060", "sm_86", 11 * GIB, 12 * GIB, 16)
    return base._replace(**changes)


def fits(_card: object) -> bool:
    return True


class BandwidthTests(unittest.TestCase):
    def test_a_3060_ti_is_not_a_3060(self) -> None:
        # "3060" is a substring of "RTX 3060 Ti"; the Ti (256-bit x 14 Gbps = 448 GB/s) needs its own, earlier row.
        self.assertEqual(ranking.bandwidth_gbs("NVIDIA GeForce RTX 3060 Ti"), 448.0)
        self.assertEqual(ranking.bandwidth_gbs("NVIDIA GeForce RTX 3060"), 360.0)
        self.assertEqual(ranking.bandwidth_gbs("rtx 3060 ti"), 448.0)

    def test_the_more_specific_row_precedes_the_general_one(self) -> None:
        names = list(ranking._BANDWIDTH_GBS)
        self.assertLess(names.index("3060 ti"), names.index("3060"))
        self.assertLess(names.index("4060 ti"), names.index("4060"))

    def test_unknown_and_missing_names_are_zero(self) -> None:
        self.assertEqual((ranking.bandwidth_gbs("Mystery"), ranking.bandwidth_gbs(None)), (0.0, 0.0))


class RankValidationTests(unittest.TestCase):
    def test_well_formed_input_still_ranks(self) -> None:
        big = card(0, total=24 * GIB, free=23 * GIB, name="RTX 3090")
        ordered, basis = ranking.rank([card(1), big], None, None, fits)
        self.assertEqual([c.key for c in ordered], [("cuda", 0), ("cuda", 1)])
        self.assertEqual((basis.card1, basis.rest), ("proxy", "proxy"))
        self.assertEqual(ranking.rank([], None, None, fits)[0], [])

    def test_unusable_card_fields_raise_a_clear_error_not_an_obscure_one(self) -> None:
        for label, bad, error in (
            ("pcie None", card(pcie=None), TypeError),
            ("arch None", card(arch=None), TypeError),
            ("free str", card(free="11"), TypeError),
            ("total float", card(total=1.5), TypeError),
            ("name int", card(name=7), TypeError),
            ("key list", card(key=["cuda", 0]), TypeError),
        ):
            with self.subTest(label), self.assertRaises(error) as raised:
                ranking.rank([bad, card(1)], None, None, fits)
            self.assertIn("card", str(raised.exception))

    def test_a_card_without_the_documented_attributes_is_a_type_error(self) -> None:
        with self.assertRaisesRegex(TypeError, r"has no \.key"):
            ranking.rank([object()], None, None, fits)
        with self.assertRaises(TypeError):
            ranking.rank(7, None, None, fits)

    def test_none_free_total_and_name_stay_supported(self) -> None:
        ordered, _ = ranking.rank([card(0, free=None, total=None, name=None), card(1)], None, None, fits)
        self.assertEqual(len(ordered), 2)

    def test_duplicate_keys_are_a_value_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate card key"):
            ranking.rank([card(0), card(0)], None, None, fits)

    def test_measurement_maps_must_be_mappings_or_none(self) -> None:
        for label, args in (("bench_lead", ([1.0, 2.0], None)), ("bench_rows", (None, [1.0, 2.0]))):
            with self.subTest(label), self.assertRaisesRegex(TypeError, label):
                ranking.rank([card(0), card(1)], *args, fits)

    def test_fits_main_and_concurrency_are_checked(self) -> None:
        with self.assertRaisesRegex(TypeError, "fits_main"):
            ranking.rank([card(0)], None, None, None)
        for bad in ("1", 1.5, True, None):
            with self.subTest(bad), self.assertRaisesRegex(TypeError, "concurrency"):
                ranking.rank([card(0)], None, None, fits, concurrency=bad)

    def test_an_unknown_force_main_is_refused_instead_of_ignored(self) -> None:
        with self.assertRaisesRegex(ValueError, "force_main"):
            ranking.rank([card(0), card(1)], None, None, fits, force_main=("cuda", 9))
        with self.assertRaisesRegex(ValueError, "force_main"):
            ranking.rank([], None, None, fits, force_main=("cuda", 0))

    def test_a_known_force_main_wins_card_one(self) -> None:
        small = card(1, total=8 * GIB, free=7 * GIB)
        ordered, basis = ranking.rank([card(0), small], None, None, fits, force_main=small.key)
        self.assertEqual((ordered[0].key, basis.card1), (small.key, "user"))

    def test_missing_or_junk_measurements_still_fall_back_to_the_proxy_without_raising(self) -> None:
        a, b = card(0), card(1)
        for lead in ({a.key: 1.0}, {a.key: float("nan"), b.key: 2.0}, {a.key: True, b.key: 2.0}, {a.key: "x", b.key: 2.0}):
            with self.subTest(lead):
                self.assertEqual(ranking.rank([a, b], lead, None, fits)[1].card1, "proxy")


class RuleTests(unittest.TestCase):
    """R1 (card 1), R2 (cards 2..N) and R3 (admission), one fact each."""

    def test_r1_the_lowest_lead_time_among_cards_that_fit_is_card_one(self) -> None:
        a, b = card(0), card(1)
        ordered, basis = ranking.rank([a, b], {a.key: 30.0, b.key: 20.0}, None, fits)
        self.assertEqual((ordered[0].key, basis.card1), (b.key, "lead"))

    def test_r1_a_card_that_cannot_hold_main_is_never_card_one_however_fast_it_measured(self) -> None:
        held, other = card(0, free=1 * GIB), card(1)
        ordered, _ = ranking.rank([held, other], {held.key: 1.0, other.key: 99.0}, None, lambda c: c.free > 2 * GIB)
        self.assertEqual([c.key for c in ordered], [other.key, held.key])

    def test_r1_the_proxy_order_is_vram_then_bandwidth_then_architecture_then_pcie(self) -> None:
        big = card(0, total=24 * GIB, name="RTX 3090")
        small = card(1, total=12 * GIB, name="RTX 4090", arch="sm_89")
        self.assertEqual(ranking.rank([small, big], None, None, fits)[0][0].key, big.key)  # VRAM beats bandwidth
        fast = card(2, name="RTX 4090")
        slow = card(3, name="RTX 3060")
        self.assertEqual(ranking.rank([slow, fast], None, None, fits)[0][0].key, fast.key)  # equal VRAM: bandwidth
        newer = card(4, name="Mystery", arch="sm_89")
        older = card(5, name="Mystery", arch="sm_80")
        self.assertEqual(ranking.rank([older, newer], None, None, fits)[0][0].key, newer.key)  # then architecture
        wide, narrow = card(6, name="Mystery", pcie=16), card(7, name="Mystery", pcie=4)
        self.assertEqual(ranking.rank([narrow, wide], None, None, fits)[0][0].key, wide.key)  # then PCIe

    def test_r1_a_lead_measurement_applies_only_to_the_regimes_it_describes(self) -> None:
        self.assertEqual((ranking.lead_class(1), ranking.lead_class(2), ranking.lead_class(3), ranking.lead_class(4)),
                         (1, None, None, 4))
        a, b = card(0), card(1)
        _, basis = ranking.rank([a, b], {a.key: 30.0, b.key: 20.0}, None, fits, concurrency=2)
        self.assertEqual(basis.card1, "proxy")

    def test_r2_cards_after_the_first_are_ordered_by_rows_per_second_then_free_vram(self) -> None:
        first, x, y, z = card(0), card(1, free=5 * GIB), card(2, free=9 * GIB), card(3, free=7 * GIB)
        rows = {first.key: 1.0, x.key: 100.0, y.key: 100.0, z.key: 200.0}
        ordered, basis = ranking.rank([first, x, y, z], None, rows, fits, force_main=first.key)
        self.assertEqual([c.key for c in ordered], [first.key, z.key, y.key, x.key])  # 200, then the 100s by free VRAM
        self.assertEqual(basis.rest, "rows")

    def test_r2_rows_per_second_never_decides_card_one(self) -> None:
        a, b = card(0, total=24 * GIB), card(1, total=12 * GIB)
        ordered, _ = ranking.rank([a, b], None, {a.key: 1.0, b.key: 1000.0}, fits)
        self.assertEqual(ordered[0].key, a.key)

    def test_r3a_a_card_that_holds_nothing_the_earlier_cards_could_not_is_excluded(self) -> None:
        slots = [ranking.Slot(("cuda", 0), "big", 30 * GIB), ranking.Slot(("cuda", 1), "small", 8 * GIB)]
        admitted, excluded = ranking.admit(slots, 20 * GIB)
        self.assertEqual([a.slot.name for a in admitted], ["big"])
        self.assertEqual(excluded[0][0], ("cuda", 1))
        self.assertTrue(excluded[0][1].startswith("R3a"))

    def test_r3a_a_card_is_admitted_for_the_bytes_that_would_otherwise_spill(self) -> None:
        slots = [ranking.Slot(("cuda", 0), "big", 12 * GIB), ranking.Slot(("cuda", 1), "small", 8 * GIB)]
        admitted, excluded = ranking.admit(slots, 16 * GIB)
        self.assertEqual((len(admitted), excluded), (2, []))
        self.assertEqual(admitted[1].held_bytes, 4 * GIB)
        self.assertIn("R3b not evaluated", admitted[1].note)  # no floors given: said aloud, not assumed

    def test_r3b_a_card_whose_layer_is_slower_with_it_than_without_is_excluded_with_its_terms(self) -> None:
        slots = [ranking.Slot(("cuda", 0), "big", 12 * GIB), ranking.Slot(("cuda", 1), "small", 8 * GIB)]
        slow = ranking.Terms(host_us=60.0, hop_us=130.0, t1_us=50.0, tk_us=200.0, pull_us=30.0)
        _, excluded = ranking.admit(slots, 16 * GIB, {("cuda", 1): slow})
        self.assertEqual(excluded[0][0], ("cuda", 1))
        self.assertTrue(excluded[0][1].startswith("R3b"))
        fast = ranking.Terms(host_us=10.0, hop_us=10.0, t1_us=50.0, tk_us=20.0, pull_us=500.0)
        admitted, _ = ranking.admit(slots, 16 * GIB, {("cuda", 1): fast})
        self.assertEqual(len(admitted), 2)

    def test_r3b_the_layer_cost_is_host_plus_the_larger_of_t1_and_hop_plus_tk(self) -> None:
        cost = ranking.layer_cost(ranking.Terms(host_us=60.0, hop_us=10.0, t1_us=50.0, tk_us=20.0, pull_us=100.0))
        self.assertEqual((cost.with_card_us, cost.without_card_us, cost.ok), (110.0, 150.0, True))
        broken = ranking.layer_cost(ranking.Terms(host_us=float("nan"), hop_us=1.0, t1_us=1.0, tk_us=1.0, pull_us=1.0))
        self.assertFalse(broken.ok)  # a broken floor is never "ok"


if __name__ == "__main__":
    unittest.main()
