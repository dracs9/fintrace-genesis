import unittest
from datetime import date

from numpy.random import Generator, PCG64

from auditgym.world import build
from test_step0 import _preset


class WorldTests(unittest.TestCase):
    def test_presets_and_replay(self) -> None:
        for name in ("easy", "medium", "hard"):
            config = _preset(name)
            worlds = [build(config, Generator(PCG64(seed))) for seed in range(32)]
            self.assertEqual(worlds[0], build(config, Generator(PCG64(0))))
            self.assertNotEqual(worlds[0].market, worlds[1].market)
            for world in worlds:
                low, high = config["n_entities"]
                self.assertTrue(low <= len(world.entities) <= high)
                self.assertEqual(len({e.index for e in world.entities}), len(world.entities))
                self.assertTrue(all(e.currency in config["currencies"] for e in world.entities))
                self.assertEqual((world.base_currency, world.timezone, world.tail_days), ("KZT", "+05:00", 3))

    def test_probability_extremes_and_leap_month(self) -> None:
        config = dict(_preset("medium"), period={"start": "2024-02-01", "end": "2024-02-29"})
        for value in (0, 1):
            world = build(dict(config, foreign_share=value, tz_spread=value), Generator(PCG64(7)))
            self.assertEqual(world.market.end, date(2024, 3, 3))
            for entity in world.entities:
                self.assertEqual(entity.currency != "KZT", bool(value))
                self.assertEqual(entity.timezone != "+05:00", bool(value))
        local = build(dict(config, currencies=["KZT"]), Generator(PCG64(1)))
        self.assertEqual(local.market.quotes, ())

    def test_invalid_inputs(self) -> None:
        for change in ({"n_entities": [4, 40]}, {"n_entities": [40, 41]}, {"currencies": ["GBP"]},
                       {"tz_spread": 2}, {"period": {"start": "2024-02-01", "end": "2024-02-28"}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                build(dict(_preset("medium"), **change), Generator(PCG64(0)))
