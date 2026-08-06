import unittest

from experiment.commands import rebuild_selection_aggregates as rebuild


class RebuildSelectionAggregatesTests(unittest.TestCase):
    def test_tracked_cells_reproduce_registered_aggregate_hashes(self):
        for directory, _, experiment_name, expected in rebuild.AGGREGATES:
            with self.subTest(directory=directory):
                payload = rebuild.build_payload(
                    rebuild.ROOT / directory, experiment_name
                )
                self.assertEqual(rebuild.payload_sha256(payload), expected)


if __name__ == "__main__":
    unittest.main()
