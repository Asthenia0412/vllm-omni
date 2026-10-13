"""CPU validation of private-retention admission; no device proof is inferred."""
import ast
from pathlib import Path
import tempfile
import unittest
import json

import check_qk_private_retention_matrix as checker
import aggregate_qk_private_retention_matrix as aggregate


def stress_record():
    before = dict(submitted=8, callbacks_done=8, plan_reuses=5,
                  producer_reclaimed_calls=4, producer_reclaim_tid=10, producer_tid=10,
                  pending=0, poisoned=0, pool_exhaustions=0, retirement_overflows=0)
    after = dict(before, submitted=20, callbacks_done=20, plan_reuses=7,
                 producer_reclaimed_calls=6)
    return dict(before=before, after=after, calls=12, retained_send_receive_pairs=12,
                churn_allocations=72, per_call_device_drain=False, inflight_coverage_claim=False,
                fresh_native_source_owners=True, python_source_locals_released=True,
                python_lease_locals_released=True, all_retained_outputs_bitwise=True,
                prior_arena_canaries_bitwise=True, original_sources_unchanged=True, original_hccl=True,
                pressure_allocated_on_pressure_stream=True, pressure_stream_distinct=True,
                pressure_submitted_on_separate_stream=True,
                fresh_source_contents_after_release_observed=False,
                native_private_reuse_addresses_observed=False, device_overlap_observed=False)


def batch(durations, scope, rank):
    wall, events = [], []
    for index, value in enumerate(durations):
        start = 1_000_000_000 + index * 1_000_000_000 + rank * 1000
        wall.append(dict(index=index, start_ns=start, return_ns=start + value // 2,
                         end_ns=start + value, wall_ms=value / 1e6))
        start += 100_000_000_000
        events.append(dict(index=index, start_ns=start, return_ns=start + value // 2,
                           end_ns=start + value, event_ms=100.0))
    return dict(warmup=20, wall_samples=wall, event_samples=events,
                generation_start=1, generation_end=81 if scope == aggregate.SCOPES[0] else 1,
                events_are_separate_batch=True,
                phase_boundaries=[dict(name="warm20", enter_ns=0, ready_ns=1),
                                  dict(name="wall30", enter_ns=2, ready_ns=3),
                                  dict(name="event30_separate_batch", enter_ns=90_000_000_000,
                                       ready_ns=90_000_000_001)])


def reports(ratios):
    result = [dict(boot_id="one-boot", cases=[]) for _ in range(8)]
    for ratio in ratios:
        for rank in range(8):
            records = {scope: dict(baseline=batch([1_000_000] * 30, scope, rank),
                                   pool=batch([int(1_000_000 * ratio)] * 30, scope, rank))
                       for scope in aggregate.SCOPES}
            result[rank]["cases"].append(dict(records=records))
    return result


class PrivateStressAdmission(unittest.TestCase):
    def test_success_requires_reuse_and_release_witnesses(self):
        checker.validate_private_churn(stress_record())
        for name in ("fresh_native_source_owners", "python_source_locals_released",
                     "python_lease_locals_released", "all_retained_outputs_bitwise",
                     "prior_arena_canaries_bitwise", "original_sources_unchanged", "original_hccl",
                     "pressure_allocated_on_pressure_stream", "pressure_stream_distinct",
                     "pressure_submitted_on_separate_stream"):
            record = stress_record()
            record[name] = False
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                checker.validate_private_churn(record)

    def test_unobserved_overlap_address_reuse_and_released_source_bytes_cannot_be_claimed(self):
        for name in ("device_overlap_observed", "native_private_reuse_addresses_observed",
                     "fresh_source_contents_after_release_observed"):
            record = stress_record()
            record[name] = True
            with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, "exceeds observed scope"):
                checker.validate_private_churn(record)

    def test_no_reclaim_wrong_thread_incomplete_burst_and_poison_rejected(self):
        for field, value in (("producer_reclaimed_calls", 4), ("plan_reuses", 5),
                             ("producer_reclaim_tid", 11), ("callbacks_done", 19),
                             ("submitted", 19), ("pending", 1), ("poisoned", 1),
                             ("pool_exhaustions", 1), ("retirement_overflows", 1)):
            record = stress_record()
            record["after"][field] = value
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                checker.validate_private_churn(record)

    def test_per_call_drain_and_wrong_churn_extent_rejected(self):
        for field, value in (("calls", 11), ("retained_send_receive_pairs", 11),
                             ("churn_allocations", 71), ("per_call_device_drain", True),
                             ("inflight_coverage_claim", True)):
            record = stress_record()
            record[field] = value
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                checker.validate_private_churn(record)

    def test_old_v2_correctness_cannot_admit_new_timing(self):
        report = dict(schema="lingbot_qk_repeatable_matrix_replay_v1", passed=True,
                      mode="correctness", rank=0)
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "rank0.json").write_text(json.dumps(report))
            with self.assertRaisesRegex(RuntimeError, "successful same-rank"):
                checker.verify_correctness(directory, {}, {}, {})

    def test_schema_and_checks_are_separate_from_frozen_replay(self):
        self.assertEqual(checker.SCHEMA, aggregate.SCHEMA)
        self.assertEqual(tuple(checker.CHECKS), aggregate.CHECKS)
        self.assertIn("released_sources_and_private_churn", checker.CHECKS)
        self.assertNotEqual(checker.SCHEMA, "lingbot_qk_repeatable_matrix_replay_v1")

    def test_source_does_not_insert_per_call_drains_or_return_source_owners(self):
        tree = ast.parse(Path(checker.__file__).read_text())
        replay = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Replay")
        fresh = next(node for node in replay.body if isinstance(node, ast.FunctionDef)
                     and node.name == "_fresh_sources_call")
        returns = [node for node in ast.walk(fresh) if isinstance(node, ast.Return)]
        self.assertEqual(len(returns), 1)
        self.assertEqual(ast.unparse(returns[0].value), "(lease.send, receive)")
        stress = next(node for node in replay.body if isinstance(node, ast.FunctionDef)
                      and node.name == "private_churn")
        burst = next(node for node in ast.walk(stress) if isinstance(node, ast.For)
                     and ast.unparse(node.iter) == "range(12)")
        self.assertFalse(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                             and node.func.attr in ("synchronize", "barrier", "cpu", "item")
                             for node in ast.walk(burst)))
        self.assertFalse(any(isinstance(node, ast.Assert) for node in ast.walk(tree)))


class UnchangedNumericalGate(unittest.TestCase):
    def test_exact_boundaries_and_outlier_retention(self):
        self.assertTrue(aggregate.summarize_round(reports([0.9] * 24), 1)["round_gate_passed"])
        row = aggregate.summarize_round(reports([0.8] * 23 + [1.050001]), 1)
        for scope in row["scopes"].values():
            self.assertTrue(scope["aggregate_10_percent_gate_passed"])
            self.assertFalse(scope["all24_case_5_percent_gate_passed"])
        self.assertFalse(row["round_gate_passed"])

    def test_event_time_is_not_added_to_wall(self):
        row = aggregate.summarize_round(reports([0.9] * 24), 1)
        case = row["scopes"][aggregate.SCOPES[0]]["cases"][0]
        self.assertEqual(case["arms"]["pool"]["case_metric_ms"], 0.9)
        self.assertEqual(case["arms"]["pool"]["event_batch"]["per_rank_event_mean_ms"], [100.0] * 8)

    def test_wrong_clock_and_bad_wall_durations_are_rejected(self):
        rows = reports([0.8] * 24)
        rows[1]["boot_id"] = "different"
        with self.assertRaisesRegex(ValueError, "same host boot"):
            aggregate.summarize_round(rows, 1)
        value = batch([1_000_000] * 30, aggregate.SCOPES[0], 0)
        value["wall_samples"][0]["wall_ms"] = 9
        with self.assertRaisesRegex(ValueError, "recorded complete-call timestamps"):
            aggregate.validate_batch(value, aggregate.SCOPES[0])


if __name__ == "__main__":
    unittest.main()
