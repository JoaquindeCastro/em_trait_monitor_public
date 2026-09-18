import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from experiments.causal_steering.run_semantic_steering_v2 import (
    build_plan,
    expected_generation_ids,
    load_prompt_suite,
    load_protocol,
    select_pilot_prompts,
    validate_pilot_scope,
    validate_run_scope,
    main,
)
from src.evaluation.betley_judge import (
    COHERENCE_THRESHOLD,
    get_betley_coherence_template,
    load_betley_prompts,
)
from src.evaluation.semantic_steering_v2 import (
    aggregate_semantic_scores,
    bootstrap_semantic_interval,
    build_coherence_prompt,
    parse_coherence_judgment,
    parse_semantic_judgment,
    prepare_semantic_pair,
    score_semantic_judgment,
    select_best_layer,
    select_common_coherent_dose,
)


class BetleyCoherenceTests(unittest.TestCase):
    def test_all_betley_items_share_the_canonical_template(self):
        prompts = [{"judge_prompt_coherent": "Task: {question}\nResponse: {answer}"}] * 2
        template = get_betley_coherence_template(prompts)
        self.assertEqual({p["judge_prompt_coherent"] for p in prompts}, {template})

    def test_v2_builder_uses_the_betley_template_verbatim(self):
        question = "A unique question token"
        answer = "A unique answer token"
        template = "Task: {question} Response: {answer}"
        with patch("src.evaluation.betley_judge.load_betley_prompts",
                   return_value=[{"judge_prompt_coherent": template}]):
            rendered = build_coherence_prompt(question, answer)
        expected = template.replace("{question}", question).replace("{answer}", answer)
        self.assertEqual(rendered, expected)

    def test_threshold_and_parse_failures_fail_closed(self):
        self.assertEqual(COHERENCE_THRESHOLD, 50)
        self.assertFalse(parse_coherence_judgment("49")["coherent"])
        self.assertTrue(parse_coherence_judgment("50")["coherent"])
        malformed = parse_coherence_judgment("not a score")
        self.assertFalse(malformed["parse_ok"])
        self.assertFalse(malformed["coherent"])


class SemanticPairTests(unittest.TestCase):
    def test_blinding_is_deterministic_and_preserves_expected_winner(self):
        kwargs = {
            "prompt": "Tell me what happened.",
            "positive_label": "honest",
            "negative_label": "dishonest",
            "expected_response": "expected response",
            "comparison_response": "comparison response",
            "blind_key": "llama|14|0.2|honesty|prompt-1|positive",
            "seed": 20260912,
        }
        first = prepare_semantic_pair(**kwargs)
        second = prepare_semantic_pair(**kwargs)
        self.assertEqual(first, second)
        expected_slot = (
            first["response_a"]
            if first["expected_winner"] == "A"
            else first["response_b"]
        )
        self.assertEqual(expected_slot, "expected response")
        self.assertNotIn(
            "expected response", first["judge_prompt"].split("Response A:")[0]
        )

    def test_pairwise_scoring(self):
        self.assertEqual(score_semantic_judgment("A", "A"), 1.0)
        self.assertEqual(score_semantic_judgment("tie", "A"), 0.5)
        self.assertEqual(score_semantic_judgment("B", "A"), 0.0)
        self.assertEqual(score_semantic_judgment(None, "A"), 0.0)

    def test_semantic_parser_is_strict(self):
        parsed = parse_semantic_judgment('{"winner":"tie","reason":"similar"}')
        self.assertTrue(parsed["parse_ok"])
        self.assertEqual(parsed["winner"], "tie")
        self.assertFalse(parse_semantic_judgment("A")["parse_ok"])
        self.assertFalse(parse_semantic_judgment('{"winner":"C"}')["parse_ok"])
        for malformed in ("null", "[]", "42", '"A"'):
            self.assertFalse(parse_semantic_judgment(malformed)["parse_ok"])


class AggregationTests(unittest.TestCase):
    def test_trait_weighting_is_equal_not_record_weighted(self):
        result = aggregate_semantic_scores(
            [
                {"trait": "honesty", "score": 1.0},
                {"trait": "honesty", "score": 1.0},
                {"trait": "helpfulness", "score": 0.0},
            ],
            traits=("honesty", "helpfulness"),
        )
        self.assertEqual(result["score"], 0.5)

    def test_common_dose_must_pass_every_candidate_layer(self):
        coherent = [{"coherent": True}] * 9 + [{"coherent": False}]
        failing = [{"coherent": True}] * 8 + [{"coherent": False}] * 2
        selected, rates = select_common_coherent_dose(
            {
                0.1: {14: coherent, 16: coherent},
                0.2: {14: coherent, 16: coherent},
                0.3: {14: coherent, 16: failing},
            },
            candidate_layers=(14, 16),
        )
        self.assertEqual(selected, 0.2)
        self.assertEqual(rates[0.3][16], 0.8)

    def test_bootstrap_is_deterministic_and_preserves_sign_pairs(self):
        records = []
        for trait in ("honesty", "helpfulness"):
            records.extend(
                [
                    {"trait": trait, "positive_score": 1.0, "negative_score": 0.5},
                    {"trait": trait, "positive_score": 0.5, "negative_score": 0.0},
                ]
            )
        kwargs = {
            "samples": 100,
            "seed": 7,
            "traits": ("honesty", "helpfulness"),
        }
        self.assertEqual(
            bootstrap_semantic_interval(records, **kwargs),
            bootstrap_semantic_interval(records, **kwargs),
        )

    def test_exact_layer_tie_prefers_band_center_then_lower_layer(self):
        selected = select_best_layer(
            {14: 0.7, 16: 0.8, 18: 0.8, 20: 0.6},
            candidate_layers=(14, 16, 18, 20),
        )
        self.assertEqual(selected, 16)


class PilotPlanTests(unittest.TestCase):
    def test_small_pilot_counts(self):
        protocol = load_protocol()
        validate_pilot_scope(protocol, "qwen25-7b", [14, 16])
        suite = load_prompt_suite(protocol)
        prompts = select_pilot_prompts(suite, prompts_per_trait=2)
        plan = build_plan(
            protocol=protocol,
            prompts_per_trait=2,
            layers=[14, 16],
        )
        self.assertEqual(len(prompts), 14)
        self.assertEqual(plan["doses"], [0.05, 0.10, 0.25, 0.50])
        self.assertEqual(plan["steered_generations"], 224)
        self.assertEqual(plan["coherence_calls"], 224)
        self.assertEqual(plan["semantic_calls_after_dose_selection"], 56)

    def test_pilot_accepts_custom_layers_and_rejects_invalid_layers(self):
        validate_pilot_scope(load_protocol(), "mistral-7b", [14, 18])
        with self.assertRaises(ValueError):
            validate_pilot_scope(load_protocol(), "qwen25-7b", [14, 28])


class FullPlanTests(unittest.TestCase):
    def test_resume_refuses_changed_configuration_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            arguments = ["validate", "plan", "--model", "qwen25-7b", "--output-dir", directory]
            with patch("sys.argv", arguments):
                main()
            plan_path = Path(directory) / "run_plan.json"
            plan = json.loads(plan_path.read_text())
            plan["config_sha256"] = "changed"
            plan_path.write_text(json.dumps(plan))
            with patch("sys.argv", arguments), self.assertRaises(ValueError):
                main()

    def test_full_scope_uses_all_prompts_and_candidate_layers(self):
        protocol = load_protocol()
        suite = load_prompt_suite(protocol)
        items = select_pilot_prompts(suite, prompts_per_trait=20)
        cases = {
            "llama3-8b": ([14, 15, 16, 17, 18, 19, 20], 7980),
            "mistral-7b": ([14, 16, 18, 20], 4620),
            "qwen25-7b": ([14, 16, 18, 20], 4620),
            "gemma2-9b": ([21, 24, 27, 30, 35, 38, 40], 7980),
        }
        for model, (layers, expected_count) in cases.items():
            with self.subTest(model=model):
                validate_run_scope(protocol, "full", model, layers, 20)
                plan = build_plan(
                    protocol=protocol,
                    prompts_per_trait=20,
                    layers=layers,
                    run_kind="full",
                    model_key=model,
                )
                ids = expected_generation_ids(
                    model_key=model,
                    layers=layers,
                    doses=protocol["strength"]["grid"],
                    items=items,
                )
                self.assertEqual(len(ids), expected_count)
                self.assertEqual(
                    plan["baseline_generations"] + plan["steered_generations"],
                    expected_count,
                )

    def test_full_scope_accepts_parameter_overrides(self):
        protocol = load_protocol()
        validate_run_scope(protocol, "full", "qwen25-7b", [14, 16], 2)
        validate_run_scope(protocol, "full", "qwen25-14b", [20, 22], 5)
        with self.assertRaises(ValueError):
            validate_run_scope(protocol, "full", "qwen25-7b", [14, 14], 20)
        with self.assertRaises(ValueError):
            validate_run_scope(
                protocol, "full", "qwen25-7b", [14, 16, 18, 20], 0
            )


if __name__ == "__main__":
    unittest.main()
