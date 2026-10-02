"""CPU-only regression tests for subprocess portability and GPU allocation."""
import argparse
import ast
import contextlib
import importlib
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.env import (
    REPO_ROOT, hf_offline_env, resolve_num_processes, runtime_paths, validate_cuda_visibility,
)
from src.generation import run_vllm_generate
from src.train_inpo import run_train_inpo


def parse_worker_arguments(worker, argv):
    """Use the worker's actual argparse function without importing GPU packages."""
    module = ast.parse(worker.read_text(), filename=str(worker))
    parser_function = next(
        node for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "parse_args"
    )
    isolated_module = ast.Module(body=[parser_function], type_ignores=[])
    namespace = {"argparse": argparse}
    exec(compile(isolated_module, str(worker), "exec"), namespace)
    with patch.object(sys, "argv", [str(worker), *argv]):
        return namespace["parse_args"]()


class RuntimePathsTests(unittest.TestCase):
    def test_default_training_and_generation_use_current_python(self):
        self.assertEqual(
            runtime_paths({}, "/ordinary/venv/bin/python"),
            ("/ordinary/venv/bin/python", "/ordinary/venv/bin/accelerate",
             "/ordinary/venv/bin/python"),
        )

    def test_explicit_overrides_win(self):
        overrides = {
            "LLM_TRAIN_PYTHON": "/custom/train/bin/python",
            "LLM_TRAIN_ACCEL": "/custom/launcher",
            "LLM_VLLM_PYTHON": "/custom/generation/bin/python",
        }
        self.assertEqual(
            runtime_paths(overrides, "/current/bin/python"),
            (overrides["LLM_TRAIN_PYTHON"], overrides["LLM_TRAIN_ACCEL"],
             overrides["LLM_VLLM_PYTHON"]),
        )

    def test_accelerate_tracks_training_override_but_fallback_is_current_python(self):
        self.assertEqual(
            runtime_paths({"LLM_TRAIN_PYTHON": "/custom/bin/python"},
                          "/current/bin/python"),
            ("/custom/bin/python", "/custom/bin/accelerate", "/current/bin/python"),
        )

    def test_discover_sibling_of_training_or_current_conda_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate = Path(tmp) / "envs/llm_vllm/bin/python"
            candidate.parent.mkdir(parents=True)
            candidate.touch(mode=0o755)
            training = str(Path(tmp) / "envs/llm_train/bin/python")
            cases = [
                ({"LLM_TRAIN_PYTHON": training}, "/ordinary/bin/python"),
                ({"LLM_TRAIN_PYTHON": "/ordinary/bin/python"}, training),
            ]
            for env, current in cases:
                with self.subTest(env=env, current=current):
                    self.assertEqual(runtime_paths(env, current)[2], str(candidate))

    def test_discover_from_conda_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "conda-meta").mkdir()
            candidate = Path(tmp) / "envs/llm_vllm/bin/python"
            candidate.parent.mkdir(parents=True)
            candidate.touch(mode=0o755)
            self.assertEqual(
                runtime_paths({}, str(Path(tmp) / "bin/python"))[2], str(candidate)
            )

    def test_missing_or_nonexecutable_sibling_falls_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            current = str(Path(tmp) / "envs/llm_train/bin/python")
            self.assertEqual(runtime_paths({}, current)[2], current)
            candidate = Path(tmp) / "envs/llm_vllm/bin/python"
            candidate.parent.mkdir(parents=True)
            candidate.touch(mode=0o644)
            self.assertEqual(runtime_paths({}, current)[2], current)


class EnvironmentTests(unittest.TestCase):
    def test_offline_and_login_choices_are_inherited_without_reading_token_file(self):
        incoming = {
            "HF_HUB_OFFLINE": "0", "HF_DATASETS_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "0", "HF_TOKEN": "test-token",
            "WANDB_MODE": "offline", "CUDA_VISIBLE_DEVICES": "3,7",
        }
        with patch.dict(os.environ, incoming, clear=True), patch(
            "builtins.open", side_effect=AssertionError("must not read token cache")
        ):
            result = hf_offline_env()
        self.assertEqual(result, incoming)
        result["HF_TOKEN"] = "changed"
        self.assertEqual(incoming["HF_TOKEN"], "test-token")

    def test_empty_environment_is_not_replaced_with_process_environment(self):
        incoming = {}
        with patch.dict(os.environ, {"HF_TOKEN": "must-not-leak"}, clear=True):
            result = hf_offline_env(incoming)
        self.assertEqual(result, {"WANDB_MODE": "disabled"})
        self.assertEqual(incoming, {})

    def test_unset_offline_flags_remain_unset_and_no_token_is_loaded(self):
        with patch("builtins.open", side_effect=AssertionError("no token reads")):
            result = hf_offline_env({"PATH": "/bin"})
        self.assertEqual(result, {"PATH": "/bin", "WANDB_MODE": "disabled"})

    def test_explicit_environment_is_copied(self):
        incoming = {"HF_HUB_OFFLINE": "1"}
        self.assertEqual(hf_offline_env(incoming)["WANDB_MODE"], "disabled")
        self.assertEqual(incoming, {"HF_HUB_OFFLINE": "1"})

    def test_process_count_precedence_and_validation(self):
        self.assertEqual(resolve_num_processes(env={}), 8)
        self.assertEqual(resolve_num_processes(env={"LLM_NUM_PROCESSES": "2"}), 2)
        self.assertEqual(resolve_num_processes(1, {"LLM_NUM_PROCESSES": "bad"}), 1)
        for invalid in ("0", "-2", "", "1.5", "many"):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                ValueError, "LLM_NUM_PROCESSES must be a positive integer"
            ):
                resolve_num_processes(env={"LLM_NUM_PROCESSES": invalid})
        for invalid in (0, -1, 1.5, True):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                ValueError, "num_processes must be a positive integer"
            ):
                resolve_num_processes(invalid, {})


class CudaVisibilityTests(unittest.TestCase):
    def test_explicit_masks_are_preserved(self):
        for mask in ("3,7", "GPU-abcd,GPU-efgh", "MIG-abcd,MIG-efgh", " 3, 7 "):
            incoming = {"CUDA_VISIBLE_DEVICES": mask}
            with self.subTest(mask=mask):
                validate_cuda_visibility(2, incoming)
                self.assertEqual(incoming, {"CUDA_VISIBLE_DEVICES": mask})

    def test_mask_with_insufficient_devices_is_rejected_without_modification(self):
        for mask in ("3", "", "-1", "3,-1,7", "3,,7", "3,3"):
            incoming = {"CUDA_VISIBLE_DEVICES": mask}
            with self.subTest(mask=mask), self.assertRaisesRegex(
                ValueError, "tp_size=2.*CUDA_VISIBLE_DEVICES"
            ):
                validate_cuda_visibility(2, incoming)
            self.assertEqual(incoming, {"CUDA_VISIBLE_DEVICES": mask})

    def test_missing_mask_is_not_created(self):
        incoming = {}
        validate_cuda_visibility(8, incoming)
        self.assertEqual(incoming, {})

    def test_invalid_tensor_parallel_size_is_rejected(self):
        for invalid in (0, -1, 1.5, True):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                ValueError, "tp_size must be a positive integer"
            ):
                validate_cuda_visibility(invalid, {})

    def test_generation_forwards_inherited_mask_or_leaves_it_unset(self):
        for incoming in ({"CUDA_VISIBLE_DEVICES": "3,7,8"}, {}):
            with self.subTest(env=incoming), tempfile.TemporaryDirectory() as tmp, \
                    patch.dict(os.environ, incoming, clear=True), \
                    patch("src.generation.subprocess.run") as run, \
                    contextlib.redirect_stdout(io.StringIO()):
                output = run_vllm_generate("model", ["prompt"], tmp, tp_size=2)
                self.assertEqual(output, str(Path(tmp) / "completions.json"))
                child_env = run.call_args.kwargs["env"]
                self.assertEqual(child_env.get("CUDA_VISIBLE_DEVICES"),
                                 incoming.get("CUDA_VISIBLE_DEVICES"))
                self.assertEqual(os.environ.copy(), incoming)
                command = run.call_args.args[0]
                self.assertEqual(command[command.index("--tensor_parallel_size") + 1], "2")
                self.assertTrue(run.call_args.kwargs["check"])

    def test_generation_fails_before_writing_or_launching_for_insufficient_mask(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "7"}, clear=True), \
                patch("src.generation.subprocess.run") as run:
            with self.assertRaisesRegex(ValueError, "tp_size=2"):
                run_vllm_generate("model", ["prompt"], tmp, tp_size=2)
            run.assert_not_called()
            self.assertEqual(list(Path(tmp).iterdir()), [])
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "7")


class AccelerateWrapperTests(unittest.TestCase):
    def test_retained_wrappers_launch_existing_workers_with_valid_arguments(self):
        with tempfile.TemporaryDirectory() as tmp:
            training = dict(
                policy_path="policy", preference_data_dir="data", output_dir=tmp,
                per_device_train_batch_size=3, gradient_accumulation_steps=4,
                learning_rate=3e-6, num_train_epochs=2, max_length=512,
                wandb_run_name="test-run", seed=9,
            )
            scoring = dict(judge_model="judge", judge_batch_size=3, judge_max_length=256)
            cases = [
                ("train_inpo", "run_train_inpo", dict(training, eta=0.2, tau=0.1)),
                ("train_enpo_step_a", "run_train_enpo_step_a",
                 dict(training, eta=0.2, tau=0.1, alpha=0.1)),
                ("train_enpo_step_b", "run_train_enpo_step_b", dict(training, beta=0.1)),
                ("train_xpo", "run_train_xpo", dict(policy_path="policy",
                 preference_data_dirs=["data-a", "data-b"], output_dir=tmp,
                 beta=0.1, alpha=0.2, seed=9, wandb_run_name="xpo-test")),
                ("precompute", "run_precompute_logp", dict(model_path="model",
                 dataset_dir="data", output_dataset_dir=tmp,
                 logp_chosen_field="chosen", logp_rejected_field="rejected",
                 max_length=512, per_device_batch_size=3)),
                ("score", "run_score", dict(scoring, prompts=["prompt"], instructions=["instruction"],
                 message_lists=[[]], completions_file="completions", iter_dir=tmp,
                 margin_keep_pct=0.6)),
                ("score_xpo", "run_score_xpo", dict(scoring, prompts=["prompt"],
                 instructions=["instruction"], message_lists=[[]],
                 completions_pi_file="pi", completions_ref_file="ref",
                 iter_dir=tmp, margin_keep_pct=0.6)),
                ("score_enpo", "run_score_enpo", dict(scoring, triples_dataset_dir="data",
                 output_dataset_dir=tmp)),
            ]
            incoming = {"CUDA_VISIBLE_DEVICES": "3,7", "LLM_NUM_PROCESSES": "2"}
            for module_name, function_name, kwargs in cases:
                function = getattr(importlib.import_module(f"src.{module_name}"), function_name)
                with self.subTest(wrapper=function_name), \
                        patch.dict(os.environ, incoming, clear=True), \
                        patch(f"src.{module_name}.subprocess.run") as run, \
                        contextlib.redirect_stdout(io.StringIO()):
                    function(**kwargs)
                    command = run.call_args.args[0]
                    self.assertEqual(command[command.index("--num_processes") + 1], "2")
                    self.assertEqual(run.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "3,7")
                    self.assertEqual(os.environ.copy(), incoming)
                    self.assertTrue(run.call_args.kwargs["check"])
                    worker_name = "precompute_logp" if module_name == "precompute" else module_name
                    worker = Path(REPO_ROOT) / "workers" / f"{worker_name}.py"
                    self.assertTrue(worker.is_file(), str(worker))
                    worker_index = command.index(str(worker))
                    parsed = parse_worker_arguments(worker, command[worker_index + 1:])
                    for key, value in kwargs.items():
                        if key in {"prompts", "instructions", "message_lists"}:
                            continue
                        argument = "output_dir" if key == "iter_dir" else key
                        self.assertEqual(getattr(parsed, argument), value, key)
                    if "iter_dir" in kwargs:
                        self.assertEqual(parsed.prompts_file, str(Path(tmp) / "prompts.json"))
                        self.assertTrue(Path(parsed.prompts_file).is_file())

    def test_explicit_process_count_overrides_environment(self):
        with patch.dict(os.environ, {"LLM_NUM_PROCESSES": "bad"}, clear=True), \
                patch("src.train_inpo.subprocess.run") as run, \
                contextlib.redirect_stdout(io.StringIO()):
            run_train_inpo(policy_path="policy", preference_data_dir="data",
                           output_dir="output", eta=0.2, tau=0.1, num_processes=1)
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("--num_processes") + 1], "1")

    def test_invalid_process_count_prevents_launch(self):
        with patch.dict(os.environ, {"LLM_NUM_PROCESSES": "0"}, clear=True), \
                patch("src.train_inpo.subprocess.run") as run:
            with self.assertRaisesRegex(ValueError, "LLM_NUM_PROCESSES"):
                run_train_inpo(policy_path="policy", preference_data_dir="data",
                               output_dir="output", eta=0.2, tau=0.1)
        run.assert_not_called()


class ReleaseClosureTests(unittest.TestCase):
    def test_release_has_only_requested_runner_entrypoints(self):
        self.assertEqual(
            {path.stem for path in (Path(REPO_ROOT) / "runners").glob("*.py")},
            {"__init__", "enpo", "inpo", "xpo"},
        )

    def test_retained_local_imports_resolve(self):
        root = Path(REPO_ROOT)
        local_packages = {"runners", "src", "workers"}
        for package in local_packages:
            for path in (root / package).glob("*.py"):
                tree = ast.parse(path.read_text(), filename=str(path))
                for node in ast.walk(tree):
                    if not isinstance(node, ast.ImportFrom) or not node.module:
                        continue
                    if node.module.split(".")[0] not in local_packages:
                        continue
                    with self.subTest(source=path.name, module=node.module):
                        target = root.joinpath(*node.module.split(".")).with_suffix(".py")
                        self.assertTrue(target.is_file(), str(target))
                        target_tree = ast.parse(target.read_text(), filename=str(target))
                        names = {
                            item.name for item in target_tree.body
                            if isinstance(item, (ast.FunctionDef, ast.ClassDef))
                        }
                        names.update(
                            item.id for item in ast.walk(target_tree)
                            if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store)
                        )
                        for imported in node.names:
                            self.assertIn(imported.name, names)


if __name__ == "__main__":
    unittest.main()
