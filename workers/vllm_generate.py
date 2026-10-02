#!/usr/bin/env python3
"""
Standalone vLLM generation worker. Reads JSON config, writes JSON completions.

Designed to be invoked as a subprocess from runners/ (running under a
different conda env). All inputs/outputs are file-based to keep the env
boundary clean.

Input JSON (--input):
  {
    "model":           "<HF id or local path>",
    "prompts":         ["<chat-templated prompt>", ...],
    "n":               4,        # completions per prompt
    "temperature":     0.7,
    "top_p":           0.9,
    "max_new_tokens":  512,
    "seed":            null,     # optional
    "max_model_len":   1536,     # optional, default 1024 + max_new_tokens
    "tokenizer":       null      # optional; load tokenizer from this path/id
                                  # instead of `model` (see note below)
  }

Locally-saved iteration checkpoints carry a tokenizer_config.json written
by whichever `transformers` version the training env used; if that env's
transformers is newer than this env's, the saved tokenizer_class may not
be loadable here (e.g. transformers 5.x's "TokenizersBackend" backend
doesn't exist in transformers 4.x). The tokenizer itself never changes
across iterations for these algorithms — only the weights do — so callers
should pass `tokenizer` = the original base model id/path to sidestep the
cross-env serialization mismatch entirely.

Output JSON (--output):
  [["<completion 1a>", "<completion 1b>", ...], ["<completion 2a>", ...], ...]
"""
import argparse
import json
import sys
import time

from vllm import LLM, SamplingParams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--tensor_parallel_size",   type=int,   default=8)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    args = ap.parse_args()

    with open(args.input) as f:
        cfg = json.load(f)

    max_model_len = cfg.get("max_model_len", 1024 + cfg["max_new_tokens"])

    print(f"[vllm_generate] loading {cfg['model']} (TP={args.tensor_parallel_size})", file=sys.stderr)
    t0 = time.time()
    llm = LLM(
        model=cfg["model"],
        tokenizer=cfg.get("tokenizer") or cfg["model"],
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="bfloat16",
        max_model_len=max_model_len,
        enforce_eager=False,
        disable_custom_all_reduce=True,
    )
    print(f"[vllm_generate] model loaded in {time.time()-t0:.1f}s", file=sys.stderr)

    sp = SamplingParams(
        n=cfg["n"],
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        max_tokens=cfg["max_new_tokens"],
        seed=cfg.get("seed"),
    )

    print(f"[vllm_generate] generating {len(cfg['prompts'])} prompts × n={cfg['n']}", file=sys.stderr)
    t0 = time.time()
    outputs = llm.generate(cfg["prompts"], sp)
    dt = time.time() - t0
    n_total = len(cfg["prompts"]) * cfg["n"]
    print(f"[vllm_generate] generated {n_total} completions in {dt:.1f}s "
          f"({n_total/dt:.1f} comp/s)", file=sys.stderr)

    completions = [[o.text for o in out.outputs] for out in outputs]
    with open(args.output, "w") as f:
        json.dump(completions, f)
    print(f"[vllm_generate] wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
