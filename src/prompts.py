"""Prompt loading and pre-truncation for online DPO."""
from datasets import load_dataset


def load_prompts(dataset_name, num_prompts, tokenizer, start_idx=0,
                 split="train_prefs"):
    """Load N preference prompts and return three parallel lists.

    Returns:
        prompts:        chat-templated strings with `add_generation_prompt=True`,
                        ready for vLLM.
        instructions:   concatenated user-text only (legacy/debug).
        message_lists:  raw [{"role", "content"}, ...] lists; the judge applies
                        its own chat template to these when scoring (chosen +
                        completion). Keeping the structured form avoids
                        round-tripping through the policy's chat template.
    """
    data = load_dataset(dataset_name)[split]
    end_idx = start_idx + num_prompts
    if end_idx > len(data):
        raise ValueError(
            f"Slice [{start_idx}, {end_idx}) exceeds {split} size {len(data)}")
    data = data.select(range(start_idx, end_idx))

    prompts, instructions, message_lists = [], [], []
    for ex in data:
        msgs = ex["chosen"][:-1]  # everything up to the assistant turn we'll generate
        prompts.append(tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True))
        instructions.append(
            "\n".join(m["content"] for m in msgs if m["role"] == "user"))
        message_lists.append(msgs)
    return prompts, instructions, message_lists


def truncate_prompts(prompts, tokenizer, max_input_tokens):
    """Left-truncate prompts that exceed max_input_tokens (keep last N tokens).
    Mirrors the implicit behavior of the original HF code path that used
    tokenizer(..., truncation=True, max_length=1024).

    Returns (truncated_prompts, n_truncated).
    """
    out, n_truncated = [], 0
    for p in prompts:
        ids = tokenizer.encode(p, add_special_tokens=False)
        if len(ids) > max_input_tokens:
            ids = ids[-max_input_tokens:]
            p = tokenizer.decode(ids, skip_special_tokens=False)
            n_truncated += 1
        out.append(p)
    return out, n_truncated
