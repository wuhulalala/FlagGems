"""DeepSeek-V2-Lite fixed greedy decoding, adapted from upstream swjbr."""

import argparse
import inspect
import json
import os
import statistics
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

STATIC_TRIDENT_WRAPPERS = ["index_select", "mean_dim", "sum_dim"]


def prompts(task, limit, cache, offset=0):
    if task == "smoke":
        rows = [
            "Explain what a prime number is.",
            "Briefly explain photosynthesis.",
            "What causes ocean tides?",
            "Describe binary search.",
            "Why is the sky blue?",
            "What is a neural network?",
            "How does a compass work?",
            "Explain the water cycle.",
            "What is natural selection?",
            "Describe how batteries store energy.",
        ]
        return rows[offset : offset + limit]
    datasets = {
        "mmlu": ("cais/mmlu", "all"),
        "humaneval": ("openai/openai_humaneval", None),
        "gsm8k": ("openai/gsm8k", "main"),
    }
    name, subset = datasets[task]
    ds = load_dataset(name, subset, split="test", cache_dir=cache)
    rows = ds.select(range(offset, min(offset + limit, len(ds))))
    if task == "mmlu":
        return [
            f"Question: {r['question']}\nChoices: {r['choices']}\nAnswer:" for r in rows
        ]
    if task == "gsm8k":
        return [
            f"Solve this problem step by step.\n{r['question']}\nAnswer:" for r in rows
        ]
    return [r["prompt"] for r in rows]


@torch.inference_mode()
def decode(model, ids, tokens):
    result, past = ids, None
    for _ in range(tokens):
        mask = torch.ones_like(result)
        out = model(
            input_ids=ids,
            attention_mask=mask,
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        past = out.past_key_values
        ids = out.logits[:, -1].argmax(-1, keepdim=True)
        result = torch.cat((result, ids), dim=1)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="/mnt/op/models/DeepSeek-V2-Lite")
    parser.add_argument(
        "--backend", choices=("eager", "trident", "flaggems"), required=True
    )
    parser.add_argument(
        "--include",
        nargs="+",
        default=None,
        help="Explicit FlagGems implementation names; required for replacement modes",
    )
    parser.add_argument(
        "--task",
        choices=("smoke", "mmlu", "humaneval", "gsm8k", "all"),
        default="smoke",
    )
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--warmup-limit", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "flash_attention_2", "gems_flash"),
        default="eager",
    )
    parser.add_argument("--cache-dir", default="/workspace/deepseek-datasets")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--max-compiles",
        type=int,
        default=128,
        help="Stop runaway recompilation during model validation",
    )
    parser.add_argument(
        "--trident-scope",
        choices=("all", "dynamic"),
        default="all",
        help="Enable all validated wrappers, or only validated dynamic wrappers",
    )
    parser.add_argument("--skip-trident-wrappers", nargs="*", default=[])
    args = parser.parse_args()
    skipped = list(args.skip_trident_wrappers)
    static = STATIC_TRIDENT_WRAPPERS
    if args.trident_scope == "dynamic":
        skipped += static
    if (
        args.limit < 1
        or args.warmup_limit < 1
        or args.max_new_tokens < 1
        or args.warmup < 0
    ):
        parser.error(
            "limit/warmup-limit/tokens must be positive and warmup nonnegative"
        )
    tasks = ("mmlu", "humaneval", "gsm8k") if args.task == "all" else (args.task,)
    warmup_data = {
        task: prompts(task, args.warmup_limit, args.cache_dir, offset=0)
        for task in tasks
    }
    data = {
        task: prompts(task, args.limit, args.cache_dir, offset=args.warmup_limit)
        for task in tasks
    }
    if args.download_only:
        print(
            {
                task: {"warmup": len(warmup_data[task]), "measure": len(data[task])}
                for task in tasks
            }
        )
        return
    ctx = nullcontext()
    compile_stats = {"count": 0, "phase": "setup"}
    result = {
        "backend": args.backend,
        "model": args.model_path,
        "attn_implementation": args.attn_implementation,
        "include": args.include,
        "warmup_samples": args.warmup_limit,
        "measure_samples": args.limit,
        "trident_scope": args.trident_scope,
        "trident_static_wrappers": static if args.trident_scope == "all" else [],
        "trident_skipped_wrappers": skipped,
        "tasks": {},
    }
    if args.backend != "eager":
        if not args.include:
            parser.error("pass --include with validated implementation names")
        from deepseek_registry import configure_trident, prepare_registry

        configure_trident(
            enabled=args.backend == "trident",
            skip=skipped,
            static=static if args.trident_scope == "all" else (),
        )
        os.environ["FLAGGEMS_POINTWISE_WRAPPER"] = "@trident.jit"

        if args.backend == "trident":
            from trident.backend import TridentGraphModule

            original_compile = TridentGraphModule.compile

            def metadata(value):
                if isinstance(value, torch.Tensor):
                    return {
                        "shape": list(value.shape),
                        "stride": list(value.stride()),
                        "offset": value.storage_offset(),
                        "dtype": str(value.dtype),
                    }
                return repr(value)

            def counted_compile(self, *values, **options):
                compile_stats["count"] += 1
                if compile_stats["count"] > args.max_compiles:
                    raise RuntimeError(
                        "Compilation limit reached; inspect repeated specializations before benchmarking"
                    )
                started = time.perf_counter()
                print(
                    json.dumps(
                        dict(
                            event="compile_begin",
                            name=self.fn.__name__,
                            version=len(self._sub_modules),
                            inputs=[metadata(v) for v in values],
                            options={k: metadata(v) for k, v in options.items()},
                            **compile_stats,
                        )
                    ),
                    flush=True,
                )
                try:
                    return original_compile(self, *values, **options)
                finally:
                    print(
                        json.dumps(
                            dict(
                                event="compile_end",
                                name=self.fn.__name__,
                                seconds=time.perf_counter() - started,
                            )
                        ),
                        flush=True,
                    )

            TridentGraphModule.compile = counted_compile

        flag_gems = prepare_registry()

        missing = set(args.include) - flag_gems.FULL_CONFIG_BY_FUNC.keys()
        if missing:
            parser.error(f"Unknown FlagGems names: {sorted(missing)}")
        from trident.backend import TridentGraphModule

        result["implementation_sources"] = {
            name: sorted(
                set(
                    inspect.getsourcefile(
                        e[1].fn if isinstance(e[1], TridentGraphModule) else e[1]
                    )
                    for e in flag_gems.FULL_CONFIG_BY_FUNC[name]
                )
            )
            for name in args.include
        }
        print(
            json.dumps(
                dict(
                    event="implementation_sources",
                    sources=result["implementation_sources"],
                )
            ),
            flush=True,
        )
        result["flaggems_file"] = flag_gems.__file__
        ctx = flag_gems.use_gems(include=args.include)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True
    )
    load_attention = (
        "eager"
        if args.attn_implementation == "gems_flash"
        else args.attn_implementation
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="cuda",
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation=load_attention,
    ).eval()
    flash_calls = 0
    if args.attn_implementation == "gems_flash":
        modeling = __import__(model.__class__.__module__, fromlist=["*"])

        def gems_flash(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, **_):
            nonlocal flash_calls
            flash_calls += 1
            output = torch.ops.aten._flash_attention_forward.default(
                q,
                k,
                v,
                None,
                None,
                q.shape[-2],
                k.shape[-2],
                dropout_p,
                causal,
                False,
                scale=softmax_scale,
            )[0]
            return output

        modeling.flash_attn_func = gems_flash
        model.model._use_flash_attention_2 = True
        for layer in model.model.layers:
            layer.self_attn.__class__ = modeling.DeepseekV2FlashAttention2
            layer.self_attn._flash_attn_uses_top_left_mask = False
    with ctx, torch.inference_mode():
        if args.backend != "eager":
            result["registered_keys"] = flag_gems.all_registered_keys()
        for task, rows in data.items():
            warmup_inputs = [
                tokenizer(p, return_tensors="pt").input_ids.cuda()
                for p in warmup_data[task]
            ]
            inputs = [tokenizer(p, return_tensors="pt").input_ids.cuda() for p in rows]
            for input_id, ids in enumerate(warmup_inputs):
                for warm_id in range(args.warmup):
                    compile_stats["phase"] = f"{task}/warmup/{input_id}/{warm_id}"
                    print("BEGIN", compile_stats["phase"], flush=True)
                    decode(model, ids, args.max_new_tokens)
                    torch.cuda.synchronize()
                    print(
                        "END",
                        compile_stats["phase"],
                        "compiles",
                        compile_stats["count"],
                        flush=True,
                    )
            elapsed, outputs, recompiles = [], [], []
            for input_id, ids in enumerate(inputs):
                compile_stats["phase"] = f"{task}/measure/{input_id}"
                before = compile_stats["count"]
                print("BEGIN", compile_stats["phase"], flush=True)
                torch.cuda.synchronize()
                start = time.perf_counter()
                out = decode(model, ids, args.max_new_tokens)
                torch.cuda.synchronize()
                elapsed.append(time.perf_counter() - start)
                outputs.append(out[0, ids.shape[1] :].tolist())
                recompiles.append(compile_stats["count"] - before)
                print(
                    "END",
                    compile_stats["phase"],
                    "seconds",
                    elapsed[-1],
                    "recompiles",
                    recompiles[-1],
                    flush=True,
                )
            result["tasks"][task] = {
                "latency_s": elapsed,
                "latency_mean_s": statistics.mean(elapsed),
                "latency_p50_s": statistics.median(elapsed),
                "tokens_per_s": len(inputs) * args.max_new_tokens / sum(elapsed),
                "warmup": args.warmup,
                "generated_ids": outputs,
                "measured_recompiles": recompiles,
                "steady_state": not any(recompiles),
            }
    result["flash_attention_calls"] = flash_calls
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
