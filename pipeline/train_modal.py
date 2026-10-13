"""QLoRA fine-tune of Llama 3.1 8B Instruct on Modal (step 6).

    caffeinate -i modal run --detach pipeline/train_modal.py
    modal run pipeline/train_modal.py --smoke          # same code and model on 8 rows, 1 epoch; output to models/smoke
    modal run pipeline/train_modal.py --fetch-only     # if the local client dropped mid-run

Settings follow the plan: 4-bit NF4 base (real QLoRA; v1 loaded 8-bit), LoRA r=8,
alpha=16, dropout 0.05 on q_proj/v_proj, 2 epochs, lr 2e-4, batch 1 x grad-accum 8.
Loss is computed on the assistant answer only (v1 also trained on the prompt).
valid.jsonl is scored after every epoch by generating answers with the serving prompt,
and the checkpoint is selected by valid fact coverage, not train loss.

Training rows go in with the call and are hashed inside the job. Results (adapter,
adapter_config.json, tokenizer, trainer_state.json, per-epoch valid scores) are saved to
the `visabuddy-train-out` volume and downloaded to visa_buddy_data/models/v2/.
"""
import hashlib
import json
from pathlib import Path

import modal

BASE = "meta-llama/Llama-3.1-8B-Instruct"
GPU = "A100"  # bf16 support and enough memory for 4k-token rows; the full run took 13 minutes
# Settings from the plan (same r/alpha/targets/lr/batch as v1, so the comparison is about data
# and method, not knobs). max_len 4096 fits system prompt + 5 sources + answer; no row exceeded it.
HP = {"lora_r": 8, "lora_alpha": 16, "lora_dropout": 0.05, "target_modules": ["q_proj", "v_proj"],
      "epochs": 2, "learning_rate": 2e-4, "per_device_batch": 1, "grad_accum": 8, "max_len": 4096,
      "warmup_ratio": 0.05, "lr_scheduler": "cosine", "seed": 13, "quantization": "4bit-nf4, double quant",
      "compute_dtype": "bfloat16", "max_new_tokens_valid": 400}

ROOT = Path(__file__).resolve().parent.parent
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch")
    .pip_install_from_requirements(str(ROOT / "requirements-gpu.txt"))
    .add_local_python_source("pipeline")  # the job scores validation with the same grading code as eval
)
hf_cache = modal.Volume.from_name("visabuddy-hf-cache", create_if_missing=True)
out_vol = modal.Volume.from_name("visabuddy-train-out", create_if_missing=True)
app = modal.App("visabuddy-train", image=image)


def _digest(obj) -> str:  # proves the job trained on exactly the rows this Mac sent
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _messages(row, with_answer):  # same system/user messages the chat server builds
    m = [{"role": "system", "content": row["instruction"]}, {"role": "user", "content": row["input"]}]
    return m + [{"role": "assistant", "content": row["output"]}] if with_answer else m


@app.function(gpu=GPU, timeout=3 * 60 * 60, secrets=[modal.Secret.from_name("hf-secret")],
              volumes={"/root/.cache/huggingface": hf_cache, "/out": out_vol})
def train(train_rows: list, valid_rows: list, smoke: bool = False) -> dict:
    import os
    import shutil

    import peft
    import torch
    import transformers
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer,
                              TrainerCallback, TrainingArguments)

    from pipeline.valid_metrics import score_valid

    hp = {**HP, **({"epochs": 1, "max_new_tokens_valid": 32} if smoke else {})}
    token = next((os.environ[k] for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN", "HF_API_TOKEN")
                  if os.environ.get(k)), None)
    assert token, "hf-secret has no HF token under a known key"
    torch.manual_seed(HP["seed"])  # reproducible LoRA init and data order
    tok = AutoTokenizer.from_pretrained(BASE, token=token)
    tok.pad_token = tok.pad_token or tok.eos_token

    def encode(row):
        # return_dict=False: in transformers 5.x the default is a BatchEncoding, not token ids
        prompt = tok.apply_chat_template(_messages(row, False), add_generation_prompt=True, tokenize=True, return_dict=False)
        full = tok.apply_chat_template(_messages(row, True), tokenize=True, return_dict=False)
        assert full[:len(prompt)] == prompt, "chat template is not prefix-stable"
        return {"input_ids": full, "labels": [-100] * len(prompt) + full[len(prompt):]}

    def dataset(rows):
        enc = [encode(r) for r in rows]
        kept = [e for e in enc if len(e["input_ids"]) <= HP["max_len"]]
        return kept, len(enc) - len(kept)

    train_ds, train_dropped = dataset(train_rows)
    valid_ds, valid_dropped = dataset(valid_rows)

    def collate(batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), tok.pad_token_id)
        lab = torch.full((len(batch), n), -100)
        att = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            k = len(b["input_ids"])
            ids[i, :k], lab[i, :k], att[i, :k] = torch.tensor(b["input_ids"]), torch.tensor(b["labels"]), 1
        return {"input_ids": ids, "labels": lab, "attention_mask": att}

    model = AutoModelForCausalLM.from_pretrained(
        BASE, token=token, device_map={"": 0}, dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True))
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                            gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(r=HP["lora_r"], lora_alpha=HP["lora_alpha"], lora_dropout=HP["lora_dropout"],
                                             target_modules=HP["target_modules"], task_type="CAUSAL_LM"))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    @torch.inference_mode()
    def generate_valid():
        model.eval()
        outs = []
        for r in valid_rows:
            enc = tok.apply_chat_template(_messages(r, False), add_generation_prompt=True, return_tensors="pt", return_dict=True)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            o = model.generate(**enc, max_new_tokens=hp["max_new_tokens_valid"], do_sample=False, pad_token_id=tok.pad_token_id,
                               use_cache=True)  # gradient checkpointing turns the cache off by default
            outs.append(tok.decode(o[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip())
        model.train()
        return outs

    epoch_scores, ckpt_dirs = [], []

    class ValidGen(TrainerCallback):
        def on_epoch_end(self, args, state, control, **kw):
            outs = generate_valid()
            s = score_valid(valid_rows, outs)
            ep = round(state.epoch)
            d = f"/tmp/ckpt-epoch{ep}"
            model.save_pretrained(d)
            ckpt_dirs.append(d)
            epoch_scores.append({"epoch": ep, "step": state.global_step, **s, "outputs": outs})
            print(json.dumps({"epoch": ep, **s}), flush=True)

    args = TrainingArguments(
        output_dir="/tmp/run", num_train_epochs=hp["epochs"], learning_rate=HP["learning_rate"],
        per_device_train_batch_size=HP["per_device_batch"], per_device_eval_batch_size=1,
        gradient_accumulation_steps=HP["grad_accum"], warmup_steps=HP["warmup_ratio"],  # float < 1 = ratio in transformers 5.x
        lr_scheduler_type=HP["lr_scheduler"], logging_steps=1, eval_strategy="epoch", save_strategy="no",
        bf16=True, seed=HP["seed"], report_to=[], remove_unused_columns=False)
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=valid_ds,
                      data_collator=collate, callbacks=[ValidGen()])
    trainer.train()

    best = max(range(len(epoch_scores)), key=lambda i: (epoch_scores[i]["fact_coverage"] or 0, epoch_scores[i]["epoch"]))
    out = Path("/out/smoke" if smoke else "/out/v2")
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(ckpt_dirs[best], out)
    tok.save_pretrained(out)
    trainer.state.save_to_json(str(out / "trainer_state.json"))
    (out / "valid_scores.json").write_text(json.dumps(
        {"selected_epoch": epoch_scores[best]["epoch"], "selection_metric": "valid fact_coverage",
         "epochs": [{k: v for k, v in e.items() if k != "outputs"} for e in epoch_scores]}, indent=1))
    with open(out / "valid_outputs.jsonl", "w") as f:
        for e in epoch_scores:
            for r, o in zip(valid_rows, e["outputs"]):
                f.write(json.dumps({"epoch": e["epoch"], "id": r["id"], "kind": r["kind"], "output": o}, ensure_ascii=False) + "\n")
    meta = {"base": BASE, "gpu": GPU, "hp": hp, "smoke": smoke, "trainable_params": trainable,
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
            "received_sha256": {"train": _digest(train_rows), "valid": _digest(valid_rows)},
            "rows": {"train": len(train_ds), "valid": len(valid_ds), "train_dropped_too_long": train_dropped,
                     "valid_dropped_too_long": valid_dropped}}
    (out / "train_meta.json").write_text(json.dumps(meta, indent=1))
    out_vol.commit()
    return {"files": {p.name: p.read_bytes() for p in out.iterdir() if p.is_file()}, "meta": meta}


@app.function(volumes={"/out": out_vol})
def fetch() -> dict:
    out = Path("/out/v2")
    return {"files": {p.name: p.read_bytes() for p in out.iterdir() if p.is_file()},
            "meta": json.loads((out / "train_meta.json").read_text())}


@app.local_entrypoint()
def main(fetch_only: bool = False, smoke: bool = False):
    sft = ROOT / "visa_buddy_data" / "sft"
    train_rows = [json.loads(l) for l in (sft / "train.jsonl").read_text().splitlines() if l.strip()]
    valid_rows = [json.loads(l) for l in (sft / "valid.jsonl").read_text().splitlines() if l.strip()]
    print(f"training on {len(train_rows)} rows, validating on {len(valid_rows)} ({GPU})", flush=True)
    if smoke:  # plumbing check only: same code and model, tiny data, output kept apart
        train_rows, valid_rows = train_rows[:8], valid_rows[:2]
    res = fetch.remote() if fetch_only else train.remote(train_rows, valid_rows, smoke)
    assert res["meta"]["received_sha256"] == {"train": _digest(train_rows), "valid": _digest(valid_rows)}, \
        "job trained on different data"
    dest = ROOT / "visa_buddy_data" / "models" / ("smoke" if smoke else "v2")
    dest.mkdir(parents=True, exist_ok=True)
    for name, data in res["files"].items():
        (dest / name).write_bytes(data)
    print("wrote", dest, sorted(res["files"]))
