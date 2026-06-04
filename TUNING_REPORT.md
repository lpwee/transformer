# Transformer LM — Tuning Log & Strategy

> Working notes for minimizing validation loss within the assignment budget.
> Not part of the 6-file submission ZIP; intended to seed the Canvas experiment report.

## Problem & constraints

Decoder-only Transformer (RoPE, RMSNorm w/o gain, SwiGLU FFN, weight-tied head) on
TinyStories. The implementation was already complete and passes all 85 unit tests; this
log is purely about the **training-quality / loss-optimization** part.

**Fixed by staff (cannot change):** `vocab_size=512`, `context_length=256`,
`max_steps=5000`, `seed=42`, optimizer `AdamW`, bf16 autocast on CUDA.

**Param cap:** 500,000, counted as
`params = 512·d_model + n_layers·(4·d_model² + 3·d_model·d_ff)` (tied head, no biases).

**Tunable** (`config.yaml` + `lr_schedule.py`): `d_model, n_layers, n_heads, d_ff,
dropout, learning_rate, min_lr, batch_size, warmup_steps, weight_decay,
gradient_clip_norm, beta1, beta2`, and the LR schedule shape.

**Targets:** sanity `val<3.0`; Part 5 `val≤1.70`; bonus **hidden-test** loss
`≤1.53 / 1.47 / 1.43` (Pass / Good / Excellent). Local val ≈ test − (small gap);
staff reference: default ≈ 1.85 val ≈ 1.92 test, so I treat **val ≈ test − ~0.06** and
aim well below the target on val.

## Method

- Every run is the real 5000-step budget on an RTX 4090 (bf16), so numbers are directly
  comparable to grading. Runs are launched in parallel (`tools/psweep.sh`) and tracked by
  a live leaderboard watchdog (`tools/watchdog.py`).
- **One lever per wave.** Screen architecture first, then attack the biggest remaining
  lever, freezing what's already decided. Coarser `eval_interval=250` during sweeps;
  fine eval reserved for the final pick.
- `tools/paramcalc.py --enum` enumerates valid configs in the top 10% of the param
  budget so every architecture candidate actually spends the 500K.

## Results by wave

Baseline (untuned default `d64/L5/H4/ff256`, lr1e-3, bs32) → **val ≈ 1.857**.

### Wave 1 — architecture screen (shared lr=3e-3, bs32, min_lr=1e-4)
Ten near-cap shapes varying depth/width/ff-ratio.

| config | params | best val |
|---|---|---|
| **d96/L4/H8/ff256** | 491K | **~1.59** (best) |
| d112/L4/H8/ff176 | 494K | ~1.58 |
| d96/L6/H8/ff128 | 491K | 1.61 |
| d96/L5/H8/ff176 | 487K | 1.62 |
| baseline d64/L5 | 360K | 1.86 |

*Takeaways:* (1) just raising LR 1e-3→3e-3 moved baseline-shape loss ~1.86→~1.59 — the
biggest early win. (2) Among near-cap shapes the spread is small (~0.03); `d96/L4/H8/ff256`
(ff/d≈2.67, the SwiGLU-friendly ratio) is the pick. Queued lower-priority shapes were
cancelled once the winner was clear, to save GPU.

### Wave 2 — batch size = total-tokens lever (on d96/L4/H8/ff256)
At a **fixed** 5000 steps, batch size sets how much data is seen
(bs32≈2.8 epochs … bs256≈22 epochs). LR scaled up with batch.

| batch | lr | best val |
|---|---|---|
| 32 | 6e-3 | 1.553 |
| 64 | 8e-3 | 1.471 |
| **128** | **8e-3** | **1.429** |
| 128 | 12e-3 | ~1.61 (LR too high) |

*Takeaway:* batch size was the **single biggest lever** (1.55→1.43). bs128/lr8e-3 is the
sweet spot; pushing LR to 12e-3 at bs128 destabilized it.

### Wave 3 — HP refinement (bs128, lr8e-3)
| change | best val |
|---|---|
| min_lr 1e-4 → **1e-5** | **1.4289** |
| β₂ 0.95 → 0.99 | 1.4302 |
| weight_decay 0.1 → 0.05 | 1.4326 |

*Takeaway:* these knobs are near-neutral (±0.005). Six configs cluster at **1.428–1.434
val** — `d96/L4/bs128` has hit a plateau. Annealing to `min_lr=1e-5` is marginally best
and kept. Depth re-tests (L5/L6) at bs128 did not beat L4.

### Wave 4 — dropout / bs256 / WSD (plateau held)
- **dropout** 0.05/0.10: *worse* (1.49 / 1.57). At this scale the model underfits, so
  regularization only slows convergence — no overfit to remove.
- **WSD schedule** (warmup→stable→decay): 1.456, behind cosine. Cosine kept.
- **bs256**: the runs OOM'd here purely from running 4 procs at once (4×~6GB > 24GB),
  not because bs256 is too big — retried properly in wave 5.

### Wave 5 — proper large batch (low concurrency) + head dimension
- **bs256/lr12e-3** (annealed): **1.40** — bs256 *does* beat bs128, but only after the
  cosine tail; mid-run it looked worse, which is why earlier coarse reads undersold it.
- **head dimension**: `n_heads` 8→**6** (d_head 12→**16**) improved 1.4289→**1.4217** at
  bs128. d_head=16 is the sweet spot; H4 (d_head 24) was worse.

### Wave 6 — combine the two winners (bs256 + H6)
The levers stack almost additively:

| config | best val |
|---|---|
| **bs256 / H6 / lr12e-3** | **1.3883** ← global best |
| bs256 / H6 / lr10e-3 | 1.3924 |
| bs256 / H8 / lr12e-3 | ~1.40 |
| bs256 / H4 / lr12e-3 | worse |

## Final config (in `config.yaml`)

```
d_model 96  n_layers 4  n_heads 6 (d_head 16)  d_ff 256  dropout 0.0
batch_size 256  learning_rate 0.012  min_lr 1e-5  warmup_steps 600
weight_decay 0.1  gradient_clip_norm 1.0  beta1 0.9  beta2 0.95
```
**491,520 params (98.3% of cap). Best local val ≈ 1.388** (seed 42, 5000 steps, bf16/CUDA)
— down from the 1.857 untuned default. Clears Part 5 (≤1.70) by a wide margin. Hidden
**test** loss runs a small gap above local val (staff ref: 1.85 val ≈ 1.92 test), so this
maps to roughly the **Good** test tier (~1.45) and may graze **Excellent** (≤1.43) if the
gap is small. Schedule is the stock cosine + linear warmup (WSD lost, so `lr_schedule.py`
was reverted to the clean cosine).

## What didn't help / dead ends
- **LR too high**: ≥1e-2 at bs128, and 14e-3 at bs256 — diverged early / ended worse.
- **dropout**: hurts here (underfitting regime, not overfitting).
- **WSD schedule**: behind cosine.
- **β₂ (0.99), weight_decay (0.05), extra depth (L5/L6), H4 (d_head 24)**: neutral or worse.
- **bs384**: no clear gain over bs256 (batch returns flattened).

## Verification (pre-submission)
- `pytest tests/` → **85/85 pass**; `sanity_check.py` → overfits to 0.0019.
- `validate_submission.py submission.zip` → **PASSED**.
- `final_check.py` steps 1–5 pass. Step 6 (50-step *training preview*) **times out — but
  only because that step forces `--device cpu` and inherits `batch_size=256`**, which is
  far too slow on CPU. Verified the identical preview on **GPU**: val loss decreases
  5.82→4.70 in 9s, and the full 5000-step GPU run reaches 1.388. Step 7 equivalent
  (strict state-dict round-trip + `lm_head.weight is token_emb.weight` weight tying)
  passes. Grading retrains on GPU, so bs256 is not an issue there.