# Experiment Report

> Ping Wee Loo (piloo@ucsc.edu)

## Baseline Expectations
After implementing the methods and modules required by the assignment spec, I ran the complete pipeline to check the baseline with the default settings in `config.yaml`.

The results were as follows:
```yaml
d_model: 64
n_layers: 5
n_heads: 4
d_ff: 256
dropout: 0.0

learning rate: 1e-3
batch_size: 32
```
**val = 1.857**.

This clears the requirement for < 3.0, but still needs improvement to attain the 5% for being < 1.70

## Constraints

There are many knobs to tune in `config.yaml`, but I had to be mindful of the cap on the number of parameters the model could be, 500K.

Model Parameters are calculated by:



$$ params = 512 * dmodel + nlayers * (4 * dmodel^2 + 3 * dmodel * dff)$$
(no biases).

<!-- **Param cap:** 500,000, counted as
`params = 512·d_model + n_layers·(4·d_model² + 3·d_model·d_ff)` (tied head, no biases). -->

## Training

- I rented an RTX 4090 on Runpod in order to speed up training and evaluating. This is also in line with the reccomendation to train on CUDA for bf16 autocast and as the tiers are calibrated as such.

## Turning the levers

### Parameters

According to scaling laws, a larger model typically means a better model, so I pushed the architectural configs as close to 500K as I could.

From my past experience finetuning models, I also first reached to increase learning rates: <br>
From lr=1e-3 $\to$ lr=3e-3

These are the first runs results:

| config | params | val |
|---|---|---|
| **d96/L4/H8/ff256** | 491K | **~1.59** (best) |
| d96/L6/H8/ff128 | 491K | 1.61 |
| baseline d64/L5 | 360K | 1.86 |

As expected, raising lr=1e-3 $\to$ lr=3e-3 is an easy win.
The best result of 1.59 is a good improvement over the previous val of 1.86. 

### Batch size
Kept the config of (d96/L4/H8/ff256), and also experimented with increasing batch size. Then continued increasing LR by different amounts.

Here are the second run's results:

| batch | lr | val |
|---|---|---|
| 32 | 6e-3 | 1.553 |
| 64 | 8e-3 | 1.471 |
| **128** | **8e-3** | **1.429** (best) |
| 128 | 12e-3 | ~1.61 |

Increasing batch size to allow the model to see more data was helping, as seen by the decrease in loss(1.47→1.43) even when lr was kept still at 8e-3. 

Pushed learning rate to 12e-3, caused a worse result so we shall stay at 8e-3 for now.

### Other things that didnt help

Building on top of (bs128, lr8e-3), I tried to play with some other hyperparameters:
| change | val |
|---|---|
| min_lr 1e-4 → **1e-5** | **1.4289** |
| β₂ 0.95 → 0.99 | 1.4302 |
| weight_decay 0.1 → 0.05 | 1.4326 |

Which did barely anything for the val<br>
`d96/L4/bs128` has hit a plateau. <br>
Annealing to `min_lr=1e-5` is marginally best
and kept.

### Dropout

| dropout | val |
|---|---|
| 0.05 | 1.49 |
| 0.10 | 1.57 |

At this scale the model underfits, so regularization only slows convergence — no overfit to remove.


### Only things left to try
There were only two independent levers left to try: 
- Even larger batch size!!
- narrower heads at fixed d_model.

| change | val |
|---|---|
| bs256 / lr12e-3 (annealed) | **1.40** |
| bs128 / H6 (d_head 16) | **1.4217** |
| bs128 / H4 (d_head 24) | worse |

- batch_size=256 beats bs128
- `n_heads` 8→**6**
(d_head 12→**16**) is a clean win; d_head=16 is the sweet spot.

## Stacking the best options (batch_size=256 + H6)

| config | best val |
|---|---|
| **bs256 / H6 / lr12e-3** | **1.3883**  (global best) |
| bs256 / H6 / lr10e-3 | 1.3924 |
| bs256 / H8 / lr12e-3 | ~1.40 |
| bs256 / H4 / lr12e-3 | worse |

lr=12e-3 wins at bs256/H6, and at this point we are comfortably within the excellent grade.

Final `config.yaml`
```yaml
# Architecture (tunable — must satisfy constraints in assignment.pdf)
d_model: 96
n_layers: 4
n_heads: 6          
d_ff: 256
dropout: 0.0

# Training
batch_size: 256
learning_rate: 0.012
min_lr: 0.00001
warmup_steps: 600
weight_decay: 0.1
gradient_clip_norm: 1.0
beta1: 0.9
beta2: 0.95
```