# PyTorch Debug Challenge — Interviewer Rubric

## Overview

| Item | Detail |
|------|--------|
| **Level** | Hard |
| **Time** | 45 minutes (suggested) |
| **Bugs** | 8 total |
| **Goal** | >85 % val accuracy within 20 epochs once all bugs are fixed |
| **Categories** | Convergence (4), Performance (2), Memory (1), Correctness (1) |

Hand the candidate `pytorch_debug_problem.py` and tell them:

> *"This training script should reach over 85 % accuracy within 20 epochs,
> but it barely learns.  There are 8 bugs — find and fix as many as you
> can.  Think out loud as you go."*

---

## Bug Table

| # | Bug | Category | Location | Difficulty | What Happens |
|---|-----|----------|----------|------------|-------------|
| 1 | `nn.Softmax` before `CrossEntropyLoss` | Convergence | `SpiralNet.classifier` | Medium-Hard | `CrossEntropyLoss` already applies `log_softmax`. Double-softmax squashes gradients → model barely learns. |
| 2 | Missing `optimizer.zero_grad()` | Convergence | `train_one_epoch` | Medium | Gradients accumulate across batches → erratic, exploding updates → divergence. |
| 3 | `running_loss.append(loss)` instead of `loss.item()` | Memory | `train_one_epoch` | Medium | Keeps the entire backward graph alive per batch → GPU/CPU memory grows linearly per epoch. |
| 4 | Missing `model.eval()` in validation | Correctness | `validate` | Easy-Medium | Dropout still active + BatchNorm uses batch stats → noisy, unreliable val metrics. |
| 5 | Missing `torch.no_grad()` in validation | Performance / Memory | `validate` | Easy-Medium | Autograd builds backward graphs during inference → wasted memory and slower eval. |
| 6 | `device = torch.device("cpu")` hardcoded | Performance | `main` | Easy | Ignores available GPU (CUDA / MPS) → training is orders of magnitude slower. |
| 7 | `shuffle=False` on training DataLoader | Convergence | `main` | Easy-Medium | Model sees same sample order every epoch → poor generalisation, slower convergence. |
| 8 | `lr=0.5` with Adam | Convergence | `main` | Easy | Adam default is 1e-3. A learning rate of 0.5 causes wild oscillation / divergence. |

---

## Detailed Probing Questions

### Bug 1 — Softmax + CrossEntropyLoss

- *"What does `nn.CrossEntropyLoss` compute internally?"*
  - Expected: It combines `log_softmax` + `NLLLoss`.
- *"What happens to the gradient of `log(softmax(softmax(x)))` as x grows?"*
  - Expected: Gradients become vanishingly small; the double softmax compresses the distribution and its Jacobian.
- *"How would you catch this kind of bug in production code?"*
  - Bonus: Check that raw logits fed to CE have a wide range (not [0, 1]).

### Bug 2 — Missing zero_grad

- *"What is the default accumulation behaviour of `.backward()` in PyTorch?"*
  - Expected: Gradients are **added** to existing `.grad` buffers. Without zeroing, they compound.
- *"When might you intentionally skip `zero_grad`?"*
  - Expected: Gradient accumulation over micro-batches to simulate a larger effective batch size.

### Bug 3 — Memory Leak (loss tensor)

- *"What does appending a `loss` tensor — instead of `loss.item()` — keep alive?"*
  - Expected: The entire autograd computation graph for that batch remains in memory.
- *"How would you detect this in a real training run?"*
  - Expected: Watch GPU memory with `nvidia-smi` / `torch.cuda.memory_allocated()`; memory should **not** grow linearly per batch.

### Bug 4 — Missing model.eval()

- *"Which layers behave differently in eval vs. train mode?"*
  - Expected: `Dropout` (disabled in eval), `BatchNorm` (uses running stats in eval).
- *"Could this bug affect training too, not just validation?"*
  - Expected: If you forget to call `model.train()` before the next epoch, yes. Here `model.train()` is correctly called at the start of `train_one_epoch`, so the effect is limited to validation being noisy.

### Bug 5 — Missing torch.no_grad()

- *"Besides memory, is there a numerical difference?"*
  - Expected: No difference in the forward values — only in whether autograd metadata is recorded.
- *"What about `@torch.inference_mode()` — how does it compare?"*
  - Bonus: `inference_mode` is stricter (disallows any grad-requiring ops), slightly faster.

### Bug 6 — Hardcoded CPU

- *"How would you write device selection that works on CUDA, MPS, and CPU?"*
  - Expected: Check `torch.cuda.is_available()`, then `torch.backends.mps.is_available()`, else CPU.
- *"What else do you need to move besides the model?"*
  - Expected: Data tensors (already `.to(device)` in the loops), criterion if it has buffers.

### Bug 7 — shuffle=False

- *"Why does sample ordering matter for SGD convergence?"*
  - Expected: Correlated samples in consecutive batches increase gradient variance and can bias updates.
- *"Should you shuffle the validation set too?"*
  - Expected: Not necessary — evaluation is order-independent.

### Bug 8 — Learning Rate Too High

- *"How would you diagnose a bad learning rate in practice?"*
  - Expected: Loss NaN/Inf, oscillation, or flat loss curve. Use an LR finder sweep.
- *"What is Adam's typical recommended range?"*
  - Expected: 1e-4 to 1e-3 for most tasks.

---

## Scoring Guide

| Bugs Fixed | Rating | Interpretation |
|------------|--------|----------------|
| **8 / 8** | Strong Hire | Deep PyTorch expertise; catches subtle memory and convergence issues. |
| **6–7 / 8** | Hire | Solid understanding; may miss one subtle issue (e.g., memory leak or softmax). |
| **4–5 / 8** | Borderline | Knows basics (LR, zero_grad) but misses model-eval or memory nuances. |
| **2–3 / 8** | Below Bar | Limited hands-on PyTorch experience. |
| **0–1 / 8** | No Hire | Cannot debug basic training loops. |

### Bonus Points (not required)

- Suggests using a **learning-rate scheduler** (e.g., CosineAnnealing).
- Suggests `torch.compile()` or mixed-precision (`torch.autocast`) for performance.
- Mentions `pin_memory=True` for CUDA data loading.
- Identifies that `num_workers=0` is suboptimal for larger datasets.
- Adds gradient clipping as a safety net.

---

## Suggested Timeline

| Phase | Minutes | Notes |
|-------|---------|-------|
| Read & understand code | 5–10 | Candidate should scan the full file before editing. |
| Identify & fix bugs | 20–25 | Encourage thinking out loud. |
| Run & verify fixes | 5 | Use the verification UI or run the script directly. |
| Discussion / probing Qs | 5–10 | Pick 2–3 bugs to dive deeper on. |

---

## Running the Challenge

```bash
# Show broken behaviour
python pytorch_debug_problem.py

# After candidate edits the file, verify
python pytorch_debug_ui.py

# Or compare against the reference solution
python pytorch_debug_solution.py
```
