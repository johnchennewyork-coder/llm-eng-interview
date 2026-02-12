#!/usr/bin/env python3
"""
=================================================================
  PyTorch Debug Challenge — Verification Dashboard
=================================================================
Run this after editing pytorch_debug_problem.py to check which
bugs you've fixed:

    python pytorch_debug_ui.py

Opens a Gradio web UI at http://localhost:7860
=================================================================
"""

import importlib.util
import io
import os
import re
import sys
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import gradio as gr
except ImportError:
    print("Installing gradio …")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gradio"])
    import gradio as gr


PROBLEM_FILE = os.path.join(os.path.dirname(__file__) or ".", "pytorch_debug_problem.py")

# ────────────────────────────────────────────────────────────────
# Static analysis helpers
# ────────────────────────────────────────────────────────────────

BUG_META = [
    {
        "id": 1,
        "name": "Softmax + CrossEntropyLoss",
        "category": "Convergence",
        "hint": "Remove nn.Softmax from the model — CrossEntropyLoss already applies log_softmax.",
    },
    {
        "id": 2,
        "name": "Missing optimizer.zero_grad()",
        "category": "Convergence",
        "hint": "Call optimizer.zero_grad() before loss.backward() in the training loop.",
    },
    {
        "id": 3,
        "name": "Memory leak — raw loss tensor",
        "category": "Memory",
        "hint": "Use loss.item() when recording the running loss.",
    },
    {
        "id": 4,
        "name": "Missing model.eval()",
        "category": "Correctness",
        "hint": "Call model.eval() at the start of validate().",
    },
    {
        "id": 5,
        "name": "Missing torch.no_grad()",
        "category": "Performance",
        "hint": "Wrap the validation loop with torch.no_grad().",
    },
    {
        "id": 6,
        "name": "Hardcoded CPU device",
        "category": "Performance",
        "hint": "Check for cuda / mps availability instead of hardcoding 'cpu'.",
    },
    {
        "id": 7,
        "name": "Training shuffle=False",
        "category": "Convergence",
        "hint": "Set shuffle=True for the training DataLoader.",
    },
    {
        "id": 8,
        "name": "Learning rate too high (0.5)",
        "category": "Convergence",
        "hint": "Use a sensible lr for Adam, e.g. 1e-3.",
    },
]


def _read_source() -> str:
    with open(PROBLEM_FILE, encoding="utf-8") as f:
        return f.read()


def _extract_function_body(source: str, func_name: str) -> str:
    """Rough extraction of a top-level function body."""
    pattern = rf"^def {func_name}\b.*?(?=\ndef |\nclass |\Z)"
    match = re.search(pattern, source, re.MULTILINE | re.DOTALL)
    return match.group(0) if match else ""


def _strip_comments_and_strings(source: str) -> str:
    """Remove comments and docstrings so we only inspect live code."""
    # Remove triple-quoted strings (docstrings / multi-line strings)
    source = re.sub(r'""".*?"""', "", source, flags=re.DOTALL)
    source = re.sub(r"'''.*?'''", "", source, flags=re.DOTALL)
    # Remove single-line # comments
    source = re.sub(r"#.*", "", source)
    return source


def check_bug_1(source: str) -> tuple[bool, str]:
    """nn.Softmax should not appear in live code (ignoring comments/docstrings)."""
    code_only = _strip_comments_and_strings(source)
    if re.search(r"nn\.Softmax", code_only):
        return False, "nn.Softmax still present in model architecture."
    return True, "nn.Softmax removed — correct!"


def check_bug_2(source: str) -> tuple[bool, str]:
    """optimizer.zero_grad() must be in the training loop."""
    train_body = _extract_function_body(source, "train_one_epoch")
    if "zero_grad()" in train_body:
        return True, "optimizer.zero_grad() found in training loop."
    return False, "optimizer.zero_grad() missing from training loop."


def check_bug_3(source: str) -> tuple[bool, str]:
    """loss should be stored via .item(), not as a raw tensor."""
    train_body = _extract_function_body(source, "train_one_epoch")
    # Bad pattern: append(loss) without .item()/.detach()
    if re.search(r"\.append\(\s*loss\s*\)", train_body):
        return False, "Raw loss tensor appended — use loss.item() to avoid memory leak."
    if re.search(r"\.append\(\s*loss\.(item|detach)", train_body):
        return True, "loss.item() / loss.detach() used — no memory leak."
    # Fallback: if the pattern changed significantly, assume OK
    return True, "Could not detect the original pattern (probably fixed)."


def check_bug_4(source: str) -> tuple[bool, str]:
    """model.eval() should appear in the validate function."""
    val_body = _extract_function_body(source, "validate")
    if "model.eval()" in val_body:
        return True, "model.eval() found in validate()."
    return False, "model.eval() missing from validate()."


def check_bug_5(source: str) -> tuple[bool, str]:
    """torch.no_grad() should wrap the validation loop."""
    val_body = _extract_function_body(source, "validate")
    if "no_grad()" in val_body:
        return True, "torch.no_grad() found in validate()."
    return False, "torch.no_grad() missing from validate()."


def check_bug_6(source: str) -> tuple[bool, str]:
    """Device selection should check for GPU availability."""
    main_body = _extract_function_body(source, "main")
    # Must reference cuda or mps — not just hardcode cpu
    if "cuda" in main_body or "mps" in main_body:
        return True, "GPU availability check detected."
    if re.search(r'device\s*=\s*torch\.device\(\s*["\']cpu', main_body):
        return False, "Device hardcoded to CPU — check for cuda/mps."
    return False, "No GPU selection logic found."


def check_bug_7(source: str) -> tuple[bool, str]:
    """Training DataLoader should have shuffle=True."""
    main_body = _extract_function_body(source, "main")
    # Look for shuffle=True associated with train_loader
    if re.search(r"train_loader\s*=.*shuffle\s*=\s*True", main_body, re.DOTALL):
        return True, "Training DataLoader has shuffle=True."
    if "shuffle=True" in main_body:
        return True, "shuffle=True detected in main()."
    return False, "Training DataLoader still has shuffle=False."


def check_bug_8(source: str) -> tuple[bool, str]:
    """Learning rate should be reasonable (≤ 0.01 for Adam)."""
    main_body = _extract_function_body(source, "main")
    # Match lr= followed by a numeric literal (handles 0.001, 1e-3, 1E-4, etc.)
    match = re.search(r"lr\s*=\s*([\d.]+(?:[eE][+-]?\d+)?)", main_body)
    if match:
        try:
            lr = float(match.group(1))
        except ValueError:
            return False, f"Could not parse lr value: {match.group(1)}"
        if lr <= 0.01:
            return True, f"lr={lr} — reasonable for Adam."
        return False, f"lr={lr} — too high for Adam (try 1e-3)."
    return False, "Could not find lr= in optimizer definition."


CHECKERS = [
    check_bug_1,
    check_bug_2,
    check_bug_3,
    check_bug_4,
    check_bug_5,
    check_bug_6,
    check_bug_7,
    check_bug_8,
]


# ────────────────────────────────────────────────────────────────
# Dynamic analysis — actually run training
# ────────────────────────────────────────────────────────────────

def _run_training() -> tuple[dict | None, str]:
    """Import the problem file fresh and run main(). Returns (history, stdout)."""
    # Remove stale module from cache
    mod_name = "pytorch_debug_problem"
    if mod_name in sys.modules:
        del sys.modules[mod_name]

    spec = importlib.util.spec_from_file_location(mod_name, PROBLEM_FILE)
    mod = importlib.util.module_from_spec(spec)

    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        spec.loader.exec_module(mod)
        history = mod.main()
    except Exception:
        sys.stdout = old_stdout
        return None, buf.getvalue() + "\n" + traceback.format_exc()
    finally:
        sys.stdout = old_stdout

    return history, buf.getvalue()


def _plot_history(history: dict):
    """Return a matplotlib Figure with loss & accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], "o-", label="Train", markersize=3)
    ax1.plot(epochs, history["val_loss"], "s-", label="Val", markersize=3)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], "o-", label="Train", markersize=3)
    ax2.plot(epochs, history["val_acc"], "s-", label="Val", markersize=3)
    ax2.axhline(y=0.85, color="green", ls="--", alpha=0.6, label="85 % target")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


# ────────────────────────────────────────────────────────────────
# Gradio callbacks
# ────────────────────────────────────────────────────────────────

def run_static_analysis():
    """Read the problem file and check all 8 bugs."""
    try:
        source = _read_source()
    except FileNotFoundError:
        return "<p style='color:red;'>pytorch_debug_problem.py not found!</p>", "—"

    results = []
    fixed = 0
    for checker, meta in zip(CHECKERS, BUG_META):
        ok, msg = checker(source)
        results.append((meta, ok, msg))
        fixed += int(ok)

    # Build HTML table
    rows = ""
    for meta, ok, msg in results:
        bg = "#d4edda" if ok else "#f8d7da"
        icon = "✅" if ok else "❌"
        rows += (
            f"<tr style='background:{bg};'>"
            f"<td style='padding:8px;text-align:center;'><b>{meta['id']}</b></td>"
            f"<td style='padding:8px;'>{meta['name']}</td>"
            f"<td style='padding:8px;'>{meta['category']}</td>"
            f"<td style='padding:8px;text-align:center;font-size:18px;'>{icon}</td>"
            f"<td style='padding:8px;font-size:13px;'>{msg}</td>"
            f"<td style='padding:8px;font-size:12px;color:#666;'>{meta['hint']}</td>"
            f"</tr>"
        )

    table = (
        "<table style='width:100%;border-collapse:collapse;font-family:system-ui;'>"
        "<thead><tr style='background:#1a1a2e;color:white;'>"
        "<th style='padding:8px;'>#</th>"
        "<th style='padding:8px;text-align:left;'>Bug</th>"
        "<th style='padding:8px;text-align:left;'>Category</th>"
        "<th style='padding:8px;'>Status</th>"
        "<th style='padding:8px;text-align:left;'>Details</th>"
        "<th style='padding:8px;text-align:left;'>Hint</th>"
        "</tr></thead><tbody>" + rows + "</tbody></table>"
    )

    # Score badge
    pct = fixed / len(CHECKERS) * 100
    if pct >= 90:
        color = "#28a745"
    elif pct >= 60:
        color = "#ffc107"
    else:
        color = "#dc3545"

    score_html = (
        f"<div style='text-align:center;padding:16px;'>"
        f"<div style='font-size:56px;font-weight:bold;color:{color};'>"
        f"{fixed}/{len(CHECKERS)}</div>"
        f"<div style='font-size:16px;color:#888;'>Bugs Fixed ({pct:.0f} %)</div>"
        f"</div>"
    )

    return table, score_html


def run_full_check():
    """Static analysis + training run."""
    table_html, score_html = run_static_analysis()

    # Run training
    history, stdout = _run_training()

    if history is None:
        return (
            table_html,
            score_html,
            stdout,
            None,
            "<p style='color:red;'>Training crashed — see log above.</p>",
        )

    fig = _plot_history(history)

    final_val_acc = history["val_acc"][-1]
    avg_time = np.mean(history["epoch_time"])
    converged = final_val_acc > 0.85

    summary_color = "#28a745" if converged else "#dc3545"
    summary = (
        f"<div style='padding:12px;border-left:4px solid {summary_color};background:#f8f9fa;'>"
        f"<b>Final val accuracy:</b> {final_val_acc:.2%} "
        f"{'✅ Converged!' if converged else '❌ Did not converge (need >85%)'}<br>"
        f"<b>Avg epoch time:</b> {avg_time:.3f}s"
        f"</div>"
    )

    return table_html, score_html, stdout, fig, summary


# ────────────────────────────────────────────────────────────────
# Gradio UI
# ────────────────────────────────────────────────────────────────

def build_ui():
    with gr.Blocks(title="PyTorch Debug Challenge") as demo:
        gr.Markdown(
            "# 🔬 PyTorch Debug Challenge — Verification Dashboard\n"
            "Edit **`pytorch_debug_problem.py`** in your editor, then click a button below "
            "to check your progress.\n\n"
            "- **Analyze Code** — fast static checks (no training)\n"
            "- **Analyze + Run Training** — static checks *and* trains the model to verify convergence"
        )

        with gr.Row():
            analyze_btn = gr.Button("🔍  Analyze Code", variant="secondary", scale=1)
            full_btn = gr.Button("🚀  Analyze + Run Training", variant="primary", scale=1)

        with gr.Row():
            with gr.Column(scale=3):
                analysis_html = gr.HTML(label="Bug Analysis")
            with gr.Column(scale=1):
                score_html = gr.HTML(label="Score")

        gr.Markdown("---")
        gr.Markdown("### Training Results")
        summary_html = gr.HTML()

        with gr.Row():
            with gr.Column():
                plot_out = gr.Plot(label="Training Curves")
            with gr.Column():
                log_out = gr.Textbox(
                    label="Training Log",
                    lines=18,
                    max_lines=30,
                    interactive=False,
                )

        # Wire buttons
        analyze_btn.click(
            fn=run_static_analysis,
            outputs=[analysis_html, score_html],
        )
        full_btn.click(
            fn=run_full_check,
            outputs=[analysis_html, score_html, log_out, plot_out, summary_html],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(theme=gr.themes.Soft())
