#!/usr/bin/env python3
"""논문 Figure 생성 — results.json 기반, paper_ieie/figures/*.pdf 출력."""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "paper_ieie", "figures")
os.makedirs(OUT, exist_ok=True)
R = json.load(open(os.path.join(HERE, "results.json")))
V = R["variants"]

plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "font.sans-serif": ["AppleGothic", "NanumGothic", "Arial Unicode MS", "DejaVu Sans"],
    "axes.unicode_minus": False,
})
C = {"scalar": "#9aa0a6", "cmsis": "#4c8bf5", "fused": "#e8710a", "direct": "#f4b400"}

# ---------------- Fig: latency comparison ----------------
def fig_latency():
    keys = ["V1_scalar", "V2_tflm_cmsis", "V3_tflm_fused"]
    labels = ["Scalar", "TFLM+\nCMSIS-NN", "TFLM+Fusion\n(ours)"]
    colors = [C["scalar"], C["cmsis"], C["fused"]]
    total = [V[k]["total_ms_per_image"] for k in keys]
    nn = [V[k]["nn_ms_per_image"] for k in keys]
    if any(v is None for v in total):
        print("latency: missing data, skip"); return
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    x = np.arange(len(keys))
    b = ax.bar(x, total, 0.55, color=colors, edgecolor="black", linewidth=0.5)
    for xi, (t, n) in enumerate(zip(total, nn)):
        ax.text(xi, t + max(total) * 0.015, f"{t:,.0f}", ha="center", fontsize=8.5, fontweight="bold")
        sp = total[0] / t
        if xi > 0:
            ax.text(xi, t * 0.45, f"{sp:.2f}$\\times$", ha="center", fontsize=9, color="white", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Latency per image, T=30 (ms)")
    ax.set_ylim(0, max(total) * 1.12)
    fig.savefig(f"{OUT}/fig_latency.pdf"); plt.close(fig)
    print("fig_latency.pdf")

# ---------------- Fig: per-timestep + runtime overhead ----------------
def fig_timestep():
    keys = ["V1_scalar", "V2_tflm_cmsis", "V4_direct_fused", "V3_tflm_fused"]
    labels = ["Scalar", "TFLM+\nCMSIS-NN", "Fusion\n(direct)", "Fusion\n(TFLM)"]
    colors = [C["scalar"], C["cmsis"], C["direct"], C["fused"]]
    nn = [V[k]["nn_ms_per_image"] for k in keys]
    if any(v is None for v in nn):
        print("timestep: missing data, skip"); return
    ts = [v / R["timesteps"] for v in nn]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    x = np.arange(len(keys))
    ax.bar(x, ts, 0.55, color=colors, edgecolor="black", linewidth=0.5)
    for xi, t in enumerate(ts):
        ax.text(xi, t + max(ts) * 0.015, f"{t:.1f}", ha="center", fontsize=8.5, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("NN latency per timestep (ms)")
    fig.savefig(f"{OUT}/fig_timestep.pdf"); plt.close(fig)
    print("fig_timestep.pdf")

# ---------------- Fig: memory ----------------
def fig_memory():
    keys = ["V1_scalar", "V2_tflm_cmsis", "V3_tflm_fused"]
    labels = ["Scalar", "TFLM+\nCMSIS-NN", "TFLM+Fusion\n(ours)"]
    ram = [V[k]["ram_bytes"] / 1024 for k in keys]
    flash = [V[k]["flash_bytes"] / 1024 for k in keys]
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    x = np.arange(len(keys)); w = 0.36
    ax.bar(x - w / 2, ram, w, label="SRAM", color="#4c8bf5", edgecolor="black", linewidth=0.5)
    ax.bar(x + w / 2, flash, w, label="Flash", color="#e8710a", edgecolor="black", linewidth=0.5)
    ax.axhline(320, color="#4c8bf5", ls="--", lw=0.8); ax.text(-0.45, 332, "SRAM 320 KiB", fontsize=7, color="#4c8bf5", ha="left")
    ax.axhline(1024, color="#e8710a", ls="--", lw=0.8); ax.text(2.45, 1034, "Flash 1,024 KiB", fontsize=7, color="#e8710a", ha="right")
    for xi, (r, f) in enumerate(zip(ram, flash)):
        ax.text(xi - w / 2, r + 12, f"{r:.0f}", ha="center", fontsize=7.5)
        ax.text(xi + w / 2, f + 12, f"{f:.0f}", ha="center", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Memory (KiB)"); ax.set_ylim(0, 1100); ax.legend(fontsize=8, loc="upper left")
    fig.savefig(f"{OUT}/fig_memory.pdf"); plt.close(fig)
    print("fig_memory.pdf")

# ---------------- 공통 다이어그램 헬퍼 ----------------
def _box(ax, x, y, w, h, text, fc, fs=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008",
                                fc=fc, ec="black", lw=0.7, mutation_aspect=0.5))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

def _arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=8, color="black", lw=0.9,
                                 shrinkA=0, shrinkB=0))

# ---------------- Fig: fused operator diagram (위/아래 2단) ----------------
def fig_fusion_diagram():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.2, 3.0))
    for ax in (ax1, ax2):
        ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    # (a) 비퓨전
    ax1.text(0.0, 0.92, "(a) Unfused: 3 kernels, 2 intermediate tensors", fontsize=9, ha="left")
    _box(ax1, 0.04, 0.25, 0.20, 0.5, "CONV_2D 5x5\n(stride 1)", "#dbe8ff", 8)
    _box(ax1, 0.36, 0.25, 0.20, 0.5, "AVG_POOL\n2x2 s2", "#dbe8ff", 8)
    _box(ax1, 0.68, 0.25, 0.20, 0.5, "LIF", "#dbe8ff", 8)
    _arrow(ax1, 0.24, 0.5, 0.36, 0.5); _arrow(ax1, 0.56, 0.5, 0.68, 0.5)
    ax1.text(0.30, 0.60, "64 KB", fontsize=7, ha="center", color="#555")
    ax1.text(0.62, 0.60, "16 KB", fontsize=7, ha="center", color="#555")
    ax1.text(0.30, 0.36, "int16\ntensor", fontsize=6.5, ha="center", color="#555")
    ax1.text(0.62, 0.36, "int16\ntensor", fontsize=6.5, ha="center", color="#555")
    # (b) 퓨전
    ax2.text(0.0, 0.94, "(b) Fused SNN_CONV_POOL_LIF (ours): 1 kernel, no intermediates", fontsize=9, ha="left")
    ax2.add_patch(FancyBboxPatch((0.02, 0.12), 0.85, 0.66, boxstyle="round,pad=0.008",
                                 fc="#ffe9cf", ec="black", lw=0.9, mutation_aspect=0.5))
    _box(ax2, 0.05, 0.22, 0.24, 0.42, "6x6 stride-2 conv\n= sum of four 5x5\nSMLALD (int64 acc)", "#fffaf2", 7)
    _box(ax2, 0.33, 0.22, 0.24, 0.42, "requantize\n(pool /4 folded\ninto scale)", "#fffaf2", 7)
    _box(ax2, 0.61, 0.22, 0.24, 0.42, "LIF epilogue\nQADD16/SSUB16/SEL\n(2-pixel tile)", "#fffaf2", 7)
    _arrow(ax2, 0.29, 0.43, 0.33, 0.43); _arrow(ax2, 0.57, 0.43, 0.61, 0.43)
    ax2.text(0.895, 0.45, "MACs/step\n15.6M -> 5.6M\n(2.8x)", fontsize=7.5, ha="left", va="center")
    fig.tight_layout(h_pad=0.5)
    fig.savefig(f"{OUT}/fig_fusion.pdf"); plt.close(fig)
    print("fig_fusion.pdf")

# ---------------- Fig: pipeline (위: 오프라인 / 아래: 온디바이스) ----------------
def fig_pipeline():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.2, 3.0))
    for ax in (ax1, ax2):
        ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax1.text(0.0, 0.92, "Offline (host) — CHACHA Compiler", fontsize=9, ha="left", fontweight="bold")
    _box(ax1, 0.02, 0.20, 0.27, 0.52, "snnTorch + Brevitas\nQAT training\n(CIFAR-10)", "#e6f4ea", 7.5)
    _box(ax1, 0.36, 0.20, 0.27, 0.52, "Lowering\nint8 W / Q15 act\nscale, bias64", "#e6f4ea", 7.5)
    _box(ax1, 0.70, 0.20, 0.28, 0.52, "Fusion pass\n6x6/s2 int16 weights\n+ .tflite flatbuffer", "#e6f4ea", 7.5)
    _arrow(ax1, 0.29, 0.46, 0.36, 0.46); _arrow(ax1, 0.63, 0.46, 0.70, 0.46)
    ax2.text(0.0, 0.92, "On-device (STM32F746G, Cortex-M7 @200 MHz)", fontsize=9, ha="left", fontweight="bold")
    _box(ax2, 0.02, 0.20, 0.20, 0.52, "Rate coding\nLUT + xorshift32", "#dbe8ff", 7)
    _box(ax2, 0.28, 0.20, 0.30, 0.52, "TFLM interpreter\narena planner\nfused custom ops x2", "#ffe9cf", 7)
    _box(ax2, 0.64, 0.20, 0.19, 0.52, "FC (CMSIS-NN)\n+ LIF out", "#dbe8ff", 7)
    _box(ax2, 0.88, 0.20, 0.10, 0.52, "vote\nT=30", "#eee", 7)
    _arrow(ax2, 0.22, 0.46, 0.28, 0.46); _arrow(ax2, 0.58, 0.46, 0.64, 0.46); _arrow(ax2, 0.83, 0.46, 0.88, 0.46)
    fig.tight_layout(h_pad=0.5)
    fig.savefig(f"{OUT}/fig_pipeline.pdf"); plt.close(fig)
    print("fig_pipeline.pdf")

# ---------------- Fig: deployability envelope ----------------
def fig_envelope():
    # 비용 모델: 측정된 V3 처리량으로 스케일 모델 추정
    nn_ms = V["V3_tflm_fused"]["nn_ms_per_image"]
    if nn_ms is None: print("envelope: skip"); return
    macs_fused = R["macs_per_timestep"]["conv1_fused"] + R["macs_per_timestep"]["conv2_fused"] + R["macs_per_timestep"]["fc"]
    ms_per_mac = (nn_ms / R["timesteps"]) / macs_fused  # 인코딩/LIF 포함 유효치
    configs = [
        ("LeNet-5 SNN (this work)", 32, 64, True),
        ("2$\\times$ channels", 64, 128, None),
        ("4$\\times$ channels", 128, 256, None),
        ("VGG-7 SNN 급", None, None, False),
    ]
    rows = []
    arena_used_kb = V["V3_tflm_fused"]["arena_used_bytes"] / 1000.0  # 80.3KB 실측
    for name, c1, c2 in [(c[0], c[1], c[2]) for c in configs[:3]]:
        # 층별 스케일링(십진 KB): conv1은 입력 3ch 고정, conv2는 c1×c2, FC int8
        w_flash = (c1*36*3*2 + c2*36*c1*2 + 10*(c2*64)) / 1000  # int16 fused + fc int8
        act_ram = arena_used_kb * (c1 / 32.0)  # 실측 80.3KB의 채널 비례 근사(추정)
        macs = (16*16*c1*36*3 + 8*8*c2*36*c1 + c2*64*10)
        lat = macs * ms_per_mac * R["timesteps"] / 1000
        rows.append((name, w_flash, act_ram, lat, macs))
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    names = [r[0] for r in rows]
    flash_kb = [r[1] for r in rows]; lat_s = [r[3] for r in rows]
    ax.scatter(flash_kb, lat_s, s=[70, 55, 55], c=["#e8710a", "#4c8bf5", "#4c8bf5"], zorder=3, edgecolor="black", linewidth=0.6)
    for (n, f, a, l, m) in rows:
        ax.annotate(n, (f, l), textcoords="offset points", xytext=(6, 4), fontsize=7)
    # 가중치 예산 = 1MB Flash(1,048,576B) - 코드/런타임(V3 전체 373,860B - 가중치계 195,328B) ≈ 870KB
    code_kb = (V["V3_tflm_fused"]["flash_bytes"] - 195328) / 1000.0
    budget_kb = (1048576 / 1000.0) - code_kb
    ax.axvline(budget_kb, color="red", ls="--", lw=0.9)
    ax.text(budget_kb, ax.get_ylim()[1]*0.75, f" Flash 한계\n (가중치 예산 ~{budget_kb:.0f}KB)", fontsize=7, color="red")
    ax.set_xlabel("Fused weight footprint (KB, Flash)")
    ax.set_ylabel("Est. latency / image (s, T=30)")
    ax.set_xscale("log"); ax.set_yscale("log")
    fig.savefig(f"{OUT}/fig_envelope.pdf"); plt.close(fig)
    print("fig_envelope.pdf")
    for r in rows:
        print(f"  {r[0]}: flash {r[1]:.1f}KB, act-ram {r[2]:.1f}KB(est), macs/step {r[4]:,}, est {r[3]:.2f}s/img ({r[3]*1000/R['timesteps']:.1f}ms/step)")

if __name__ == "__main__":
    fig_latency(); fig_timestep(); fig_memory(); fig_fusion_diagram(); fig_pipeline(); fig_envelope()
