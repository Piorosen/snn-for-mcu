#!/usr/bin/env python3
"""paper_sparse 그림 생성 (실측 데이터: info/results/snn_bench_results.json).

색상: Okabe-Ito 부분집합 (CVD 검증 통과: #0072B2,#E69F00,#009E73,#CC79A7).
CVD ΔE 7.6(6-8 밴드)·명도 대비 경고는 2차 부호화로 충족한다 — 세그먼트 간
흰 간격, 해칭(흑백 인쇄 대비), 직접 라벨, 본문 표에 정확값 병기.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": ["Apple SD Gothic Neo", "AppleGothic"],
    "axes.unicode_minus": False,
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

BLUE, ORANGE, GREEN, PINK = "#0072B2", "#E69F00", "#009E73", "#CC79A7"

# ---------------- 그림 1: 누적 최적화 단계별 지연시간 ----------------
variants = ["A0\n(기존 재현)", "A1 +ART", "A2 +ITCM", "A3 +DTCM", "A4 +이벤트 커널\n(제안)"]
totals = [2328.5, 2328.9, 2331.2, 2178.7, 380.6]
ratios = ["1.00×", "1.00×", "1.00×", "1.07×", "6.12×"]

fig, ax = plt.subplots(figsize=(3.5, 2.3))
y = range(len(variants))
bars = ax.barh(y, totals, height=0.62, color=BLUE, edgecolor="white", linewidth=0.5)
for i, (v, r) in enumerate(zip(totals, ratios)):
    ax.text(v + 40, i, f"{v:,.1f} ms  ({r})", va="center", fontsize=8)
ax.set_yticks(y, variants, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("이미지당 지연시간 (ms), $T=30$")
ax.set_xlim(0, 3350)
ax.xaxis.grid(True, color="0.9", linewidth=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("fig_ablation.pdf")
plt.close(fig)

# ---------------- 그림 2: 레이어 분해 (스택 막대) ----------------
rows = ["A0 (dense,\n기본 배치)", "A3 (dense,\n+DTCM)", "A4 (이벤트 커널,\n제안)"]
enc = [35.4, 37.1, 5.9]
cpl1 = [446.0, 401.8, 222.1]
cpl2 = [1828.0, 1719.8, 132.2]
rest = [t - a - b - c for t, a, b, c in
        zip([2328.5, 2178.7, 380.6], enc, cpl1, cpl2)]

segs = [
    ("인코딩", enc, ORANGE, "//"),
    ("CPL1", cpl1, GREEN, None),
    ("CPL2", cpl2, BLUE, None),
    ("FC+런타임 등", rest, PINK, "xx"),
]
fig, ax = plt.subplots(figsize=(3.5, 2.0))
y = range(len(rows))
left = [0.0] * len(rows)
for name, vals, color, hatch in segs:
    ax.barh(y, vals, left=left, height=0.6, color=color, hatch=hatch,
            edgecolor="white", linewidth=1.2, label=name)
    for i, (l, v) in enumerate(zip(left, vals)):
        if v > 180:  # 큰 세그먼트만 직접 라벨
            ax.text(l + v / 2, i, f"{v:,.0f}", va="center", ha="center",
                    fontsize=7.5, color="white", fontweight="bold")
    left = [l + v for l, v in zip(left, vals)]
ax.set_yticks(y, rows, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("이미지당 시간 (ms)")
ax.legend(fontsize=7, ncol=4, loc="lower right", bbox_to_anchor=(1.0, 1.0),
          frameon=False, handlelength=1.2, columnspacing=0.9)
ax.xaxis.grid(True, color="0.9", linewidth=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("fig_layers.pdf")
plt.close(fig)

# ---------------- 그림 3: 희소 커널 반복 개선 ----------------
steps = ["A3 dense\n(+DTCM)", "k1\n인덱스 목록", "k2\nim2col 제거",
         "k3\n가중치 전치", "k4\n8ch 블로킹", "k5\n최종"]
tot = [2178.7, 579.2, 504.3, 457.5, 382.2, 380.6]
c1 = [401.8, 253.9, 228.7, 282.8, 223.2, 222.1]
c2 = [1719.8, 298.8, 249.5, 148.4, 132.9, 132.2]

fig, ax = plt.subplots(figsize=(3.5, 2.4))
x = range(len(steps))
ax.plot(x, tot, marker="o", ms=5, lw=2, color=BLUE, label="전체")
ax.plot(x, c2, marker="s", ms=5, lw=2, color=GREEN, label="CPL2")
ax.plot(x, c1, marker="^", ms=5, lw=2, color=ORANGE, label="CPL1")
ax.set_yscale("log")
ax.set_yticks([100, 200, 400, 800, 1600, 3200],
              ["100", "200", "400", "800", "1,600", "3,200"])
ax.set_ylabel("이미지당 시간 (ms, 로그)")
ax.set_xticks(x, steps, fontsize=7)
ax.annotate("k3: CPL1 역효과", xy=(3, 282.8), xytext=(3.05, 700),
            fontsize=7, ha="center",
            arrowprops=dict(arrowstyle="-", color="0.5", lw=0.8))
ax.legend(fontsize=7.5, frameon=False, loc="upper right")
ax.yaxis.grid(True, color="0.9", linewidth=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("fig_kernel_iter.pdf")
plt.close(fig)

# ---------------- 그림 4: 개념도 (조밀 경로 vs 이벤트 경로) ----------------
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

GRAY = "#6f6f6f"

def box(ax, x, y, w, h, text, fc, ec, fs=6.8, tc="white", lw=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, linespacing=1.25)

def arrow(ax, x0, y0, x1, y1, color="0.45"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=8, color=color, lw=1.0))

fig, ax = plt.subplots(figsize=(3.6, 2.05))
ax.set_xlim(0, 10)
ax.set_ylim(0, 5.6)
ax.axis("off")

ax.text(0.1, 5.25, "조밀 경로 (기존)", fontsize=7.5, color=GRAY, fontweight="bold")
box(ax, 0.1, 3.6, 2.4, 1.25, "이진 스파이크\n텐서 (96% 0)", "#ffffff", GRAY, tc=GRAY)
box(ax, 3.1, 3.6, 2.4, 1.25, "im2col 값 실체화\n147KB/스텝", "#ffffff", GRAY, tc=GRAY)
box(ax, 6.1, 3.6, 2.4, 1.25, "dense MAC\n$\\Sigma\\,w\\cdot x$", "#ffffff", GRAY, tc=GRAY)
arrow(ax, 2.5, 4.22, 3.1, 4.22)
arrow(ax, 5.5, 4.22, 6.1, 4.22)

ax.text(0.1, 2.6, "이벤트 경로 (제안)", fontsize=7.5, color=BLUE, fontweight="bold")
box(ax, 0.1, 0.95, 2.4, 1.25, "발화 위치 목록\n($\\rho r\\approx$47개/픽셀)", BLUE, BLUE)
box(ax, 3.1, 0.95, 2.4, 1.25, "가중치 gather 합산\n$S\\cdot\\Sigma\\,w$ (곱셈 0회)", BLUE, BLUE)
arrow(ax, 2.5, 1.57, 3.1, 1.57, color=BLUE)
arrow(ax, 5.5, 1.57, 6.55, 2.35, color=BLUE)

box(ax, 6.1, 2.15, 2.4, 1.25, "공유 에필로그\nrequantize·LIF", GREEN, GREEN)
arrow(ax, 7.3, 3.6, 7.3, 3.42)
ax.text(9.35, 2.78, "출력\n비트\n동일", ha="center", va="center", fontsize=7,
        color=GREEN, fontweight="bold")
arrow(ax, 8.5, 2.78, 8.85, 2.78, color=GREEN)

ax.text(3.1, 0.35, "사라지는 것: im2col · 곱셈 · 0-피연산자 로드", fontsize=6.8,
        color=BLUE, style="italic")
fig.tight_layout(pad=0.3)
fig.savefig("fig_concept.pdf")
plt.close(fig)

# ---------------- 그림 5: 비용 모델 (시간비 대 밀도) ----------------
ALPHA, BETA = 0.0321, 1.0915
rho_pts = [0.041, 0.477]
ratio_pts = [132.2 / 1719.8, 222.1 / 401.8]

fig, ax = plt.subplots(figsize=(3.3, 2.15))
xs = [i / 100 for i in range(0, 101)]
ax.plot(xs, [ALPHA + BETA * x for x in xs], color=BLUE, lw=2,
        label=r"모델 $\alpha+\beta\rho$")
ax.scatter(rho_pts, ratio_pts, s=42, color=BLUE, zorder=5,
           edgecolor="white", linewidth=1.2)
ax.annotate("CPL2 (4.1%)", xy=(rho_pts[0], ratio_pts[0]),
            xytext=(0.09, 0.02), fontsize=7.5)
ax.annotate("CPL1 (47.7%)", xy=(rho_pts[1], ratio_pts[1]),
            xytext=(0.5, 0.45), fontsize=7.5)
ax.axhline(1.0, color="0.55", lw=1.0, ls="--")
ax.text(0.02, 1.03, "조밀 경로 (=1)", fontsize=7, color="0.4")
rho_star = (1 - ALPHA) / BETA
ax.axvline(rho_star, color=ORANGE, lw=1.2, ls=":")
ax.annotate(r"손익분기 $\rho^{*}\approx$0.89", xy=(rho_star, 0.62),
            xytext=(0.44, 0.72), fontsize=7.5, color=ORANGE,
            arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.8))
ax.annotate(r"바닥 $\alpha\approx$0.032 (상한 ~31$\times$)", xy=(0.0, ALPHA),
            xytext=(0.03, 0.19), fontsize=7.5,
            arrowprops=dict(arrowstyle="-", color="0.5", lw=0.8))
ax.set_xlabel(r"활성 밀도 $\rho$")
ax.set_ylabel(r"$T_{\mathrm{sparse}}/T_{\mathrm{dense}}$")
ax.set_xlim(0, 1.0)
ax.set_ylim(0, 1.18)
ax.yaxis.grid(True, color="0.9", linewidth=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("fig_costmodel.pdf")
plt.close(fig)

print("OK: fig_ablation.pdf, fig_layers.pdf, fig_kernel_iter.pdf, fig_concept.pdf, fig_costmodel.pdf")
