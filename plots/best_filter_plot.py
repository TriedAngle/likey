import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

# ----------------------------------------------------------------------
# Style — serif font + Computer-Modern math, like the paper
# ----------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 15,
    "axes.linewidth": 1.2,
})

# ----------------------------------------------------------------------
# 1. Categories and their colours (index -> filter)
# ----------------------------------------------------------------------
labels = ["Bloom", "Cuckoo", "Morton", "Xor"]
colors = ["#2b5d8a", "#e8923a", "#f3c63d", "#c0392b"]   # blue, orange, yellow, red
cmap   = ListedColormap(colors)
cmap.set_bad("white")                                   # NaN cells -> white
norm   = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

# ----------------------------------------------------------------------
# 2. Build a grid that resembles the figure.
#    grid[r, c]: 0=Bloom 1=Cuckoo 2=Morton 3=Xor  np.nan=infeasible
#    row 0 = top  (eps = 1e-1), increasing row -> lower eps
#    col 0 = left (small m),    increasing col -> larger m
# ----------------------------------------------------------------------
ncols, nrows = 22, 17
m_vals  = np.linspace(60, 315, ncols)            # filter size [MB]
eps_exp = np.linspace(-1, -6.5, nrows)           # log10(false-positive rate)

grid = np.full((nrows, ncols), np.nan)
for r in range(nrows):
    min_c = r * 0.9                              # left feasibility boundary (staircase)
    for c in range(ncols):
        if c < min_c:
            continue                             # infeasible -> stays NaN (white)
        # default winner
        grid[r, c] = 0                           # Bloom
        # red ridge right along the feasibility boundary
        if c < min_c + 2:
            grid[r, c] = 3                       # Xor
        # orange/yellow pocket at low eps, mid-large m
        if 7 <= r <= 13 and (min_c + 2) <= c <= (min_c + 7):
            grid[r, c] = 1 if (c + r) % 3 else 2  # Cuckoo / Morton mix
# very bottom tip is Xor
grid[-1, -1] = 3

# ----------------------------------------------------------------------
# 3. Draw with pcolormesh on integer edges -> uniform square cells
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 6))
masked = np.ma.masked_invalid(grid)
ax.pcolormesh(np.arange(ncols + 1), np.arange(nrows + 1), masked,
              cmap=cmap, norm=norm, edgecolors="white", linewidth=2.5)

ax.set_aspect("equal")
ax.invert_yaxis()                                # row 0 (high eps) on top

# ----- custom tick labels: show eps and m, not cell indices -----
ytick_exp = [-1, -2, -3, -4, -5, -6]
ax.set_yticks([np.interp(e, eps_exp[::-1], np.arange(nrows)[::-1]) + 0.5
               for e in ytick_exp])
ax.set_yticklabels([rf"$10^{{{e}}}$" for e in ytick_exp])

xtick_m = [100, 150, 200, 250, 300]
ax.set_xticks([np.interp(m, m_vals, np.arange(ncols)) + 0.5 for m in xtick_m])
ax.set_xticklabels([str(m) for m in xtick_m])

ax.set_xlabel(r"Filter size $m$ [MB]")
ax.set_ylabel(r"False-positive rate $\varepsilon$")
ax.set_title("Best Performing Filter", fontsize=18, pad=12)

# ----------------------------------------------------------------------
# 4. Highlight rectangle (around m ~ 285-295 MB column)
# ----------------------------------------------------------------------
hl_col = int(np.interp(289, m_vals, np.arange(ncols)))
ax.add_patch(Rectangle((hl_col, 4), 1, nrows - 4,
                       fill=False, edgecolor="black", linewidth=2.5, zorder=5))

# ----------------------------------------------------------------------
# 5. Legend (colour swatches) placed inside the empty lower-left
# ----------------------------------------------------------------------
handles = [Line2D([0], [0], marker="s", linestyle="", markersize=14,
                  markerfacecolor=col, markeredgecolor="none", label=lab)
           for col, lab in zip(colors, labels)]
ax.legend(handles=handles, loc="lower left", frameon=False,
          handletextpad=0.4, bbox_to_anchor=(0.02, 0.02), fontsize=14)

plt.tight_layout()
plt.savefig("plots/outputs/best_filter_plot.png", dpi=150, bbox_inches="tight")
print("saved")
