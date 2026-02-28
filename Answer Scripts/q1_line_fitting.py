import numpy as np
import matplotlib.pyplot as plt

D = np.genfromtxt("../lines.csv", delimiter=",", skip_header=1)
X_cols = D[:, :3]
Y_cols = D[:, 3:]
X_all = X_cols.flatten()
Y_all = Y_cols.flatten()

# ─────────────────────────────────────────────────────────────
# Part (a) – Total Least Squares on Line 1 only
# ─────────────────────────────────────────────────────────────
x1 = X_cols[:, 0]
y1 = Y_cols[:, 0]

# Centre the data
x1_mean = np.mean(x1)
y1_mean = np.mean(y1)
A = np.column_stack([x1 - x1_mean, y1 - y1_mean])

# SVD – the right singular vector for the smallest singular value
# gives the line normal [a, b] and passes through the centroid
U, S, Vt = np.linalg.svd(A)
normal = Vt[-1]          # [a, b]
a, b = normal
c = -(a * x1_mean + b * y1_mean)

print("Part (a) – TLS result for Line 1")
print(f"  Normal vector  : a={a:.6f}, b={b:.6f}")
print(f"  Offset         : c={c:.6f}")
print(f"  Line equation  : {a:.4f}*x + {b:.4f}*y + {c:.4f} = 0")
if abs(b) > 1e-9:
    slope = -a / b
    intercept = -c / b
    print(f"  Slope form     : y = {slope:.4f}*x + {intercept:.4f}")

# ─────────────────────────────────────────────────────────────
# Part (b) – RANSAC to recover all three lines
# ─────────────────────────────────────────────────────────────

def fit_line_tls(x, y):
    mx, my = np.mean(x), np.mean(y)
    A = np.column_stack([x - mx, y - my])
    _, _, Vt = np.linalg.svd(A)
    n = Vt[-1]
    return n[0], n[1], -(n[0]*mx + n[1]*my)

def point_line_dist(x, y, a, b, c):
    return np.abs(a*x + b*y + c) / np.sqrt(a**2 + b**2)

def ransac_line(x, y, n_iter=2000, threshold=0.3):
    best_inliers = []
    rng = np.random.default_rng(seed=42)
    n = len(x)
    for _ in range(n_iter):
        idx = rng.choice(n, 2, replace=False)
        a, b, c = fit_line_tls(x[idx], y[idx])
        dists = point_line_dist(x, y, a, b, c)
        inliers = np.where(dists < threshold)[0]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
    a, b, c = fit_line_tls(x[best_inliers], y[best_inliers])
    return a, b, c, best_inliers

print("\nPart (b) – RANSAC line detection")
remaining = np.ones(len(X_all), dtype=bool)
lines = []

for i in range(3):
    x_r = X_all[remaining]
    y_r = Y_all[remaining]
    a, b, c, local_inliers = ransac_line(x_r, y_r)
    lines.append((a, b, c))
    global_indices = np.where(remaining)[0][local_inliers]
    remaining[global_indices] = False
    print(f"  Line {i+1}: {a:.4f}*x + {b:.4f}*y + {c:.4f} = 0  "
          f"({len(local_inliers)} inliers)")

# ─────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────
colors = ['tab:blue', 'tab:orange', 'tab:green']
labels_data = ['Line 1 data', 'Line 2 data', 'Line 3 data']
xs = np.linspace(X_all.min(), X_all.max(), 300)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: TLS on line 1
axes[0].scatter(x1, y1, s=10, color='tab:blue', label='Line 1 data')
if abs(b) > 1e-9:
    slope = -a_tls if False else -lines[0][0]/lines[0][1]
    a0, b0, c0 = fit_line_tls(x1, y1)
    ys_tls = -(a0*xs + c0) / b0
    axes[0].plot(xs, ys_tls, 'r-', linewidth=2, label='TLS fit')
axes[0].set_title('Part (a): TLS – Line 1')
axes[0].set_xlabel('x'); axes[0].set_ylabel('y')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

# Right: RANSAC – all three lines
for i, col in enumerate(colors):
    xi = X_cols[:, i]; yi = Y_cols[:, i]
    axes[1].scatter(xi, yi, s=10, color=col, label=labels_data[i])

for i, (a, b, c) in enumerate(lines):
    if abs(b) > 1e-9:
        ys = -(a*xs + c) / b
        axes[1].plot(xs, ys, color=colors[i], linewidth=2.5,
                     linestyle='--', label=f'RANSAC line {i+1}')

axes[1].set_title('Part (b): RANSAC – Three Lines')
axes[1].set_xlabel('x'); axes[1].set_ylabel('y')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('q1_result.png', dpi=150)
plt.show()
print("\nFigure saved to q1_result.png")
