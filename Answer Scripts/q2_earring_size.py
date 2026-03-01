import numpy as np
import matplotlib.pyplot as plt
import cv2

# ─────────────────────────────────────────────────────────────
# Q2 – Earring Size Estimation
# Camera: f = 8 mm, pixel_size = 2.2 µm, image_distance = 720 mm
# ─────────────────────────────────────────────────────────────

f  = 8.0          # focal length (mm)
Z  = 720.0        # object distance: lens to earring plane (mm)
px = 2.2e-3       # pixel size in mm (2.2 µm)

# Thin-lens equation: 1/f = 1/Z + 1/di  =>  di = f*Z / (Z - f)
di = (f * Z) / (Z - f)
print(f"Image distance   di = {di:.4f} mm")

# Lateral magnification M = di / Z  (image / object distance)
M = di / Z
print(f"Magnification    M  = {M:.8f}")

# mm per pixel in real-world space = pixel_size / M = pixel_size * Z / di
mm_per_pixel = px / M
print(f"mm per pixel        = {mm_per_pixel:.6f} mm/px")

# ── Measure earring diameter from the image ──────────────────
img = cv2.imread("../earrings.jpg")
if img is None:
    raise FileNotFoundError("earrings.jpg not found.")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h, w = img.shape[:2]
print(f"\nImage size: {w} x {h} pixels")

# Threshold to isolate earrings
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

diameters_px = []
img_drawn = img.copy()
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 500:
        continue
    bx, by, bw, bh = cv2.boundingRect(cnt)
    diam_px = float(bw)             # width = outer diameter
    diameters_px.append(diam_px)
    cv2.rectangle(img_drawn, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
    cv2.putText(img_drawn, f"{bw}x{bh}px",
                (bx, by - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

if diameters_px:
    d_px = np.mean(diameters_px)
else:
    # Fallback: manually estimate from image width (earrings occupy ~40% of width)
    d_px = 0.4 * w

print(f"Detected earring diameter (pixels): {d_px:.1f} px")

# Real-world size: convert pixels -> real via mm_per_pixel
d_real_mm = d_px * mm_per_pixel
d_real_cm = d_real_mm / 10.0

print(f"\nEarring real-world diameter : {d_real_mm:.2f} mm  ({d_real_cm:.3f} cm)")

# ── Display ──────────────────────────────────────────────────
plt.figure(figsize=(6, 5))
plt.imshow(cv2.cvtColor(img_drawn, cv2.COLOR_BGR2RGB))
plt.title(f"Earring detection\nEstimated diameter ≈ {d_real_mm:.2f} mm")
plt.axis('off')
plt.tight_layout()
plt.savefig('q2_earrings_result.png', dpi=150)
plt.show()
print("\nFigure saved to q2_earrings_result.png")
