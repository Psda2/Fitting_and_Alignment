import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────
# Q3 – Superimpose Sri Lanka flag on the cricket pitch
# Student A uses the Sri Lanka flag
# ─────────────────────────────────────────────────────────────

points = []
img_display = None

def mouse_callback(event, x, y, flags, param):
    global points, img_display
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
            print(f"Point {len(points)}: ({x}, {y})")
            cv2.circle(img_display, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Image", img_display)
        if len(points) == 4:
            print("\nFour points selected:")
            for i, p in enumerate(points):
                print(f"P{i+1}: {p}")
            print("Press any key to exit.")

img = cv2.imread("../turf.jpg")
if img is None:
    raise FileNotFoundError("Image not found.")

img_display = img.copy()

cv2.namedWindow("Image")
cv2.setMouseCallback("Image", mouse_callback)

cv2.imshow("Image", img_display)
cv2.waitKey(0)
cv2.destroyAllWindows()

points = np.array(points, dtype=np.float32)

print("\nFinal array of selected points:")
print(points)

# ─────────────────────────────────────────────────────────────
# Load Sri Lanka flag image from file
# ─────────────────────────────────────────────────────────────
flag = cv2.imread("sri_lanka_flag.png")
if flag is None:
    raise FileNotFoundError("Flag image 'sri_lanka_flag.png' not found. "
                            "Please make sure it is in the Student_A folder.")

flag_h, flag_w = flag.shape[:2]

# ─────────────────────────────────────────────────────────────
# Warp Sri Lanka flag onto the selected quadrangle on the turf
# ─────────────────────────────────────────────────────────────
src_pts = np.array([
    [0, 0],
    [flag_w - 1, 0],
    [flag_w - 1, flag_h - 1],
    [0, flag_h - 1]
], dtype=np.float32)

dst_pts = points

H, _ = cv2.findHomography(src_pts, dst_pts)

turf = cv2.imread("../turf.jpg")
warped = cv2.warpPerspective(flag, H, (turf.shape[1], turf.shape[0]))

mask = np.zeros(turf.shape[:2], dtype=np.uint8)
cv2.fillConvexPoly(mask, dst_pts.astype(np.int32), 255)

# ── Opacity blend: flag at alpha, turf shows through ─────────
alpha = 0.55          # flag opacity  (0.0 = invisible, 1.0 = opaque)
beta  = 1.0 - alpha   # turf contribution inside the flag region

# Blend only inside the masked region
blended = cv2.addWeighted(warped, alpha, turf, beta, 0)

# Outside mask: keep original turf
mask_3ch = cv2.merge([mask, mask, mask])
result = np.where(mask_3ch > 0, blended, turf)

cv2.imwrite("q3_result.jpg", result)
print("\nResult saved to q3_result.jpg")
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
