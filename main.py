import os
import cv2
import numpy as np

# ---------------- CONFIG ----------------
INPUT_DIR = "data/images"
OUTPUT_DIR = "data/results"

DISTANCE_M = 4.0          # meters (distance from camera to window)
FOCAL_LENGTH_MM = 26.0    # mm
SENSOR_WIDTH_MM = 36.0   # full-frame sensor width in mm

os.makedirs(OUTPUT_DIR, exist_ok=True)
# ---------------------------------------


def detect_window_wall_border(img, debug=False):
    """
    Detect the outer wall border of the window for precise measurement
    """
    h_img, w_img = img.shape[:2]

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 1: Adaptive threshold to separate bright window
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 51, -10
    )

    # Invert so window is white
    thresh = 255 - thresh

    # Step 2: Morphology to clean noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel, iterations=1)

    # Step 3: Find contours
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Step 4: Largest contour = window
    window_cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(window_cnt)

    # Step 5: Expand ROI for wall border detection
    pad_x = int(0.6 * w)
    pad_y = int(0.6 * h)
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(w_img, x + w + pad_x)
    y1 = min(h_img, y + h + pad_y)
    roi = img[y0:y1, x0:x1]

    # Step 6: Edge detection
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)
    edges = cv2.Canny(gray_roi, 50, 150)

    # Step 7: Morphology to close edges
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel2, iterations=2)

    # Step 8: Contours on edges
    contours2, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours2:
        # fallback: use original window contour if edges fail
        return window_cnt

    # Step 9: Filter contours by area
    min_area = 0.1 * roi.shape[0] * roi.shape[1]  # skip small noise
    large_contours = [c for c in contours2 if cv2.contourArea(c) > min_area]
    if not large_contours:
        return window_cnt  # fallback

    # Step 10: Combine all large contours to get full wall border
    all_points = np.vstack(large_contours)
    hull = cv2.convexHull(all_points)
    hull[:, 0, 0] += x0
    hull[:, 0, 1] += y0

    if debug:
        debug_img = img.copy()
        cv2.drawContours(debug_img, [hull], 0, (0, 255, 255), 2)
        cv2.imshow("Wall Border Detection", debug_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return hull


def compute_real_dimensions(box, img_width_px):
    """
    Compute real-world width and height (meters) from pixel coordinates using
    baseline distance and focal length.
    """
    if box is None or len(box) == 0:
        return 0, 0

    x, y, w_px, h_px = cv2.boundingRect(box)

    if w_px == 0 or h_px == 0:
        return 0, 0

    # focal length in pixels
    focal_px = (FOCAL_LENGTH_MM / SENSOR_WIDTH_MM) * img_width_px

    # real dimensions using pinhole camera model
    real_w_m = (w_px * DISTANCE_M) / focal_px
    real_h_m = (h_px * DISTANCE_M) / focal_px

    return real_w_m, real_h_m


def process_image(img_path, debug=False):
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Could not read image: {img_path}")
        return

    box = detect_window_wall_border(img, debug=debug)
    if box is None:
        print(f"❌ Window not detected in {img_path}")
        return

    real_w, real_h = compute_real_dimensions(box, img.shape[1])

    if real_w == 0 or real_h == 0:
        print(f"❌ Failed to compute dimensions for {img_path}")
        return

    # Draw the precise wall border
    cv2.drawContours(img, [box], 0, (0, 255, 255), 2)

    # Put measurements on the image (inside box if possible)
    x, y, w, h = cv2.boundingRect(box)
    text_w = f"W: {real_w*100:.1f} cm"
    text_h = f"H: {real_h*100:.1f} cm"

    # Adjust text position to stay inside the image
    y_w = max(y - 10, 20)
    y_h = max(y_w + 25, 40)

    cv2.putText(img, text_w,
                (x + 5, y_w),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(img, text_h,
                (x + 5, y_h),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Save output image
    out_path = os.path.join(OUTPUT_DIR, os.path.basename(img_path))
    cv2.imwrite(out_path, img)

    print(f"✅ Processed: {img_path}")
    print(f"📏 Width : {real_w:.2f} m")
    print(f"📏 Height: {real_h:.2f} m")
    print(f"📁 Saved : {out_path}")


def main():
    print("📐 Window Measurement (Batch Processing)")

    images = sorted([
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ])

    if not images:
        print("❌ No images found")
        return

    for img_file in images:
        img_path = os.path.join(INPUT_DIR, img_file)
        process_image(img_path, debug=False)


if __name__ == "__main__":
    main()
