import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image

# --- Settings ---
INPUT_FOLDER = "data"       # Folder with .jpg images
TEMP_FOLDER = "filtered_slides"     # Folder to save changed slides
OUTPUT_PDF = "slides_output.pdf"    # Final PDF name

SSIM_THRESHOLD = 0.79
ORB_MATCH_THRESHOLD = 20
RESIZE_SHAPE = (300, 300)

# --- Ensure temp folder exists ---
os.makedirs(TEMP_FOLDER, exist_ok=True)

# --- Functions ---
def compute_ssim(img1, img2):
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    g1 = cv2.resize(g1, RESIZE_SHAPE)
    g2 = cv2.resize(g2, RESIZE_SHAPE)
    g1 = cv2.GaussianBlur(g1, (5, 5), 0)
    g2 = cv2.GaussianBlur(g2, (5, 5), 0)
    score, _ = ssim(g1, g2, full=True)
    return score

def compute_orb_diff(img1, img2):
    orb = cv2.ORB_create(1000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        return 0.0  # No descriptors

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    total = len(matches)
    match_ratio = len(good) / total if total > 0 else 0.0
    return match_ratio

def is_different(img1, img2):
    ssim_score = compute_ssim(img1, img2)
    orb_ratio = compute_orb_diff(img1, img2)
    print(f"SSIM: {ssim_score:.4f}, ORB Match Ratio: {orb_ratio:.3f}")

    return ssim_score < SSIM_THRESHOLD or orb_ratio < 0.5


# --- Main Logic ---
image_files = sorted([f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(".jpg")])
saved_slides = []
last_img = None

for i, file in enumerate(image_files):
    path = os.path.join(INPUT_FOLDER, file)
    img = cv2.imread(path)

    if last_img is None or is_different(last_img, img):
        save_path = os.path.join(TEMP_FOLDER, f"slide_{len(saved_slides):03d}.jpg")
        cv2.imwrite(save_path, img)
        saved_slides.append(save_path)
        last_img = img
        print(f"Saved: {save_path}")
    else:
        print(f"Skipped: {file}")

# --- Convert to PDF ---
images_for_pdf = [Image.open(p).convert("RGB") for p in saved_slides]
if images_for_pdf:
    images_for_pdf[0].save(OUTPUT_PDF, save_all=True, append_images=images_for_pdf[1:])
    print(f"\n✅ PDF generated: {OUTPUT_PDF}")
else:
    print("\n⚠️ No slides saved. PDF not generated.")
