import cv2
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity as ssim
import imagehash
from datetime import timedelta

# --- CONFIGURATION ---
VIDEO_PATH = r"C:\code\vidsnap\vlso.mp4"
TEMP_FOLDER = "filtered_slides"
OUTPUT_PDF = "slides_output.pdf"
FRAME_INTERVAL = 30  # Tune this to control frequency
RESIZE_WIDTH = 960   # Resize to save space
HASH_SIZE = 7       # Finer image hashing
SSIM_THRESHOLD = 0.96
ORB_THRESHOLD = 40

# --- INITIALIZE ---
os.makedirs(TEMP_FOLDER, exist_ok=True)
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("❌ Could not open video.")
    exit()

saved_hashes = set()
prev_frame = None
prev_orb_desc = None
saved_images = []

orb = cv2.ORB_create()

frame_idx = 0
slide_count = 0

def get_timestamp_string(seconds):
    return str(timedelta(seconds=int(seconds)))

def compute_ssim(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray1, gray2, full=True)
    return score

def orb_match(img1, img2):
    kp1, des1 = orb.detectAndCompute(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), None)
    kp2, des2 = orb.detectAndCompute(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), None)
    if des1 is None or des2 is None:
        return 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    return len(matches)

def image_to_hash(img):
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return str(imagehash.average_hash(pil_img, hash_size=HASH_SIZE))

def save_frame_with_timestamp(img, timestamp_str, idx):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.load_default()
    draw.text((10, 10), timestamp_str, font=font, fill=(255, 255, 255))
    img_pil = img_pil.resize((RESIZE_WIDTH, int(img_pil.height * RESIZE_WIDTH / img_pil.width)))
    filename = os.path.join(TEMP_FOLDER, f"slide_{idx:03d}.jpg")
    img_pil.save(filename)
    return filename

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % FRAME_INTERVAL == 0:
        timestamp_str = get_timestamp_string(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)

        hash_val = image_to_hash(frame)
        if hash_val in saved_hashes:
            frame_idx += 1
            continue

        is_unique = True
        if prev_frame is not None:
            ssim_score = compute_ssim(frame, prev_frame)
            orb_matches = orb_match(frame, prev_frame)

            if ssim_score > SSIM_THRESHOLD or orb_matches > ORB_THRESHOLD:
                is_unique = False

        if is_unique:
            saved_hashes.add(hash_val)
            prev_frame = frame.copy()
            saved_file = save_frame_with_timestamp(frame, timestamp_str, slide_count)
            saved_images.append(saved_file)
            slide_count += 1

    frame_idx += 1

cap.release()

# --- PDF EXPORT ---
if saved_images:
    pdf_images = []
    for img_path in saved_images:
        img = Image.open(img_path).convert("RGB")
        pdf_images.append(img.copy())
    pdf_images[0].save(os.path.join(TEMP_FOLDER, OUTPUT_PDF), save_all=True, append_images=pdf_images[1:])
    print(f"✅ PDF saved at: {os.path.join(TEMP_FOLDER, OUTPUT_PDF)}")
else:
    print("❌ No unique slides saved.")
