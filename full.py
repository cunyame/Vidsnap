import os
import cv2
import numpy as np
import imagehash
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

# --- CONFIGURATION ---
VIDEO_PATH = r'C:\code\vidsnap\vlso.mp4'  # 🔁 Change this
TEMP_FOLDER = "filtered_slides"
OUTPUT_PDF = "slides_output.pdf"
FRAME_INTERVAL = 60  # Process every nth frame

# Thresholds
HASH_THRESHOLD = 75
SSIM_THRESHOLD = 0.95
ORB_MATCH_THRESHOLD = 20
RESIZE_DIM = (1280, 720)  # Smaller resolution for output
ENABLE_SSIM = False
ENABLE_ORB = False

# --- CLEAN SETUP ---
if not os.path.exists(TEMP_FOLDER):
    os.makedirs(TEMP_FOLDER)
else:
    for f in os.listdir(TEMP_FOLDER):
        os.remove(os.path.join(TEMP_FOLDER, f))

def are_images_similar_hash(img1, img2):
    hash1 = imagehash.phash(Image.fromarray(img1))
    hash2 = imagehash.phash(Image.fromarray(img2))
    return abs(hash1 - hash2) <= HASH_THRESHOLD

def are_images_similar_ssim(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray1, gray2, full=True)
    return score >= SSIM_THRESHOLD

def are_images_similar_orb(img1, img2):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), None)
    kp2, des2 = orb.detectAndCompute(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), None)
    if des1 is None or des2 is None:
        return False
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    return len(matches) >= ORB_MATCH_THRESHOLD

def add_timestamp(image, frame_number, fps):
    timestamp_seconds = frame_number // fps
    hrs = timestamp_seconds // 3600
    mins = (timestamp_seconds % 3600) // 60
    secs = timestamp_seconds % 60
    timestamp = f"{hrs:02}:{mins:02}:{secs:02}"

    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype("arial.ttf", 36) if os.name == "nt" else ImageFont.load_default()
    draw.text((30, 30), timestamp, fill="white", font=font, stroke_width=2, stroke_fill="black")
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# --- PROCESS VIDEO ---
cap = cv2.VideoCapture(VIDEO_PATH)
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

prev_saved_images = []
frame_idx = 0
saved_idx = 0

print("🔍 Extracting and filtering unique slides...")

for i in tqdm(range(0, total_frames, FRAME_INTERVAL)):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, frame = cap.read()
    if not ret:
        break

    frame_resized = cv2.resize(frame, RESIZE_DIM)
    frame_timestamped = add_timestamp(frame_resized.copy(), i, fps)

    is_duplicate = False
    for past in prev_saved_images:
        if are_images_similar_hash(past, frame_resized):
            is_duplicate = True
            break
        if ENABLE_SSIM and are_images_similar_ssim(past, frame_resized):
            is_duplicate = True
            break
        if ENABLE_ORB and are_images_similar_orb(past, frame_resized):
            is_duplicate = True
            break

    if not is_duplicate:
        save_path = os.path.join(TEMP_FOLDER, f"slide_{saved_idx:03}.jpg")
        cv2.imwrite(save_path, frame_timestamped, [cv2.IMWRITE_JPEG_QUALITY, 85])
        prev_saved_images.append(frame_resized)
        saved_idx += 1

cap.release()

# --- CREATE PDF ---
print("🖨️ Generating PDF...")
images = sorted([img for img in os.listdir(TEMP_FOLDER) if img.endswith(".jpg")])
if not images:
    print("❌ No slides saved. PDF not generated.")
    exit()

pdf_images = []
for img_name in images:
    img_path = os.path.join(TEMP_FOLDER, img_name)
    img = Image.open(img_path).convert("RGB")
    pdf_images.append(img)

pdf_images[0].save(OUTPUT_PDF, save_all=True, append_images=pdf_images[1:])
print(f"✅ PDF saved as: {OUTPUT_PDF}")
