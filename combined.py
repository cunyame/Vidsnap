import cv2
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image, ImageDraw, ImageFont
#ffmpeg -i input_video.webm -c:v libx264 -c:a aac output_video.mp4


# --- CONFIGURATION ---
VIDEO_PATH = "vlso.mp4"  # Change this to your video file
TEMP_FOLDER = "filtered_slides"
OUTPUT_PDF = "slides_output.pdf"
FRAME_INTERVAL = 60  # Check every nth frame

SSIM_THRESHOLD = 0.75
ORB_MATCH_THRESHOLD = 0.2
RESIZE_SHAPE = (300, 300)

# --- Create folder for filtered slides ---
os.makedirs(TEMP_FOLDER, exist_ok=True)

# --- SSIM calculation ---
def compute_ssim(img1, img2):
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    g1 = cv2.resize(g1, RESIZE_SHAPE)
    g2 = cv2.resize(g2, RESIZE_SHAPE)
    g1 = cv2.GaussianBlur(g1, (5, 5), 0)
    g2 = cv2.GaussianBlur(g2, (5, 5), 0)
    score, _ = ssim(g1, g2, full=True)
    return score

# --- ORB feature matching ---
def compute_orb_diff(img1, img2):
    orb = cv2.ORB_create(1000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        return 0.0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    match_ratio = len(good) / len(matches) if matches else 0.0
    return match_ratio

# --- Combine SSIM + ORB ---
def is_different(img1, img2):
    ssim_score = compute_ssim(img1, img2)
    orb_score = compute_orb_diff(img1, img2)
    print(f"SSIM: {ssim_score:.4f}, ORB: {orb_score:.3f}")
    return ssim_score < SSIM_THRESHOLD or orb_score < ORB_MATCH_THRESHOLD

# --- Timestamp overlay ---
def add_timestamp(image_bgr, timestamp):
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 32)
    except:
        font = ImageFont.load_default()

    text = f"Timestamp: {timestamp:.2f} sec"
    draw.rectangle([(10, 10), (400, 50)], fill=(0, 0, 0, 180))
    draw.text((20, 15), text, fill="white", font=font)
    
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# --- MAIN: Extract + Filter + PDF ---
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)

prev_frame = None
saved_paths = []
frame_count = 0
slide_index = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % FRAME_INTERVAL == 0:
        current_time = frame_count / fps
        processed_frame = add_timestamp(frame.copy(), current_time)

        if prev_frame is None or is_different(prev_frame, frame):
            out_path = os.path.join(TEMP_FOLDER, f"slide_{slide_index:03d}.jpg")
            cv2.imwrite(out_path, processed_frame)
            saved_paths.append(out_path)
            prev_frame = frame
            print(f"✅ Saved slide {slide_index} at {current_time:.2f}s")
            slide_index += 1
        else:
            print(f"⏩ Skipped at {current_time:.2f}s")

    frame_count += 1

cap.release()

# --- Create PDF from saved slides ---
if saved_paths:
    images = [Image.open(p).convert("RGB") for p in saved_paths]
    images[0].save(OUTPUT_PDF, save_all=True, append_images=images[1:])
    print(f"\n✅ PDF saved as: {OUTPUT_PDF}")
else:
    print("❌ No slides saved. PDF not generated.")
