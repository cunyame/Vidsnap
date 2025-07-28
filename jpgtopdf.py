from PIL import Image
import os

output_dir = "data"
images = sorted([img for img in os.listdir(output_dir) if img.endswith('.jpg')])

if not images:
    print("❌ No images found in 'data/' folder.")
    exit()

pdf_images = []
for img_name in images:
    img_path = os.path.join(output_dir, img_name)
    with Image.open(img_path) as img:
        img = img.convert("RGB")
        pdf_images.append(img.copy())

pdf_path = os.path.join(output_dir, "output.pdf")
pdf_images[0].save(pdf_path, save_all=True, append_images=pdf_images[1:])
print(f"✅ PDF saved as: {pdf_path}")
