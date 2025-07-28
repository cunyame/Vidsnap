import cv2
import os

# Load video
vid = cv2.VideoCapture(r'C:\Users\Admin\Videos\Captures\vid.mp4')
fps = vid.get(cv2.CAP_PROP_FPS)

output_dir = 'data'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

frame_interval = int(fps * 1)  # Extract every 1 second
frame_count = 0
saved_frame = 0

while True:
    success, frame = vid.read()
    if not success:
        break

    if frame_count % frame_interval == 0:
        timestamp_sec = int(frame_count / fps)
        hours = timestamp_sec // 3600
        minutes = (timestamp_sec % 3600) // 60
        seconds = timestamp_sec % 60
        timestamp = f"{hours:02}:{minutes:02}:{seconds:02}"

        # Add timestamp visibly
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        text_size, _ = cv2.getTextSize(timestamp, font, font_scale, thickness)
        text_w, text_h = text_size
        position = (10, 30 + text_h)

        # Background rectangle for visibility
        cv2.rectangle(frame, (5, 5), (15 + text_w, 40 + text_h), (255, 255, 255), -1)
        cv2.putText(frame, timestamp, position, font, font_scale, (0, 0, 0), thickness)

        # Save with quality
        filename = os.path.join(output_dir, f"frame{saved_frame:03}.jpg")
        cv2.imwrite(filename, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        saved_frame += 1

    frame_count += 1

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
