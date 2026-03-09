import cv2
import google.generativeai as genai
import serial, time

# === Gemini Setup ===
genai.configure(api_key="")
model = genai.GenerativeModel("gemini-1.5-flash")

# === Arduino Setup ===
arduino = serial.Serial('COM3', 9600, timeout=1)
time.sleep(2)  # wait for Arduino to initialize

# === Video Setup ===
cap = cv2.VideoCapture(1)  # 0 = laptop cam, 1 = phone cam (via DroidCam/Iriun)
frame_count = 0
SAMPLE_RATE = 60  # ~2 seconds if 30 FPS
last_label = "Detecting..."

# Allowed outputs for Arduino
VALID_CLASSES = ["Plastic", "Glass", "Paper", "Aluminum", "Other", "Empty"]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Only send every Nth frame to Gemini
    if frame_count % SAMPLE_RATE == 0:
        try:
            # Compress image for faster upload
            _, img_encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            img_bytes = img_encoded.tobytes()

            response = model.generate_content([
                {"mime_type": "image/jpeg", "data": img_bytes},
                {"text": "Classify this trash into one of: Plastic, Glass, Paper, Aluminum, Other, Empty. "
                         "If the bin is empty, always choose Empty. "
                         "If trash is in a plastic bag, choose Other. Output only one word."}
            ])

            label = response.text.strip()

            # Sanitize: force into valid set
            if label not in VALID_CLASSES:
                label = "Other"

            last_label = label

            # Send to Arduino ONLY if not Empty
            if label != "Empty":
                arduino.write((label + "\n").encode())

        except Exception as e:
            last_label = f"Error"

    # Show latest prediction on every frame
    cv2.putText(frame, f"Detected: {last_label}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Gemini Waste Sorter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
arduino.close()
cv2.destroyAllWindows()
