import cv2
import google.generativeai as genai
import serial, time, threading

# === Gemini Setup ===
genai.configure(api_key="")
model = genai.GenerativeModel("gemini-2.5-flash-lite")

# === Arduino Setup ===
arduino = serial.Serial('COM6', 9600, timeout=1)  # adjust COM if needed
time.sleep(2)

# === Video Setup ===
cap = cv2.VideoCapture(1)
frame_count = 0
SAMPLE_RATE = 300  # ~2 sec if 30 FPS
last_label = "Detecting..."
VALID_CLASSES = ["Plastic", "Glass", "Paper", "Aluminum", "Other", "Empty"]

# === Thread Worker for Gemini ===
def classify_frame(img_bytes):
    global last_label
    try:
        response = model.generate_content([
            {"mime_type": "image/jpeg", "data": img_bytes},
            {"text": (
                "You are a trash sorting AI. Look at the image and classify the content into one of EXACTLY these words: "
                "Plastic, Glass, Paper, Aluminum, Other, Empty.\n\n"
                "- If no trash or nothing is visible → respond Empty.\n"
                "- If trash is inside a plastic bag → respond Other.\n"
                "- Do NOT explain, ONLY respond with one of the six words."
            )}
        ])

        # Normalize label
        label = response.text.strip().capitalize()
        if label not in VALID_CLASSES:
            label = "Empty"  # default if response is weird

        last_label = label

        # Send only valid classification to Arduino
        if label != "Empty":
            arduino.write((label + "\n").encode())

    except Exception as e:
        last_label = "Error"

# === Main Loop ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % SAMPLE_RATE == 0:
        _, img_encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        img_bytes = img_encoded.tobytes()
        threading.Thread(target=classify_frame, args=(img_bytes,), daemon=True).start()

    # Show prediction overlay
    cv2.putText(frame, f"Detected: {last_label}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Gemini Waste Sorter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
arduino.close()
cv2.destroyAllWindows()
