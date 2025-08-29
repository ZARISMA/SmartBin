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

# Timing setup for 3–4 second interval
last_capture_time = time.time()
CAPTURE_INTERVAL = 3  # seconds (change to 4 for 4 sec)

last_label = "Detecting..."
VALID_CLASSES = ["Plastic", "Glass", "Paper", "Aluminum", "Other", "Empty"]

# Flag to prevent overlapping classification calls
is_classifying = False

# === Thread Worker for Gemini ===
def classify_frame(img_bytes):
    global last_label, is_classifying
    try:
        response = model.generate_content([
            {"mime_type": "image/jpeg", "data": img_bytes},
            {"text": (
                "You are a trash sorting AI. The camera always sees a plastic container, but your job is to check what is INSIDE it.\n"
                "Classify ONLY the trash inside the container, ignoring the container itself.\n\n"
                "Return EXACTLY one of these words: Plastic, Glass, Paper, Aluminum, Other, Empty.\n\n"
                "- If the container is empty (no trash inside) → respond Empty.\n"
                "- If there is trash inside, classify it by its main material:\n"
                "   • Plastic\n"
                "   • Glass\n"
                "   • Paper\n"
                "   • Aluminum\n"
                "- If the trash is a mix or unknown → respond Other.\n"
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
    finally:
        is_classifying = False

# === Main Loop ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    # Capture every CAPTURE_INTERVAL seconds if not currently classifying
    if (current_time - last_capture_time >= CAPTURE_INTERVAL) and not is_classifying:
        last_capture_time = current_time
        is_classifying = True
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
