import cv2
import google.generativeai as genai
import time, threading, os, json
from datetime import datetime

# === Gemini Setup ===
genai.configure(api_key="AIzaSyAsPxaSz1WEuJ7uTJKQYz0fQAJOlk5djO8")  
model = genai.GenerativeModel("gemini-2.5-flash-lite")

# === Video Setup ===
cap = cv2.VideoCapture(3)  # Try 0 or 1 depending on which camera works

# === Timing Setup ===
last_capture_time = time.time()
CAPTURE_INTERVAL = 10  # seconds between classifications

# === Classification State ===
last_label = "Detecting..."
VALID_CLASSES = ["Plastic", "Glass", "Paper", "Aluminum", "Other", "Empty"]
is_classifying = False

# === Dataset Setup ===
DATASET_DIR = "waste_dataset"
os.makedirs(DATASET_DIR, exist_ok=True)
META_FILE = os.path.join(DATASET_DIR, "metadata.json")

if os.path.exists(META_FILE):
    with open(META_FILE, "r") as f:
        metadata = json.load(f)
else:
    metadata = []

# === Helper: Save dataset entry ===
def save_dataset_entry(label, img, description):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{label}_{timestamp}.jpg"
    filepath = os.path.join(DATASET_DIR, filename)
    cv2.imwrite(filepath, img)

    entry = {
        "filename": filename,
        "label": label,
        "description": description,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    metadata.append(entry)
    with open(META_FILE, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"[DATASET] Saved: {filename}")

# === Worker Thread: Classify Frame with Gemini ===
def classify_frame(img_bytes, img_original):
    global last_label, is_classifying
    try:
        # --- Ask Gemini to classify the trash ---
        response = model.generate_content([
            {"mime_type": "image/jpeg", "data": img_bytes},
            {"text": (
                "You are a trash sorting AI. The camera always sees a light green plastic container, "
                "but your job is to check what is INSIDE it.\n"
                "Classify ONLY the trash inside the light green container, ignoring the container itself.\n\n"
                "Return EXACTLY one of these words: Plastic, Glass, Paper, Aluminum, Other, Empty."
            )}
        ])

        print("Gemini raw response:", response.text)
        label = response.text.strip().capitalize()
        if label not in VALID_CLASSES:
            label = "Empty"

        if label != "Empty":
            # --- Get a short descriptive caption ---
            desc_response = model.generate_content([
                {"mime_type": "image/jpeg", "data": img_bytes},
                {"text": (
                    "Briefly describe the trash item in one sentence, mentioning material, color, and shape. "
                    "Example: 'A crushed transparent plastic bottle' or 'A piece of brown paper bag'."
                )}
            ])
            description = desc_response.text.strip()

            # --- Save dataset entry ---
            save_dataset_entry(label, img_original, description)

        last_label = label

    except Exception as e:
        print("Error occurred:", str(e))
        last_label = f"Error: {str(e)}"[:40]
    finally:
        is_classifying = False


# === Main Camera Loop ===
while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera not detected or failed to open.")
        break

    current_time = time.time()
    if (current_time - last_capture_time >= CAPTURE_INTERVAL) and not is_classifying:
        last_capture_time = current_time
        is_classifying = True

        _, img_encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        img_bytes = img_encoded.tobytes()
        threading.Thread(target=classify_frame, args=(img_bytes, frame.copy()), daemon=True).start()

    cv2.putText(frame, f"{last_label}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Gemini Waste Sorter (Camera Only)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()
