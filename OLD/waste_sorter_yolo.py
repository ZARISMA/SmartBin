import cv2
import time, threading
from ultralytics import YOLO

# === YOLO Setup ===
model = YOLO("yolov9c.pt")  # or your own model (best.pt)

# === Video Setup ===
cap = cv2.VideoCapture(1)  # change camera index if needed

# === Timing ===
CAPTURE_INTERVAL = 4
last_capture_time = time.time()

# === Classification State ===
last_label = "Detecting..."
is_classifying = False
VALID_CLASSES = ["Plastic", "Glass", "Paper", "Aluminum", "Other", "Empty"]

def classify_frame(frame):
    """Threaded YOLO classification."""
    global last_label, is_classifying
    try:
        results = model.predict(frame, verbose=False)
        detected = [model.names[int(b.cls[0])].lower() for b in results[0].boxes]

        if not detected:
            label = "Empty"
        elif any(k in detected for k in ["plastic", "bottle", "cup", "bag"]):
            label = "Plastic"
        elif any(k in detected for k in ["glass", "jar"]):
            label = "Glass"
        elif any(k in detected for k in ["paper", "cardboard"]):
            label = "Paper"
        elif any(k in detected for k in ["metal", "can", "aluminum"]):
            label = "Aluminum"
        else:
            label = "Other"

        last_label = label
        print("Detected:", label)
    except Exception as e:
        last_label = f"Error: {e}"
        print("Error:", e)
    finally:
        is_classifying = False

# === Main Loop ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()
    if (now - last_capture_time > CAPTURE_INTERVAL) and not is_classifying:
        last_capture_time = now
        is_classifying = True
        threading.Thread(target=classify_frame, args=(frame.copy(),), daemon=True).start()

    cv2.putText(frame, last_label, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("YOLO Waste Sorter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
