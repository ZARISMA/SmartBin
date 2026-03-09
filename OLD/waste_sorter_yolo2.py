import cv2, time, threading
from ultralytics import YOLO

# === YOLO Setup ===
model = YOLO("yolov9c.pt")  # or your own trained model

# === Video Setup ===
cap = cv2.VideoCapture(1)  # change index if needed

CAPTURE_INTERVAL = 2
last_capture_time = time.time()
last_label = "Detecting..."
is_classifying = False

VALID_CLASSES = ["Plastic", "Glass", "Paper", "Aluminum", "Other", "Empty"]

# ---- improved mapping keywords ----
PLASTIC_KEYWORDS  = ["plastic", "bottle", "cup", "bag", "wrapper", "container"]
GLASS_KEYWORDS    = ["glass", "jar", "wine", "beer", "flask"]
PAPER_KEYWORDS    = ["book", "paper", "envelope", "box", "cardboard", "napkin", "magazine", "tissue", "poster", "notebook", "newspaper"]
ALUMINUM_KEYWORDS = ["can", "aluminum", "metal", "tin", "foil"]

def classify_frame(frame):
    """YOLO classification thread."""
    global last_label, is_classifying
    try:
        results = model.predict(frame, verbose=False)
        detected = [model.names[int(b.cls[0])].lower() for b in results[0].boxes]

        if not detected:
            label = "Empty"
        elif any(k in detected for k in PLASTIC_KEYWORDS):
            label = "Plastic"
        elif any(k in detected for k in GLASS_KEYWORDS):
            label = "Glass"
        elif any(k in detected for k in PAPER_KEYWORDS):
            label = "Paper"
        elif any(k in detected for k in ALUMINUM_KEYWORDS):
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
