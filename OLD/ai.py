import cv2
import google.generativeai as genai

# Setup Gemini API
genai.configure(api_key="")
model = genai.GenerativeModel("gemini-2.5-flash-lite")  # Fastest for real-time

# Phone camera stream URL
#url = "http://192.168.8.85:4747/video"
cap = cv2.VideoCapture(1)

frame_count = 0
SAMPLE_RATE = 30  # send 1 frame per second if camera is ~30 FPS
last_label = "Detecting..."

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Only send every Nth frame to Gemini
    if frame_count % SAMPLE_RATE == 0:
        try:
            _, img_encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])  # compress
            img_bytes = img_encoded.tobytes()

            response = model.generate_content([
                {"mime_type": "image/jpeg", "data": img_bytes},
                {"text": "Classify this trash into one of: Plastic, Glass, Paper, Aluminum, Other. "
                         "If it is in a plastic bag, always choose Other."}
            ])

            last_label = response.text.strip()
        except Exception as e:
            last_label = f"Error: {e}"

    # Show latest prediction on every frame
    cv2.putText(frame, f"Detected: {last_label}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Gemini Waste Sorter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
