import cv2
import numpy as np
from ultralytics import YOLO
from transformers import pipeline
from PIL import Image

# ============================================================
# Load YOLO model (forcing CPU)
# ============================================================

WEIGHTS_PATH = "yolov8n.pt"
print(f"[INFO] Loading YOLO Model: {WEIGHTS_PATH}")

model = YOLO(WEIGHTS_PATH)
model.to("cpu")  # Force CPU
print("[INFO] YOLO loaded successfully on CPU.")

# ============================================================
# Load Vision Transformer for knife crop verification (forcing CPU)
# ============================================================

print("[INFO] Loading Vision Transformer (ViT) for crop verification on CPU...")
vit_classifier = pipeline(
    "image-classification",
    model="google/vit-base-patch16-224",
    device=-1  # Force CPU
)
print("[INFO] ViT loaded successfully on CPU.")

# Labels in ImageNet that correspond to knives/bladed weapons
accepted_knives = [
    "cleaver",             # hacha de cocina / cuchillo tipo hacha
    "letter opener",       # abrecartas (cuchillo pequeño)
    "paper knife",         # abrecartas
    "pocketknife",         # navaja
    "pocket knife",        # variaciones posibles
    "swiss army knife",    # navaja multiusos
    "bolo",                # cuchillo/machete largo
    "steak knife",         # aparece en variantes a veces
    "kitchen knife",       # cuando el modelo está fine o reetiquetado
    "utility knife",       # cortadora/navaja, a veces predicha
]

# ============================================================
# Basic offline preprocessing (remains unchanged)
# ============================================================

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect sharp intensity changes (useful for elongated object boundaries)
    sobel = cv2.Sobel(blur, cv2.CV_64F, 1, 1, ksize=3)
    sobel = cv2.convertScaleAbs(sobel)

    # Hard edge detection
    edges = cv2.Canny(blur, 80, 150)

    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    morph = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=1)

    return gray, sobel, morph

# ============================================================
# YOLO knife detection (remains unchanged)
# ============================================================

def detect_knives(frame):
    results = model(frame, verbose=False)[0]

    detections = []
    for box in results.boxes:
        cls = int(box.cls[0])
        label = results.names[cls].lower()

        # Keep original logic: YOLO proposes knives
        if "knife" in label or "cuchillo" in label:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            detections.append((int(x1), int(y1), int(x2), int(y2)))

    return detections

# ============================================================
# Main loop (integrates YOLO + realistic ViT verification)
# ============================================================

def main():
    print("[INFO] Starting camera...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Could not access camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Could not read camera frame.")
            break

        gray, sobel, morph = preprocess_frame(frame)
        detections = detect_knives(frame)

        # --- Verify each YOLO crop with ViT using realistic labels ---
        for (x1, y1, x2, y2) in detections:
            crop = frame[y1:y2, x1:x2]

            # Convert OpenCV BGR → RGB → PIL (required format for ViT)
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_pil = Image.fromarray(crop_rgb)

            vit_result = vit_classifier(crop_pil)[0]
            label = vit_result['label'].lower()
            confidence = vit_result['score']

            # Print ViT prediction for debugging/validation
            print(f"[ViT] Prediction: {label} (confidence: {confidence:.2f})")

            # Accept if label matches any bladed weapon category and confidence is high enough
            if any(k in label for k in accepted_knives) and confidence > 0.75:
                # ViT verified as a knife → draw alert
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "KNIFE (verified)", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                # Rejected → show label for guidance but don't alert
                cv2.putText(frame, f"Rejected (ViT: {label})", (x1, y2 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # --- Display windows (unchanged behavior) ---
        cv2.imshow("Knife Detection (YOLO + ViT)", frame)

        # Stack original grayscale + Sobel preview window
        filters = np.hstack((gray, sobel))
        cv2.imshow("Filters (Gray | Sobel)", filters)

        # Quit with Q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Camera finished.")

if __name__ == "__main__":
    main()
