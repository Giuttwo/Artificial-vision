import cv2
import numpy as np
from ultralytics import YOLO

# ============================================================
# Cargar YOLO
# ============================================================

WEIGHTS_PATH = "yolov8n.pt"   # el modelo que ya te funcionaba
print(f"[INFO] Cargando modelo YOLO: {WEIGHTS_PATH}")

model = YOLO(WEIGHTS_PATH)
print("[INFO] Modelo YOLO cargado correctamente.")


# ============================================================
# Preprocesado básico 
# ============================================================

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detecta cambios bruscos de intensidad (útil para bordes de objetos alargados)
    sobel = cv2.Sobel(blur, cv2.CV_64F, 1, 1, ksize=3)
    sobel = cv2.convertScaleAbs(sobel)

    # Filtro de bordes más duro
    edges = cv2.Canny(blur, 80, 150)

    # Pequeña expansión de bordes
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)

    # Cierra huecos pequeños en formas
    morph = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=1)

    return gray, sobel, morph


# ============================================================
# Detección usando YOLO
# ============================================================

def detect_knives(frame):
    results = model(frame, verbose=False)[0]

    detections = []
    for box in results.boxes:
        cls = int(box.cls[0])
        label = results.names[cls].lower()

        # Verifica si YOLO considera la clase como cuchillo
        if "knife" in label or "cuchillo" in label:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            detections.append((int(x1), int(y1), int(x2), int(y2)))

    return detections


# ============================================================
# Lógica principal
# ============================================================

def main():
    print("[INFO] Iniciando cámara...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] No se pudo leer el frame.")
            break

        gray, sobel, morph = preprocess_frame(frame)
        detections = detect_knives(frame)

        # Dibuja detecciones
        for (x1, y1, x2, y2) in detections:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "KNIFE", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Ventana principal
        cv2.imshow("Detección de Cuchillos (YOLO)", frame)

        # ------- Ventana extra con 2 filtros juntos -------
        filters = np.hstack((gray, sobel))
        cv2.imshow("Filtros (Gray | Sobel)", filters)

        # Salir con Q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Cámara finalizada.")


if __name__ == "__main__":
    main()
