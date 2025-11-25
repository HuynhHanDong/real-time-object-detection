from flask import Flask, render_template, Response, jsonify
import cv2
import random
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLO model
model = YOLO(r"C:\Users\HanDong\Documents\AI crew\Flask\real-time_object_detection\runs/detect/train/weights/best.pt")
#categories = model.names.values()
categories = ['backpack', 'book', 'bottle', 'phone', 'chair', 'cup', 'fork', 'laptop', 'F-Code logo', 'mouse', 'person', 'spoon', 'umbrella']

# Initial random 5 objects
required_objects = random.sample(categories, 5)
detected_objects = set()

def generate_frames():
    global detected_objects

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Could not open camera")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Run YOLO
        results = model(frame, verbose=False)[0]

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if label in required_objects:
                detected_objects.add(label)

        # Encode JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n'
               b'Content-Length: ' + str(len(frame)).encode() + b'\r\n\r\n' +
               frame + b'\r\n\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html', required_objects=required_objects)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame', direct_passthrough=True)

@app.route('/get_status')
def get_status():
    status = {obj: (obj in detected_objects) for obj in required_objects}
    found_count = sum(status.values())
    return jsonify({
        "status": status,
        "found": found_count,
        "total": len(required_objects)
    })

@app.route('/reset')
def reset():
    global required_objects, detected_objects
    required_objects = random.sample(categories, 5)
    detected_objects = set()
    return jsonify({"new_objects": required_objects})

if __name__ == "__main__":
    app.run(debug=True)