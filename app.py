from flask import Flask, render_template, Response, jsonify
import cv2
import random
from ultralytics import YOLO

app = Flask(__name__)

# Load a pretrained YOLO11n model
model = YOLO("yolo11n.pt")
cagetories = ['person', 'umbrella', 'backpack', 'bottle', 'cup', 'fork', 'spoon', 'chair', 'laptop', 'mouse', 'cell phone', 'clock']
#all_classes = model.names.values()

# Pick 5 random required objects
required_objects = random.sample(cagetories, 5)
detected_objects = set()

# Video capture
cap = cv2.VideoCapture(0)

def generate_frames():
    global detected_objects
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Run YOLO inference
        results = model(frame, verbose=False)[0]

        # Draw boxes and track detections
        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])

            # Draw rectangle
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Update checklist
            if label in required_objects:
                detected_objects.add(label)

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        
@app.route('/')
def index():
    return render_template('index.html', required_objects=required_objects)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

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
    required_objects = random.sample(cagetories, 5)
    detected_objects = set()
    return jsonify({"new_objects": required_objects})

if __name__ == "__main__":
    app.run(debug=True)