# Real-time Object Detection
A real-time detection game using Flask + YOLO

## ⚙️ Features
- Live webcam detection
- Random 5-object challenge
- Auto-updating checklist
- Reset button

## 📂 Project Structure
```
real-time-object-detection/
│
├── app.py              # Main logic
│
├── templates/
│   └── index.html      # Front-end
│
├── object_detecion.pt  # Model
├── train.ipynb         # Training script
│
├── requirements.txt
├── README.md
└── .gitignore
```
## ▶️ Running the Project

1. Clone the repository

```bash
git clone https://github.com/HuynhHanDong/real-time-object-detection.git
cd https://github.com/HuynhHanDong/real-time-object-detection.git
```
2. Create and activate a virtual environment

```bash
python -m venv venv
```

&emsp;&emsp;Windows:  
&emsp;&emsp;&emsp; ```.venv\Scripts\Activate.ps1``` (powershell)  
&emsp;&emsp;&emsp; ```.venv\Scripts\activate``` (command prompt)  

&emsp;&emsp;macOS/Linux:  ```source .venv/bin/activate```  

3. Install dependencies:

```
pip install -r requirements.txt
```  

4. Run the Flask app:

```
python app.py
```

5. Open your browser at:  ```http://127.0.0.1:5000/```