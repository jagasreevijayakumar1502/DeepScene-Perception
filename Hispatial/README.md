# HiSpatial Environmental Perception Dashboard

A Flask-based research dashboard for environmental perception, scene understanding, and spatial reasoning.

## Features

- Upload images or videos for AI-powered perception
- Live webcam feed analysis with frame-by-frame processing
- Object segmentation with SAM-style masks
- Vision-language classification of segmented objects
- Monocular depth estimation and object 3D positions
- Scene graph generation with spatial relations
- HiSpatial hierarchical reasoning layers
- Natural-language scene descriptions
- Modern dark dashboard UI with responsive layout

## Install

1. Create a Python virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Optional: if you want full SAM and MiDaS support, install the required models and packages.

## Run

```powershell
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

## Notes

- The app includes fallback vision and depth methods when advanced models are not installed.
- Uploaded media files are saved under `static/uploads`.
- Live camera processing sends periodic frames to the backend for scene analysis.
