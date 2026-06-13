import base64
import os
import uuid
from datetime import datetime
from io import BytesIO

import cv2
import numpy as np
from flask import Flask, jsonify, redirect, render_template, request, send_from_directory, url_for
from PIL import Image

from models import classifier, depth_estimation, hispatial_reasoner, language_generator, scene_graph, sam2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024
app.config['ALLOWED_IMAGE_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp'}
app.config['ALLOWED_VIDEO_EXTENSIONS'] = {'mp4', 'mov', 'avi', 'mkv', 'webm'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def allowed_file(filename, allowed_set):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_set


def save_upload(file_data, prefix='upload'):
    extension = file_data.filename.rsplit('.', 1)[1].lower()
    filename = f"{prefix}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}.{extension}"
    destination = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file_data.save(destination)
    return destination, filename


def prepare_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return np.array(image)


def pipeline_for_image(image_np):
    try:
        print("Starting segmentation...")
        segments = sam2.segment_image(image_np)
        print(f"Segmented {len(segments)} objects")
        
        print("Estimating depth...")
        depth_map = depth_estimation.estimate_depth_map(image_np)
        
        objects = []
        for seg in segments:
            try:
                mask = seg['mask']
                centroid = seg['centroid']
                bbox = seg['bbox']
                label, score = classifier.classify_object(image_np, mask)
                z = depth_estimation.estimate_object_depth(depth_map, mask)
                x, y = float(centroid[0]), float(centroid[1])
                objects.append({
                    'object_id': seg['object_id'],
                    'label': label,
                    'confidence': round(float(score), 3),
                    'mask': seg['mask_base64'],
                    'centroid': {'x': x, 'y': y},
                    'bounding_box': {
                        'x': int(bbox[0]),
                        'y': int(bbox[1]),
                        'width': int(bbox[2]),
                        'height': int(bbox[3])
                    },
                    'size': int(seg['area']),
                    'orientation': seg['orientation'],
                    'depth': round(float(z), 3),
                    'position': {'x': x, 'y': y, 'z': round(float(z), 3)}
                })
            except Exception as e:
                print(f"Error processing object: {e}")
                continue

        print(f"Classified {len(objects)} objects")
        
        print("Building scene graph...")
        graph = scene_graph.build_scene_graph(objects)
        print(f"Generated {len(graph)} edges")
        
        print("Running HiSpatial reasoning...")
        reasoning = hispatial_reasoner.analyze_scene(objects, graph)
        
        print("Generating descriptions...")
        narrative = language_generator.generate_description(objects, graph, reasoning)
        
        return {
            'objects': objects if objects else [],
            'scene_graph': graph if graph else [],
            'reasoning': reasoning if reasoning else {},
            'narrative': narrative if narrative else ['Scene analysis complete.']
        }
    except Exception as e:
        print(f"Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return {
            'objects': [],
            'scene_graph': [],
            'reasoning': {},
            'narrative': [f'Error in processing: {str(e)}']
        }


def build_video_path(filename):
    return url_for('uploaded_file', filename=filename)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/image')
def image_analysis():
    return render_template('image_analysis.html')


@app.route('/video')
def video_analysis():
    return render_template('video_analysis.html')


@app.route('/live')
def live_camera():
    return render_template('live_camera.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/results')
def results():
    return render_template('results.html')


@app.route('/upload_image', methods=['POST'])
def upload_image():
    try:
        if 'image' not in request.files:
            return redirect(url_for('image_analysis'))

        image_file = request.files['image']
        if image_file.filename == '' or not allowed_file(image_file.filename, app.config['ALLOWED_IMAGE_EXTENSIONS']):
            return redirect(url_for('image_analysis'))

        image_path, filename = save_upload(image_file, prefix='image')
        print(f"Uploaded image to: {image_path}")
        
        image_np = prepare_image(image_path)
        print(f"Image shape: {image_np.shape}")
        
        results_data = pipeline_for_image(image_np)
        results_data['media_url'] = build_video_path(filename)
        results_data['media_type'] = 'image'
        results_data['source'] = filename
        
        print(f"Pipeline complete. Objects: {len(results_data['objects'])}")
        return render_template('results.html', results=results_data)
    except Exception as e:
        print(f"Error in upload_image: {e}")
        import traceback
        traceback.print_exc()
        return f"Error processing image: {str(e)}", 500


@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return redirect(url_for('video_analysis'))

    video_file = request.files['video']
    if video_file.filename == '' or not allowed_file(video_file.filename, app.config['ALLOWED_VIDEO_EXTENSIONS']):
        return redirect(url_for('video_analysis'))

    video_path, filename = save_upload(video_file, prefix='video')
    capture = cv2.VideoCapture(video_path)
    frame_results = []
    prev_frame_objects = []
    frame_index = 0
    motion_events = []

    while capture.isOpened() and frame_index < 30:
        grabbed, frame = capture.read()
        if not grabbed:
            break

        if frame_index % 10 == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_result = pipeline_for_image(frame_rgb)
            frame_results.append(frame_result)

            if prev_frame_objects:
                movement = hispatial_reasoner.analyze_video_motion(prev_frame_objects, frame_result['objects'])
                motion_events.extend(movement)

            prev_frame_objects = frame_result['objects']
        frame_index += 1

    capture.release()

    if frame_results:
        results_data = frame_results[-1]
    else:
        results_data = {'objects': [], 'scene_graph': [], 'reasoning': {}, 'narrative': ['No valid frames found.']}

    results_data['media_url'] = build_video_path(filename)
    results_data['media_type'] = 'video'
    results_data['source'] = filename
    results_data['motion_events'] = motion_events
    return render_template('results.html', results=results_data)


@app.route('/process_frame', methods=['POST'])
def process_frame():
    payload = request.get_json(force=True)
    image_b64 = payload.get('image')
    if not image_b64:
        return jsonify({'error': 'No image provided'}), 400

    header, encoded = image_b64.split(',', 1) if ',' in image_b64 else ('', image_b64)
    image_data = base64.b64decode(encoded)
    image = Image.open(BytesIO(image_data)).convert('RGB')
    image_np = np.array(image)
    results_data = pipeline_for_image(image_np)
    return jsonify(results_data)


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
