import os
from pathlib import Path
from flask import Flask, request, send_from_directory, render_template, jsonify
import cv2
import numpy as np
from sam2.build_sam import build_sam2_video_predictor

app = Flask(__name__, static_folder="static", template_folder="templates")

UPLOAD_FOLDER = Path(__file__).parent / "static" / "uploads"
RESULT_FOLDER = Path(__file__).parent / "results"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULT_FOLDER.mkdir(parents=True, exist_ok=True)

# Configuration for SAM2 checkpoint and config file
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
CHECKPOINT = "checkpoints/sam2.1_hiera_base_plus.pt"
DEVICE = "cpu"

_predictor = None
_points = []
_labels = []
_video_path: Path | None = None


def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = build_sam2_video_predictor(MODEL_CFG, CHECKPOINT, device=DEVICE)
    return _predictor


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    global _video_path, _points, _labels
    file = request.files.get("video")
    if file is None:
        return "No file", 400
    _points = []
    _labels = []
    filepath = UPLOAD_FOLDER / file.filename
    file.save(filepath)
    _video_path = filepath
    cap = cv2.VideoCapture(str(filepath))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return "Invalid video", 400
    img_path = UPLOAD_FOLDER / "first_frame.jpg"
    cv2.imwrite(str(img_path), frame)
    return jsonify({"frame": f"/static/uploads/first_frame.jpg"})


@app.route("/static/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/add_point", methods=["POST"])
def add_point():
    global _points, _labels
    data = request.get_json(force=True)
    x = float(data.get("x"))
    y = float(data.get("y"))
    label = int(data.get("label", 1))
    _points.append([x, y])
    _labels.append(label)
    return jsonify({"status": "ok"})


@app.route("/process", methods=["POST"])
def process_video():
    if _video_path is None:
        return "No video uploaded", 400
    predictor = get_predictor()
    state = predictor.init_state(str(_video_path), offload_video_to_cpu=True)
    if _points:
        predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=0,
            obj_id=1,
            points=_points,
            labels=_labels,
            clear_old_points=True,
            normalize_coords=False,
        )
    masks = []
    for frame_idx, obj_ids, scores in predictor.propagate_in_video(
        inference_state=state, start_frame_idx=0
    ):
        mask = (scores[0, 0] > 0).cpu().numpy()
        masks.append(mask)

    cap = cv2.VideoCapture(str(_video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out_path = RESULT_FOLDER / "result.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or idx >= len(masks):
            break
        mask = masks[idx]
        color = np.array([0, 255, 0], dtype=np.uint8)
        frame[mask] = frame[mask] * 0.5 + color * 0.5
        writer.write(frame)
        idx += 1
    writer.release()
    cap.release()
    return jsonify({"download": f"/result/result.mp4"})


@app.route("/result/<path:filename>")
def result_file(filename):
    return send_from_directory(RESULT_FOLDER, filename)


if __name__ == "__main__":
    app.run(debug=True)
