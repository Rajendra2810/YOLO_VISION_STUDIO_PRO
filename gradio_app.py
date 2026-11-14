"""
YOLO Vision Studio Pro - Gradio Edition
----------------------------------------
Standalone Gradio interface based on the Streamlit app.py.

Supported tasks:
- Object Detection
- Object Counting
- Customer & Retail
- Object Tracking
- Object Segmentation
- Pose Estimation

Run locally:
    python gradio_app.py

Deploy on Hugging Face Spaces with Gradio.
"""

import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO


APP_DIR = Path(__file__).resolve().parent

DEFAULT_TASK_MODELS = {
    "Object Detection": "yolov8n.pt",
    "Object Counting": "yolov8n.pt",
    "Customer & Retail": "yolov8n.pt",
    "Object Tracking": "yolov8n.pt",
    "Object Segmentation": "yolov8n-seg.pt",
    "Pose Estimation": "yolov8n-pose.pt",
}

AVAILABLE_LOCAL_MODELS = [p.name for p in APP_DIR.glob("*.pt")]

if not AVAILABLE_LOCAL_MODELS:
    AVAILABLE_LOCAL_MODELS = [
        "yolov8n.pt",
        "yolov8n-seg.pt",
        "yolov8n-pose.pt",
        "yolo11n.pt",
    ]


MODEL_CACHE = {}


def load_yolo_model(model_identifier: str):
    """Load and cache a YOLO model."""
    if model_identifier in MODEL_CACHE:
        return MODEL_CACHE[model_identifier]

    local_path = APP_DIR / model_identifier

    if local_path.exists():
        model = YOLO(str(local_path))
    else:
        # Ultralytics can download standard model names automatically.
        model = YOLO(model_identifier)

    MODEL_CACHE[model_identifier] = model
    return model


def resolve_model_for_task(task: str, model_choice: str):
    """Select the model based on task and user selection."""
    if model_choice and model_choice != "Auto (Recommended)":
        return load_yolo_model(model_choice)

    default_model = DEFAULT_TASK_MODELS.get(task, "yolov8n.pt")
    return load_yolo_model(default_model)


def normalize_input_image(image_input):
    """Convert Gradio/PIL/NumPy input to RGB uint8 NumPy array."""
    if image_input is None:
        return None

    if isinstance(image_input, Image.Image):
        return np.array(image_input.convert("RGB"))

    if isinstance(image_input, np.ndarray):
        arr = image_input.copy()

        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        if arr.ndim == 2:
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)

        if arr.ndim == 3 and arr.shape[2] == 4:
            return cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)

        if arr.ndim == 3 and arr.shape[2] == 3:
            return arr

    try:
        arr = np.array(image_input)
        if arr.ndim == 3 and arr.shape[2] == 3:
            return arr.astype(np.uint8)
    except Exception:
        pass

    return None


def execute_yolo_inference(
    image,
    task,
    model_choice="Auto (Recommended)",
    confidence=0.25,
    iou_threshold=0.45,
    imgsz=480,
    allowed_classes=None,
    show_count_banner=True,
):
    """
    Run YOLO inference and return:
    annotated image, detection dataframe, class counts, latency, status.
    """
    start_time = time.perf_counter()

    image_rgb = normalize_input_image(image)

    if image_rgb is None:
        return None, pd.DataFrame(), {}, 0.0, "No valid image provided."

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    try:
        model = resolve_model_for_task(task, model_choice)
    except Exception as exc:
        return None, pd.DataFrame(), {}, 0.0, f"Error loading model: {exc}"

    class_indices = None

    if allowed_classes and len(allowed_classes) > 0 and hasattr(model, "names"):
        name_to_id = {v: k for k, v in model.names.items()}
        class_indices = [
            name_to_id[c]
            for c in allowed_classes
            if c in name_to_id
        ]

    # Customer & Retail focuses on the person class.
    if task == "Customer & Retail" and not class_indices:
        if hasattr(model, "names"):
            name_to_id = {v: k for k, v in model.names.items()}
            if "person" in name_to_id:
                class_indices = [name_to_id["person"]]

    try:
        if task == "Object Tracking":
            results = model.track(
                source=image_bgr,
                conf=float(confidence),
                iou=float(iou_threshold),
                imgsz=int(imgsz),
                classes=class_indices,
                persist=True,
                verbose=False,
            )
        else:
            results = model.predict(
                source=image_bgr,
                conf=float(confidence),
                iou=float(iou_threshold),
                imgsz=int(imgsz),
                classes=class_indices,
                verbose=False,
            )
    except Exception as exc:
        return (
            None,
            pd.DataFrame(),
            {},
            0.0,
            f"Inference execution failed: {exc}",
        )

    latency_ms = (time.perf_counter() - start_time) * 1000.0

    if not results:
        return image_rgb, pd.DataFrame(), {}, latency_ms, "No results returned."

    result = results[0]

    annotated_bgr = result.plot()

    if annotated_bgr is None:
        annotated_bgr = image_bgr.copy()

    boxes = getattr(result, "boxes", None)

    detections = []
    class_counts = {}

    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)

        track_ids = None

        if hasattr(boxes, "id") and boxes.id is not None:
            try:
                track_ids = boxes.id.cpu().numpy().astype(int)
            except Exception:
                track_ids = None

        names = getattr(result, "names", {})

        for i in range(len(boxes)):
            cls_id = int(clss[i])

            if isinstance(names, dict):
                cls_name = names.get(cls_id, str(cls_id))
            else:
                cls_name = str(cls_id)

            conf_value = float(confs[i])

            x1, y1, x2, y2 = map(int, xyxy[i])

            track_id = (
                int(track_ids[i])
                if track_ids is not None
                else "-"
            )

            class_counts[cls_name] = (
                class_counts.get(cls_name, 0) + 1
            )

            detections.append(
                {
                    "#": i + 1,
                    "Class": cls_name,
                    "Confidence": f"{conf_value * 100:.1f}%",
                    "Track ID": track_id,
                    "Box [X1, Y1, X2, Y2]":
                        f"[{x1}, {y1}, {x2}, {y2}]",
                    "Width (px)": x2 - x1,
                    "Height (px)": y2 - y1,
                }
            )

    # Add count banner, matching the Streamlit app behavior.
    total_detected = len(detections)

    if show_count_banner and total_detected > 0:
        banner_text = f"Count: {total_detected}"

        if task == "Customer & Retail":
            banner_text = f"Customers: {total_detected}"

        (text_w, text_h), _ = cv2.getTextSize(
            banner_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            2,
        )

        cv2.rectangle(
            annotated_bgr,
            (12, 12),
            (18 + text_w + 10, 20 + text_h + 10),
            (15, 23, 42),
            -1,
        )

        cv2.rectangle(
            annotated_bgr,
            (12, 12),
            (18 + text_w + 10, 20 + text_h + 10),
            (99, 102, 241),
            2,
        )

        cv2.putText(
            annotated_bgr,
            banner_text,
            (18, 20 + text_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (52, 211, 153),
            2,
            cv2.LINE_AA,
        )

    annotated_rgb = cv2.cvtColor(
        annotated_bgr,
        cv2.COLOR_BGR2RGB,
    )

    detections_df = pd.DataFrame(detections)

    status = "Success"

    return (
        annotated_rgb,
        detections_df,
        class_counts,
        latency_ms,
        status,
    )


COMMON_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "backpack",
    "umbrella",
    "handbag",
    "bottle",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "laptop",
    "cell phone",
]


def run_image(
    image,
    task,
    model_choice,
    confidence,
    iou_threshold,
    imgsz,
    selected_classes,
    show_count_banner,
):
    """Gradio image-processing callback."""

    annotated, df, counts, latency, status = execute_yolo_inference(
        image=image,
        task=task,
        model_choice=model_choice,
        confidence=confidence,
        iou_threshold=iou_threshold,
        imgsz=imgsz,
        allowed_classes=selected_classes,
        show_count_banner=show_count_banner,
    )

    if annotated is None:
        return (
            None,
            pd.DataFrame(),
            f"❌ {status}",
        )

    total = len(df)

    count_text = ", ".join(
        f"{name}: {count}"
        for name, count in sorted(
            counts.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    )

    if not count_text:
        count_text = "No objects detected."

    report = (
        f"Status: {status}\n"
        f"Task: {task}\n"
        f"Total Objects: {total}\n"
        f"Inference Latency: {latency:.1f} ms\n"
        f"Classes: {count_text}"
    )

    return annotated, df, report


def run_video(
    video_path,
    task,
    model_choice,
    confidence,
    iou_threshold,
    imgsz,
    frame_skip,
    max_frames,
):
    """Process an uploaded video and return an annotated video."""

    if not video_path:
        return None, "Please upload a video."

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None, "Could not open the uploaded video."

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = str(
        APP_DIR / "gradio_yolo_result.mp4"
    )

    # mp4v is broadly available in standard OpenCV builds.
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_index = 0
    processed = 0
    total_objects = 0
    total_latency = 0.0

    try:
        while cap.isOpened() and processed < int(max_frames):
            ret, frame = cap.read()

            if not ret:
                break

            frame_index += 1

            if frame_index % int(frame_skip) != 0:
                continue

            frame_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            annotated, df, _, latency, status = execute_yolo_inference(
                image=frame_rgb,
                task=task,
                model_choice=model_choice,
                confidence=confidence,
                iou_threshold=iou_threshold,
                imgsz=imgsz,
                allowed_classes=None,
                show_count_banner=True,
            )

            if annotated is None:
                continue

            annotated_bgr = cv2.cvtColor(
                annotated,
                cv2.COLOR_RGB2BGR,
            )

            # Keep output dimensions equal to the original video.
            annotated_bgr = cv2.resize(
                annotated_bgr,
                (width, height),
            )

            writer.write(annotated_bgr)

            processed += 1
            total_objects += len(df)
            total_latency += latency

    finally:
        cap.release()
        writer.release()

    if processed == 0:
        return None, "No video frames were processed."

    avg_latency = total_latency / processed

    report = (
        f"Video processing complete.\n"
        f"Task: {task}\n"
        f"Frames processed: {processed}\n"
        f"Total object detections: {total_objects}\n"
        f"Average inference latency: {avg_latency:.1f} ms"
    )

    return output_path, report


with gr.Blocks(
    title="YOLO Vision Studio Pro",
    theme=gr.themes.Soft(),
) as demo:

    gr.Markdown(
        """
        # 🎯 YOLO Vision Studio Pro
        ### Unified YOLOv8 & YOLO11 Computer Vision Interface

        Object Detection • Counting • Customer Analytics • Tracking
        • Segmentation • Pose Estimation
        """
    )

    with gr.Tab("🖼️ Image Studio"):

        with gr.Row():

            with gr.Column(scale=1):

                task = gr.Dropdown(
                    choices=list(DEFAULT_TASK_MODELS.keys()),
                    value="Object Detection",
                    label="Vision Task",
                )

                model_choice = gr.Dropdown(
                    choices=["Auto (Recommended)"] + AVAILABLE_LOCAL_MODELS,
                    value="Auto (Recommended)",
                    label="Active Model",
                )

                confidence = gr.Slider(
                    minimum=0.05,
                    maximum=0.95,
                    value=0.25,
                    step=0.05,
                    label="Confidence Threshold",
                )

                iou_threshold = gr.Slider(
                    minimum=0.10,
                    maximum=0.90,
                    value=0.45,
                    step=0.05,
                    label="NMS IoU Threshold",
                )

                imgsz = gr.Dropdown(
                    choices=[320, 480, 640],
                    value=480,
                    label="Inference Size",
                )

                selected_classes = gr.Dropdown(
                    choices=COMMON_CLASSES,
                    multiselect=True,
                    label="Filter Classes",
                    info="Leave empty to detect all available classes.",
                )

                show_count_banner = gr.Checkbox(
                    value=True,
                    label="Show Count Banner",
                )

            with gr.Column(scale=2):

                input_image = gr.Image(
                    type="numpy",
                    label="Input Image",
                )

                run_button = gr.Button(
                    "🚀 Run YOLO Inference",
                    variant="primary",
                )

                output_image = gr.Image(
                    type="numpy",
                    label="YOLO Result",
                )

    with gr.Tab("📊 Detection Results"):

        result_table = gr.Dataframe(
            headers=[
                "#",
                "Class",
                "Confidence",
                "Track ID",
                "Box [X1, Y1, X2, Y2]",
                "Width (px)",
                "Height (px)",
            ],
            datatype=[
                "number",
                "str",
                "str",
                "str",
                "str",
                "number",
                "number",
            ],
            interactive=False,
            label="Detected Objects",
        )

        report = gr.Textbox(
            label="Metrics & Counts",
            lines=6,
        )

    with gr.Tab("🎥 Video Studio"):

        gr.Markdown(
            "Upload a video and process selected frames with the YOLO model."
        )

        video_input = gr.Video(
            label="Input Video",
        )

        with gr.Row():

            video_task = gr.Dropdown(
                choices=list(DEFAULT_TASK_MODELS.keys()),
                value="Object Detection",
                label="Vision Task",
            )

            video_model = gr.Dropdown(
                choices=["Auto (Recommended)"] + AVAILABLE_LOCAL_MODELS,
                value="Auto (Recommended)",
                label="Active Model",
            )

        with gr.Row():

            video_confidence = gr.Slider(
                minimum=0.05,
                maximum=0.95,
                value=0.25,
                step=0.05,
                label="Confidence",
            )

            video_iou = gr.Slider(
                minimum=0.10,
                maximum=0.90,
                value=0.45,
                step=0.05,
                label="IoU",
            )

            video_imgsz = gr.Dropdown(
                choices=[320, 480, 640],
                value=480,
                label="Inference Size",
            )

        with gr.Row():

            frame_skip = gr.Slider(
                minimum=1,
                maximum=10,
                value=2,
                step=1,
                label="Process Every Nth Frame",
            )

            max_frames = gr.Slider(
                minimum=10,
                maximum=1000,
                value=120,
                step=10,
                label="Maximum Frames",
            )

        video_button = gr.Button(
            "🎬 Start Video Analysis",
            variant="primary",
        )

        video_output = gr.Video(
            label="Annotated Video",
        )

        video_report = gr.Textbox(
            label="Video Metrics",
            lines=5,
        )

    with gr.Tab("⚙️ System Info"):

        gr.Markdown(
            f"""
            ### YOLO Vision Studio Pro

            **Application directory:** `{APP_DIR}`

            **Available local models:**
            {chr(10).join("- " + model for model in AVAILABLE_LOCAL_MODELS)}

            **Supported tasks:**
            - Object Detection
            - Object Counting
            - Customer & Retail
            - Object Tracking
            - Object Segmentation
            - Pose Estimation
            """
        )

    run_button.click(
        fn=run_image,
        inputs=[
            input_image,
            task,
            model_choice,
            confidence,
            iou_threshold,
            imgsz,
            selected_classes,
            show_count_banner,
        ],
        outputs=[
            output_image,
            result_table,
            report,
        ],
    )

    video_button.click(
        fn=run_video,
        inputs=[
            video_input,
            video_task,
            video_model,
            video_confidence,
            video_iou,
            video_imgsz,
            frame_skip,
            max_frames,
        ],
        outputs=[
            video_output,
            video_report,
        ],
    )


if __name__ == "__main__":
    demo.launch()
