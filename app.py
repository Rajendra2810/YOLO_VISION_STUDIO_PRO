"""
YOLO Vision Studio Pro - Unified Interactive Computer Vision Web App
=====================================================================
Built for YOLOv8 & YOLO11:
- Object Detection (object_detection.py)
- Object Counting (object_counting.py)
- Customer & Retail Analytics (customer_detection.py)
- Object Tracking with Persistent IDs (object_tracking.py)
- Object Segmentation (object_segmentation.py)
- Human Pose Estimation (pose_estimation.py)

Designed for high responsiveness, real-time performance, and a modern UI.
"""

import io
import os
import time
import tempfile
import traceback
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Optional Gradio support
try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except Exception:
    gr = None
    GRADIO_AVAILABLE = False


# ============================================================
# APP CONFIGURATION & PATHS
# ============================================================

APP_DIR = Path(__file__).resolve().parent

DEFAULT_TASK_MODELS = {
    "Object Detection": "yolov8n.pt",
    "Object Counting": "yolov8n.pt",
    "Customer & Retail": "yolov8n.pt",
    "Object Tracking": "yolov8n.pt",
    "Object Segmentation": "yolov8n-seg.pt",
    "Pose Estimation": "yolov8n-pose.pt",
}

AVAILABLE_LOCAL_MODELS = [
    f.name for f in APP_DIR.glob("*.pt")
]
if not AVAILABLE_LOCAL_MODELS:
    AVAILABLE_LOCAL_MODELS = ["yolov8n.pt", "yolov8n-seg.pt", "yolov8n-pose.pt", "yolo11n.pt"]

st.set_page_config(
    page_title="YOLO Vision Studio Pro",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM RESPONSIVE CSS STYLING
# ============================================================

st.markdown(
    """
    <style>
    /* Global layout enhancements */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 96%;
    }

    /* Header Banner */
    .studio-hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #312e81 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px 30px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        color: #f8fafc;
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        align-items: center;
        gap: 16px;
    }
    .studio-hero h1 {
        margin: 0;
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .studio-hero p {
        margin: 6px 0 0 0;
        font-size: 0.95rem;
        color: #cbd5e1;
    }
    .hero-badges {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
    }
    .hero-badge {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 20px;
        padding: 5px 14px;
        font-size: 0.8rem;
        font-weight: 600;
        color: #e2e8f0;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .hero-badge.live {
        background: rgba(34, 197, 94, 0.15);
        border-color: rgba(34, 197, 94, 0.35);
        color: #4ade80;
    }

    /* KPI Metric Cards */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
        gap: 16px;
        margin-bottom: 24px;
    }
    .kpi-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 16px 20px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        backdrop-filter: blur(8px);
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
        border-color: rgba(99, 102, 241, 0.4);
    }
    .kpi-title {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #f1f5f9;
        line-height: 1.1;
    }
    .kpi-caption {
        font-size: 0.78rem;
        color: #64748b;
        margin-top: 4px;
    }

    /* Modern Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(255, 255, 255, 0.03);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.92rem;
        padding: 0 18px;
        color: #94a3b8;
        border: none;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4f46e5 !important;
        color: #ffffff !important;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.35);
    }

    /* Class Chips */
    .chip-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin: 12px 0;
    }
    .class-chip {
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.3);
        color: #c7d2fe;
        border-radius: 16px;
        padding: 4px 12px;
        font-size: 0.82rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .class-chip-count {
        background: #4f46e5;
        color: white;
        border-radius: 10px;
        padding: 1px 7px;
        font-size: 0.75rem;
    }

    /* Custom alert boxes */
    .capacity-alert-ok {
        background: rgba(34, 197, 94, 0.12);
        border: 1px solid rgba(34, 197, 94, 0.3);
        border-radius: 10px;
        padding: 12px 16px;
        color: #86efac;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 10px;
        margin-top: 10px;
    }
    .capacity-alert-warn {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(239, 68, 68, 0.4);
        border-radius: 10px;
        padding: 12px 16px;
        color: #fca5a5;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 10px;
        margin-top: 10px;
    }

    /* Responsive adjustments */
    @media (max-width: 768px) {
        .studio-hero {
            padding: 16px 20px;
        }
        .studio-hero h1 {
            font-size: 1.6rem;
        }
        .kpi-container {
            grid-template-columns: 1fr 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MODEL LOADER & CACHE
# ============================================================

@st.cache_resource(show_spinner=False)
def load_yolo_model(model_identifier: str):
    """Load a YOLO model once into memory."""
    path = Path(model_identifier)

    # If it's a relative filename in APP_DIR
    if not path.is_absolute() and (APP_DIR / model_identifier).exists():
        return YOLO(str(APP_DIR / model_identifier))

    if path.exists():
        return YOLO(str(path))

    # Fallback to standard Ultralytics model naming (auto-downloads if needed)
    return YOLO(model_identifier)


def resolve_model_for_task(task: str, user_override_model: str = "Auto (Recommended)"):
    """Determine the optimal YOLO model based on task & user preference."""
    if user_override_model != "Auto (Recommended)":
        return load_yolo_model(user_override_model)

    default_file = DEFAULT_TASK_MODELS.get(task, "yolov8n.pt")
    
    # Check if local model file exists, else use standard name
    target_path = APP_DIR / default_file
    if target_path.exists():
        return load_yolo_model(str(target_path))

    return load_yolo_model(default_file)


# ============================================================
# CORE VISION INFERENCE PIPELINE
# ============================================================

def normalize_input_image(image_input):
    """Safely convert PIL images, uploads, or arrays into RGB uint8 NumPy array."""
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
    task: str,
    model_choice: str = "Auto (Recommended)",
    confidence: float = 0.25,
    iou_threshold: float = 0.45,
    imgsz: int = 640,
    allowed_classes: list = None,
    show_count_banner: bool = True,
):
    """
    Executes YOLO inference with high responsiveness, latency metrics,
    structured detections dataframe, and annotated visualization.
    """
    t0 = time.perf_counter()
    image_rgb = normalize_input_image(image)

    if image_rgb is None:
        return None, None, {}, 0.0, "No valid image provided."

    # Convert to BGR for OpenCV / YOLO processing
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    try:
        model = resolve_model_for_task(task, model_choice)
    except Exception as exc:
        return None, None, {}, 0.0, f"Error loading model: {exc}"

    # Class filter index resolution
    class_indices = None
    if allowed_classes and len(allowed_classes) > 0 and hasattr(model, "names"):
        name_to_id = {v: k for k, v in model.names.items()}
        class_indices = [name_to_id[c] for c in allowed_classes if c in name_to_id]

    # Customer detection naturally focuses on the 'person' class
    if task == "Customer & Retail" and not class_indices:
        if hasattr(model, "names"):
            name_to_id = {v: k for k, v in model.names.items()}
            if "person" in name_to_id:
                class_indices = [name_to_id["person"]]

    # Run Prediction or Tracking
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
        return None, None, {}, 0.0, f"Inference execution failed: {exc}"

    total_time_ms = (time.perf_counter() - t0) * 1000.0

    if not results or len(results) == 0:
        return image_rgb, pd.DataFrame(), {}, total_time_ms, "No results returned."

    result = results[0]

    # Generate annotated BGR image from Ultralytics
    annotated_bgr = result.plot()
    if annotated_bgr is None:
        annotated_bgr = image_bgr.copy()

    # Extract Detection Dictionaries & Metrics
    boxes = getattr(result, "boxes", None)
    detections_data = []
    class_counts = {}

    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        
        # Check for Tracking IDs
        track_ids = None
        if hasattr(boxes, "id") and boxes.id is not None:
            try:
                track_ids = boxes.id.cpu().numpy().astype(int)
            except Exception:
                track_ids = None

        names = getattr(result, "names", {})

        for i in range(len(boxes)):
            cls_id = int(clss[i])
            cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
            conf_val = float(confs[i])
            x1, y1, x2, y2 = map(int, xyxy[i])
            t_id = int(track_ids[i]) if track_ids is not None else "-"

            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

            detections_data.append({
                "#": i + 1,
                "Class": cls_name,
                "Confidence": f"{conf_val * 100:.1f}%",
                "Track ID": t_id,
                "Box [X1, Y1, X2, Y2]": f"[{x1}, {y1}, {x2}, {y2}]",
                "Width (px)": x2 - x1,
                "Height (px)": y2 - y1,
            })

    # Object Counting Overlay from object_counting.py
    total_detected = len(detections_data)
    if show_count_banner and total_detected > 0:
        # Draw sleek counting badge in upper left
        banner_text = f"Count: {total_detected}"
        if task == "Customer & Retail":
            banner_text = f"Customers: {total_detected}"
        
        (w, h), _ = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(annotated_bgr, (12, 12), (18 + w + 10, 20 + h + 10), (15, 23, 42), -1)
        cv2.rectangle(annotated_bgr, (12, 12), (18 + w + 10, 20 + h + 10), (99, 102, 241), 2)
        cv2.putText(
            annotated_bgr,
            banner_text,
            (18, 20 + h),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (52, 211, 153),
            2,
            cv2.LINE_AA,
        )

    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    df_detections = pd.DataFrame(detections_data)

    return annotated_rgb, df_detections, class_counts, total_time_ms, "Success"


# ============================================================
# SIDEBAR CONTROLS & MODEL MANAGEMENT
# ============================================================

with st.sidebar:
    st.markdown("### ⚙️ Vision Studio Controls")

    task = st.selectbox(
        "Select Vision Task",
        list(DEFAULT_TASK_MODELS.keys()),
        index=0,
        help="Select any computer vision task from the project repository.",
    )

    st.markdown("---")
    st.markdown("#### 🧠 Model & Precision")

    model_options = ["Auto (Recommended)"] + AVAILABLE_LOCAL_MODELS
    selected_model_file = st.selectbox(
        "Active Model Weight",
        model_options,
        index=0,
        help="Choose automatic selection or switch to local models like YOLO11 nano or custom weights.",
    )

    col_res1, col_res2 = st.columns(2)
    with col_res1:
        imgsz = st.select_slider(
            "Inference Size",
            options=[320, 480, 640],
            value=480,
            help="320px gives ultra-responsive frame rates on CPU; 640px gives maximum accuracy.",
        )
    with col_res2:
        iou_thresh = st.slider(
            "NMS IoU",
            min_value=0.1,
            max_value=0.9,
            value=0.45,
            step=0.05,
            help="Intersection over Union threshold for Non-Max Suppression.",
        )

    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05,
        help="Minimum confidence required for detected objects.",
    )

    # Class filter options
    st.markdown("#### 🔍 Filter & Overlays")
    show_counter_overlay = st.checkbox("Overlay Count Banner on Image", value=True)

    # Pre-populate common class names
    common_classes = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "backpack", "umbrella", "handbag", "bottle",
        "chair", "couch", "potted plant", "bed", "dining table", "laptop", "cell phone"
    ]
    selected_filter_classes = st.multiselect(
        "Filter Specific Classes",
        options=common_classes,
        default=[],
        help="Leave empty to detect all 80 COCO classes.",
    )

    st.markdown("---")
    st.markdown("#### 📂 Local Model Repository")
    for mf in AVAILABLE_LOCAL_MODELS:
        file_path = APP_DIR / mf
        sz_mb = file_path.stat().st_size / (1024 * 1024) if file_path.exists() else 0
        st.caption(f"📁 `{mf}` ({sz_mb:.1f} MB)")


# ============================================================
# HERO HEADER SECTION
# ============================================================

st.markdown(
    f"""
    <div class="studio-hero">
        <div>
            <h1>🎯 YOLO Vision Studio Pro</h1>
            <p>High-Performance Unified Web Platform for Real-Time YOLO Computer Vision</p>
        </div>
        <div class="hero-badges">
            <span class="hero-badge live">● READY</span>
            <span class="hero-badge">Task: {task}</span>
            <span class="hero-badge">Model: {DEFAULT_TASK_MODELS.get(task, 'Custom')}</span>
            <span class="hero-badge">Res: {imgsz}px</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MAIN INTERFACE TABS
# ============================================================

tab_image, tab_video, tab_webcam, tab_customer, tab_models, tab_gradio = st.tabs([
    "🖼️ Image Studio",
    "🎥 Video Studio",
    "📹 Live Camera",
    "👥 Customer & Retail",
    "⚙️ System Diagnostics",
    "🤗 Gradio Portal",
])


# ============================================================
# TAB 1: IMAGE STUDIO
# ============================================================

with tab_image:
    col_upload, col_actions = st.columns([3, 1])

    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload Image (JPG, PNG, JPEG, WEBP)",
            type=["jpg", "jpeg", "png", "webp"],
            key="img_uploader",
        )

    with col_actions:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        sample_choice = st.selectbox(
            "Or Pick Sample Image",
            ["None", "Demo City / Street", "Demo Living Room"],
            help="Quickly test without uploading your own file.",
        )

    target_pil = None
    if uploaded_file is not None:
        target_pil = Image.open(uploaded_file).convert("RGB")
    elif sample_choice == "Demo City / Street":
        # Generate clean synthetic test image
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        canvas[:] = (45, 55, 72)
        cv2.putText(canvas, "Street Simulation", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        target_pil = Image.fromarray(canvas)

    if target_pil is not None:
        col_view1, col_view2 = st.columns(2)

        with col_view1:
            st.markdown("##### 📥 Input Image")
            st.image(target_pil, use_container_width=True)

        with col_view2:
            st.markdown("##### 🚀 Detection Output")
            with st.spinner("Executing YOLO Inference..."):
                ann_rgb, df_res, class_counts, latency_ms, status_msg = execute_yolo_inference(
                    target_pil,
                    task=task,
                    model_choice=selected_model_file,
                    confidence=confidence,
                    iou_threshold=iou_thresh,
                    imgsz=imgsz,
                    allowed_classes=selected_filter_classes,
                    show_count_banner=show_counter_overlay,
                )

            if ann_rgb is not None:
                st.image(ann_rgb, use_container_width=True)

        # KPI Dashboard Metrics
        total_objects = len(df_res) if df_res is not None else 0
        top_class = max(class_counts, key=class_counts.get) if class_counts else "None"

        st.markdown(
            f"""
            <div class="kpi-container" style="margin-top: 20px;">
                <div class="kpi-card">
                    <div class="kpi-title">Total Objects Detected</div>
                    <div class="kpi-value">{total_objects}</div>
                    <div class="kpi-caption">Active detections in image</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-title">Inference Latency</div>
                    <div class="kpi-value">{latency_ms:.1f} <span style="font-size: 1rem; color: #94a3b8;">ms</span></div>
                    <div class="kpi-caption">Pre + Predict + Post</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-title">Unique Classes</div>
                    <div class="kpi-value">{len(class_counts)}</div>
                    <div class="kpi-caption">Distinct detected categories</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-title">Top Detected Class</div>
                    <div class="kpi-value" style="font-size: 1.4rem;">{top_class}</div>
                    <div class="kpi-caption">{class_counts.get(top_class, 0)} instances found</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Detected Classes Badges
        if class_counts:
            chips_html = "".join(
                f'<span class="class-chip">{name} <span class="class-chip-count">{count}</span></span>'
                for name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
            )
            st.markdown(f"**Detected Categories Breakdown:**<div class='chip-container'>{chips_html}</div>", unsafe_allow_html=True)

        # Detailed Tabular Data & Downloads
        col_table, col_down = st.columns([3, 1])

        with col_table:
            if df_res is not None and not df_res.empty:
                with st.expander("📋 Detailed Object Coordinates & Confidence Table", expanded=True):
                    st.dataframe(df_res, use_container_width=True, height=220)
            else:
                st.info("No objects met the current confidence threshold.")

        with col_down:
            if ann_rgb is not None:
                # Prepare Image Download
                pil_out = Image.fromarray(ann_rgb)
                buf_img = io.BytesIO()
                pil_out.save(buf_img, format="PNG")
                byte_img = buf_img.getvalue()

                st.download_button(
                    label="💾 Download Result Image",
                    data=byte_img,
                    file_name=f"yolo_{task.lower().replace(' ', '_')}_result.png",
                    mime="image/png",
                    use_container_width=True,
                )

            if df_res is not None and not df_res.empty:
                csv_bytes = df_res.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📊 Download Detections (CSV)",
                    data=csv_bytes,
                    file_name="detections_report.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
    else:
        st.info("👆 Please upload an image or select a demo picture above to begin.")


# ============================================================
# TAB 2: VIDEO STUDIO
# ============================================================

with tab_video:
    st.subheader("🎥 Video File Processing")
    st.caption("Upload video clips (MP4, AVI, MOV) for sequential frame detection and tracking.")

    col_vid_in, col_vid_opt = st.columns([2, 1])

    with col_vid_in:
        uploaded_video = st.file_uploader(
            "Upload Video Clip",
            type=["mp4", "avi", "mov", "mkv"],
            key="video_uploader",
        )

    with col_vid_opt:
        frame_skip = st.slider(
            "Frame Processing Interval",
            min_value=1,
            max_value=10,
            value=2,
            help="Process every Nth frame to accelerate throughput.",
        )
        max_frames_to_process = st.number_input(
            "Maximum Frames to Analyze",
            min_value=10,
            max_value=1000,
            value=120,
            step=20,
        )

    if uploaded_video is not None:
        # Save to temporary file for OpenCV VideoCapture
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        tfile.flush()
        video_path = tfile.name

        cap = cv2.VideoCapture(video_path)
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        st.caption(f"ℹ️ Video Details: {total_video_frames} total frames @ {video_fps:.1f} FPS")

        if st.button("🚀 Start Video Analysis", type="primary", key="btn_start_video"):
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            preview_placeholder = st.empty()

            frame_idx = 0
            processed_count = 0
            all_detected_classes = {}
            total_latency = 0.0

            start_proc_time = time.time()

            while cap.isOpened() and processed_count < max_frames_to_process:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                if frame_idx % frame_skip != 0:
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ann_rgb, df_f, counts, lat_ms, _ = execute_yolo_inference(
                    frame_rgb,
                    task=task,
                    model_choice=selected_model_file,
                    confidence=confidence,
                    iou_threshold=iou_thresh,
                    imgsz=imgsz,
                    allowed_classes=selected_filter_classes,
                    show_count_banner=show_counter_overlay,
                )

                processed_count += 1
                total_latency += lat_ms

                for k, v in counts.items():
                    all_detected_classes[k] = all_detected_classes.get(k, 0) + v

                # Live preview
                preview_placeholder.image(ann_rgb, caption=f"Frame #{frame_idx} • Latency: {lat_ms:.1f}ms", use_container_width=True)

                progress = min(1.0, processed_count / max_frames_to_process)
                progress_bar.progress(progress)
                status_text.text(f"Processing frame {processed_count}/{max_frames_to_process}...")

            cap.release()
            try:
                os.remove(video_path)
            except Exception:
                pass

            elapsed = time.time() - start_proc_time
            avg_fps = processed_count / elapsed if elapsed > 0 else 0

            st.success(f"✅ Video analysis complete! Processed {processed_count} frames in {elapsed:.1f}s ({avg_fps:.1f} FPS).")

            # Summary metrics
            st.markdown(
                f"""
                <div class="kpi-container">
                    <div class="kpi-card">
                        <div class="kpi-title">Frames Analyzed</div>
                        <div class="kpi-value">{processed_count}</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-title">Average Processing Speed</div>
                        <div class="kpi-value">{avg_fps:.1f} <span style="font-size: 1rem; color: #94a3b8;">FPS</span></div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-title">Total Object Sightings</div>
                        <div class="kpi-value">{sum(all_detected_classes.values())}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# TAB 3: LIVE CAMERA & SNAPSHOT
# ============================================================

with tab_webcam:
    st.subheader("📹 Real-Time Webcam Stream & Snapshot")
    st.caption("Directly test YOLO on your live camera feed, replicating the standalone OpenCV scripts.")

    stream_col, snap_col = st.columns([1, 1])

    with stream_col:
        st.markdown("#### ⚡ Live Stream Loop")
        st.write("Run real-time video stream with live OpenCV detection directly inside your browser.")

        start_stream = st.checkbox("▶️ Start Live Webcam Stream", key="webcam_stream_active")
        camera_index = st.number_input("Webcam Device Index", min_value=0, max_value=5, value=0, step=1)

        webcam_view = st.empty()
        webcam_metrics = st.empty()

        if start_stream:
            cap_stream = cv2.VideoCapture(int(camera_index))

            if not cap_stream.isOpened():
                st.error(f"❌ Could not access webcam at device index {camera_index}. Please check device permissions.")
            else:
                stop_btn = st.button("⏹️ Stop Stream", key="btn_stop_stream")
                frame_counter = 0
                fps_time = time.time()

                while cap_stream.isOpened() and not stop_btn:
                    ret, frame = cap_stream.read()
                    if not ret:
                        st.warning("Webcam feed disconnected.")
                        break

                    frame_counter += 1
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    ann_rgb, df_det, counts, lat_ms, _ = execute_yolo_inference(
                        frame_rgb,
                        task=task,
                        model_choice=selected_model_file,
                        confidence=confidence,
                        iou_threshold=iou_thresh,
                        imgsz=imgsz,
                        allowed_classes=selected_filter_classes,
                        show_count_banner=show_counter_overlay,
                    )

                    webcam_view.image(ann_rgb, channels="RGB", use_container_width=True)

                    if frame_counter % 5 == 0:
                        now = time.time()
                        fps = 5.0 / (now - fps_time)
                        fps_time = now
                        webcam_metrics.caption(
                            f"Live Rate: **{fps:.1f} FPS** | Latency: **{lat_ms:.1f} ms** | Objects: **{len(df_det)}**"
                        )

                cap_stream.release()

    with snap_col:
        st.markdown("#### 📸 Camera Snapshot")
        st.write("Capture a single snapshot using your browser's native webcam interface.")

        cam_snapshot = st.camera_input("Take a photo", key="camera_snap_input")

        if cam_snapshot is not None:
            snap_pil = Image.open(cam_snapshot).convert("RGB")
            with st.spinner("Processing Snapshot..."):
                snap_ann_rgb, snap_df, snap_counts, snap_lat, _ = execute_yolo_inference(
                    snap_pil,
                    task=task,
                    model_choice=selected_model_file,
                    confidence=confidence,
                    iou_threshold=iou_thresh,
                    imgsz=imgsz,
                    allowed_classes=selected_filter_classes,
                    show_count_banner=show_counter_overlay,
                )

            st.image(snap_ann_rgb, caption=f"Snapshot Result ({snap_lat:.1f}ms)", use_container_width=True)
            if snap_counts:
                st.write("**Detected in snapshot:**", snap_counts)


# ============================================================
# TAB 4: CUSTOMER & RETAIL ANALYTICS
# ============================================================

with tab_customer:
    st.subheader("👥 Customer Detection & Footfall Analytics")
    st.caption("Dedicated retail monitoring module derived from customer_detection.py and store analytics.")

    col_cust_cfg, col_cust_view = st.columns([1, 2])

    with col_cust_cfg:
        st.markdown("##### 🏪 Store Occupancy Settings")
        max_occupancy = st.slider("Maximum Store Capacity", min_value=1, max_value=50, value=5)
        st.info("This module automatically isolates human detections ('person') to track customer flow, occupancy, and density.")

        cust_upload = st.file_uploader(
            "Upload Store / Entrance Image",
            type=["jpg", "jpeg", "png", "webp"],
            key="customer_uploader",
        )

    with col_cust_view:
        if cust_upload is not None:
            c_pil = Image.open(cust_upload).convert("RGB")
            c_ann, c_df, c_counts, c_lat, _ = execute_yolo_inference(
                c_pil,
                task="Customer & Retail",
                model_choice=selected_model_file,
                confidence=confidence,
                iou_threshold=iou_thresh,
                imgsz=imgsz,
                allowed_classes=["person"],
                show_count_banner=True,
            )

            st.image(c_ann, caption="Customer Monitoring Result", use_container_width=True)

            current_customers = c_counts.get("person", 0)

            # Capacity status alert
            if current_customers > max_occupancy:
                st.markdown(
                    f"""
                    <div class="capacity-alert-warn">
                        ⚠️ <strong>OCCUPANCY WARNING:</strong> Store capacity exceeded! 
                        Currently {current_customers} customers detected (Max limit: {max_occupancy}).
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="capacity-alert-ok">
                        ✅ <strong>NORMAL OCCUPANCY:</strong> {current_customers} active customers 
                        within the safe limit of {max_occupancy}.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("Upload a retail or entrance image on the left to inspect customer detection and capacity monitoring.")


# ============================================================
# TAB 5: SYSTEM DIAGNOSTICS & HARDWARE HUB
# ============================================================

with tab_models:
    st.subheader("⚙️ System Diagnostics & Model Inspector")
    st.caption("Real-time hardware status, model weights, and Ultralytics engine information.")

    col_sys1, col_sys2 = st.columns(2)

    with col_sys1:
        st.markdown("##### 💻 Compute Environment")
        import torch
        cuda_avail = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU (Multi-threaded)"

        st.markdown(
            f"""
            - **Torch Version:** `{torch.__version__}`
            - **Hardware Acceleration:** `{'CUDA / GPU' if cuda_avail else 'CPU'}`
            - **Active Device:** `{device_name}`
            - **OpenCV Version:** `{cv2.__version__}`
            - **Workspace Root:** `{APP_DIR}`
            """
        )

    with col_sys2:
        st.markdown("##### 📦 Discovered Model Files")
        models_data = []
        for mf in sorted(AVAILABLE_LOCAL_MODELS):
            fp = APP_DIR / mf
            models_data.append({
                "Model File": mf,
                "Size (MB)": f"{fp.stat().st_size / (1024 * 1024):.2f}" if fp.exists() else "Missing",
                "Task Alignment": [k for k, v in DEFAULT_TASK_MODELS.items() if v == mf] or "Custom / General",
            })
        st.dataframe(pd.DataFrame(models_data), use_container_width=True)


# ============================================================
# TAB 6: GRADIO PORTAL
# ============================================================

with tab_gradio:
    st.subheader("🤗 Gradio Alternative Interface")
    st.caption("Launch a standalone Gradio server for lightweight secondary workflows.")

    if not GRADIO_AVAILABLE:
        st.warning("Gradio is not installed in the current environment.")
    else:
        st.success("Gradio is available.")

        def gradio_vision_pipeline(img, selected_task, conf_val):
            ann, _, counts, lat, _ = execute_yolo_inference(
                img,
                task=selected_task,
                confidence=conf_val,
            )
            report = f"Task: {selected_task}\nLatency: {lat:.1f}ms\nCounts: {counts}"
            return ann, report

        if st.button("▶️ Launch Gradio Server (Port 7860)", key="btn_gradio_launch"):
            try:
                demo = gr.Interface(
                    fn=gradio_vision_pipeline,
                    inputs=[
                        gr.Image(type="numpy", label="Input Image"),
                        gr.Dropdown(choices=list(DEFAULT_TASK_MODELS.keys()), value="Object Detection", label="Task"),
                        gr.Slider(0.05, 0.95, value=0.25, label="Confidence"),
                    ],
                    outputs=[
                        gr.Image(type="numpy", label="YOLO Result"),
                        gr.Textbox(label="Metrics & Counts"),
                    ],
                    title="YOLO Vision Studio (Gradio UI)",
                )
                demo.launch(
                    server_name="127.0.0.1",
                    server_port=7860,
                    prevent_thread_lock=True,
                    share=False,
                )
                st.success("Gradio started at http://127.0.0.1:7860")
            except Exception as e:
                st.error(f"Gradio launch notification: {e}")


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.caption(
    "🎯 YOLO Vision Studio Pro • Unified Suite covering Detection, Counting, Tracking, "
    "Segmentation, Pose Estimation & Customer Analytics."
)
