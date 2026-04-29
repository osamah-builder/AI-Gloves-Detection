"""
Real Video Processing Module
============================
This module handles ACTUAL YOLOv8 inference on uploaded videos.
Use this when you want to demonstrate real detection (not mock data).

Setup:
1. pip install ultralytics opencv-python
2. Download a pre-trained gloves model (see README)
3. Replace MODEL_PATH below with your model path
"""

import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime, timedelta

# ============== CONFIGURATION ==============
MODEL_PATH = "models/gloves_yolov8.pt"  # Update this path after downloading model
DEFAULT_CONFIDENCE = 0.5

# Class mapping (adjust based on your downloaded model)
CLASS_MAPPING = {
    0: "gloved_hand",     # Compliant
    1: "bare_hand",       # Violation!
    # Add more classes if your model has them
}

VIOLATION_CLASSES = {"bare_hand"}  # Classes that count as violations


def load_model(model_path=MODEL_PATH):
    """Load YOLOv8 model for inference."""
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        return model
    except ImportError:
        raise ImportError("Install ultralytics: pip install ultralytics")
    except Exception as e:
        print(f"Could not load custom model. Falling back to YOLOv8n base model.")
        from ultralytics import YOLO
        return YOLO("yolov8n.pt")  # Falls back to base model


def process_frame(frame, model, confidence=DEFAULT_CONFIDENCE):
    """
    Run inference on a single frame.
    
    Returns:
        annotated_frame: Frame with bounding boxes drawn
        detections: List of dicts with detection info
    """
    results = model(frame, conf=confidence, verbose=False)
    detections = []
    annotated = frame.copy()
    
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = CLASS_MAPPING.get(cls_id, result.names.get(cls_id, "unknown"))
            
            is_violation = cls_name in VIOLATION_CLASSES
            color = (0, 0, 255) if is_violation else (0, 200, 0)  # BGR
            label = f"{cls_name} {conf:.2f}"
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            detections.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "class": cls_name,
                "confidence": round(conf, 3),
                "is_violation": is_violation
            })
    
    return annotated, detections


def process_video(video_path, output_path, model, confidence=DEFAULT_CONFIDENCE,
                  sample_every_n_frames=3, progress_callback=None):
    """
    Process entire video, save annotated output, return violation log.
    
    Args:
        video_path: Path to input video
        output_path: Where to save annotated video
        model: Loaded YOLO model
        confidence: Detection threshold
        sample_every_n_frames: Process every Nth frame (saves time)
        progress_callback: Function(percent, message) for progress updates
    
    Returns:
        violations: List of violation events
        stats: Dict with overall statistics
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    violations = []
    total_detections = 0
    total_violations = 0
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % sample_every_n_frames == 0:
            annotated, detections = process_frame(frame, model, confidence)
            
            for det in detections:
                total_detections += 1
                if det["is_violation"]:
                    total_violations += 1
                    timestamp_seconds = frame_idx / fps
                    violations.append({
                        "frame": frame_idx,
                        "timestamp": str(timedelta(seconds=int(timestamp_seconds))),
                        "class": det["class"],
                        "confidence": det["confidence"],
                        "bbox": det["bbox"]
                    })
            
            writer.write(annotated)
        else:
            writer.write(frame)
        
        frame_idx += 1
        
        if progress_callback and frame_idx % 30 == 0:
            pct = int((frame_idx / total_frames) * 100)
            progress_callback(pct, f"Processing frame {frame_idx}/{total_frames}")
    
    cap.release()
    writer.release()
    
    compliance_rate = 100.0
    if total_detections > 0:
        compliance_rate = round((1 - total_violations / total_detections) * 100, 1)
    
    stats = {
        "total_frames": total_frames,
        "fps": fps,
        "duration_seconds": round(total_frames / fps, 1),
        "total_detections": total_detections,
        "total_violations": total_violations,
        "compliance_rate": compliance_rate,
        "violations_count": len(violations)
    }
    
    return violations, stats


def export_violations_report(violations, stats, output_path):
    """Save violations to JSON for further processing."""
    report = {
        "generated_at": datetime.now().isoformat(),
        "statistics": stats,
        "violations": violations
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return output_path


# ============== USAGE EXAMPLE ==============
if __name__ == "__main__":
    print("Real Video Processing Module")
    print("=" * 50)
    print(f"Model path: {MODEL_PATH}")
    print(f"Default confidence: {DEFAULT_CONFIDENCE}")
    print()
    print("To use:")
    print("  from video_processor import load_model, process_video")
    print("  model = load_model()")
    print("  violations, stats = process_video('input.mp4', 'output.mp4', model)")
    print("  print(stats)")
