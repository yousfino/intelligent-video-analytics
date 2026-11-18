
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

try:
    from .zones import RestrictedZone
except ImportError:
    from zones import RestrictedZone


def log_event(
    path: Path,
    zone: str,
    track_id: int,
    frame_number: int,
):
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": "zone_intrusion",
        "zone": zone,
        "track_id": track_id,
        "frame_number": frame_number,
    }

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")

    print(f"[ALERT] Track {track_id} entered {zone}")


def run_tracking(source: int | str, zone_path: str):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available")

    model = YOLO("yolo11n.pt")
    zone = RestrictedZone.from_file(zone_path)

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    event_path = Path("outputs/events/intrusions.jsonl")
    frame_number = 0
    previous_time = time.perf_counter()

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame_number += 1
            height, width = frame.shape[:2]

            results = model.track(
                source=frame,
                device=0,
                tracker="bytetrack.yaml",
                persist=True,
                classes=[0],
                conf=0.10,
                imgsz=640,
                verbose=False,
            )

            result = results[0]
            annotated = result.plot()

            polygon = zone.polygon(width, height)

            overlay = annotated.copy()
            cv2.fillPoly(
                overlay,
                [polygon],
                (0, 0, 255),
            )
            annotated = cv2.addWeighted(
                overlay, 0.15, annotated, 0.85, 0
            )

            cv2.polylines(
                annotated,
                [polygon],
                True,
                (0, 0, 255),
                2,
            )

            inside_ids = set()
            tracked_count = 0

            if (
                result.boxes is not None
                and result.boxes.id is not None
            ):
                boxes = result.boxes.xyxy.cpu().numpy()
                ids = result.boxes.id.int().cpu().tolist()

                tracked_count = len(ids)

                for box, track_id in zip(boxes, ids):
                    x1, y1, x2, y2 = map(int, box)

                    foot_point = (
                        int((x1 + x2) / 2),
                        y2,
                    )

                    is_inside = zone.contains(
                        foot_point, width, height
                    )

                    if is_inside:
                        inside_ids.add(track_id)

                    color = (
                        (0, 0, 255) if is_inside
                        else (0, 255, 0)
                    )

                    cv2.circle(
                        annotated, foot_point, 5, color, -1
                    )

            entered_ids = zone.update(inside_ids)

            for track_id in sorted(entered_ids):
                log_event(
                    event_path,
                    zone.name,
                    track_id,
                    frame_number,
                )

            now = time.perf_counter()
            fps = 1.0 / max(now - previous_time, 1e-6)
            previous_time = now

            status = (
                "INTRUSION DETECTED"
                if inside_ids else "MONITORING"
            )

            cv2.putText(
                annotated,
                f"FPS: {fps:.1f} | Tracks: {tracked_count}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                annotated,
                status,
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255) if inside_ids
                else (0, 255, 0),
                2,
            )

            cv2.imshow(
                "Intelligent Video Analytics - Tracking",
                annotated,
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0")
    parser.add_argument(
        "--zones",
        default="configs/zones.json",
    )

    args = parser.parse_args()
    source = (
        int(args.source)
        if args.source.isdigit()
        else args.source
    )

    run_tracking(source, args.zones)
