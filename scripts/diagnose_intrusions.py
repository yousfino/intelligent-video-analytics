
import argparse
from pathlib import Path

import cv2
import numpy as np

from video_analytics.config import load_config
from video_analytics.detectors.factory import create_detector
from video_analytics.tracker import ByteTrackTracker
from video_analytics.zones import RestrictedZone


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/pipeline.docker-gpu.yaml",
    )

    parser.add_argument(
        "--source",
        default="inputs/test_video.mp4",
    )

    parser.add_argument(
        "--model",
        choices=["yolo", "rfdetr"],
        default="yolo",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--tracking-threshold",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--output",
        default="outputs/diagnostics/intrusions.mp4",
    )

    args = parser.parse_args()

    config = load_config(args.config)
    config["detector"]["model"] = args.model

    detector = create_detector(config)

    tracking = config["tracking"]

    threshold = (
        args.tracking_threshold
        if args.tracking_threshold is not None
        else tracking["track_activation_threshold"]
    )

    tracker = ByteTrackTracker(
        frame_rate=tracking["frame_rate"],
        track_activation_threshold=threshold,
        lost_track_buffer=tracking["lost_track_buffer"],
        minimum_matching_threshold=tracking[
            "minimum_matching_threshold"
        ],
    )

    zone = RestrictedZone.from_file(
        config["zones"]["config_path"]
    )

    capture = cv2.VideoCapture(args.source)

    if not capture.isOpened():
        raise RuntimeError("Unable to open video source")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        capture.release()
        raise RuntimeError("Unable to create output video")

    frames = 0
    total_detections = 0
    frames_with_detections = 0
    total_tracks = 0
    frames_with_tracks = 0
    frames_inside_zone = 0
    intrusion_events = 0
    unique_track_ids = set()
    confidence_scores = []

    try:
        while frames < args.max_frames:
            ok, frame = capture.read()

            if not ok:
                break

            frames += 1

            detections = detector.detect(frame)
            detection_count = len(detections.boxes)

            total_detections += detection_count

            if detection_count:
                frames_with_detections += 1
                confidence_scores.extend(
                    detections.scores.tolist()
                )

            tracks = tracker.update(detections)

            track_count = len(tracks)
            total_tracks += track_count

            if track_count:
                frames_with_tracks += 1

            polygon = zone.polygon(width, height)

            cv2.polylines(
                frame,
                [polygon],
                True,
                (0, 0, 255),
                2,
            )

            # Raw detections: yellow
            for box, score in zip(
                detections.boxes,
                detections.scores,
            ):
                x1, y1, x2, y2 = map(int, box)

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 255),
                    1,
                )

                cv2.putText(
                    frame,
                    f"Det: {score:.2f}",
                    (x1, max(15, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                )

            inside_ids = set()

            if tracks.tracker_id is not None:
                for box, track_id in zip(
                    tracks.xyxy,
                    tracks.tracker_id,
                ):
                    track_id = int(track_id)
                    unique_track_ids.add(track_id)

                    x1, y1, x2, y2 = map(int, box)

                    foot_point = (
                        int((x1 + x2) / 2),
                        y2,
                    )

                    inside = zone.contains(
                        foot_point,
                        width,
                        height,
                    )

                    if inside:
                        inside_ids.add(track_id)

                    color = (
                        (0, 0, 255)
                        if inside
                        else (0, 255, 0)
                    )

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        color,
                        2,
                    )

                    cv2.circle(
                        frame,
                        foot_point,
                        5,
                        color,
                        -1,
                    )

                    cv2.putText(
                        frame,
                        f"ID: {track_id}",
                        (x1, y2 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                    )

            if inside_ids:
                frames_inside_zone += 1

            entered_ids = zone.update(inside_ids)
            intrusion_events += len(entered_ids)

            cv2.putText(
                frame,
                (
                    f"Frame: {frames} "
                    f"Det: {detection_count} "
                    f"Tracks: {track_count} "
                    f"Events: {intrusion_events}"
                ),
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            writer.write(frame)

    finally:
        capture.release()
        writer.release()

    print("\n--- Intrusion diagnostic ---")
    print(f"Model: {args.model}")
    print(f"Tracking threshold: {threshold}")
    print(f"Frames inspected: {frames}")
    print(f"Frames with detections: {frames_with_detections}")
    print(f"Total detections: {total_detections}")
    print(f"Frames with tracks: {frames_with_tracks}")
    print(f"Total track observations: {total_tracks}")
    print(f"Unique track IDs: {len(unique_track_ids)}")
    print(f"Frames inside zone: {frames_inside_zone}")
    print(f"Intrusion events: {intrusion_events}")

    if confidence_scores:
        scores = np.array(confidence_scores)

        print(f"Minimum confidence: {scores.min():.4f}")
        print(f"Mean confidence: {scores.mean():.4f}")
        print(f"Maximum confidence: {scores.max():.4f}")

        print(
            "Detections meeting tracking threshold: "
            f"{np.sum(scores >= threshold)}"
        )

    print(f"Annotated video: {output}")


if __name__ == "__main__":
    main()
