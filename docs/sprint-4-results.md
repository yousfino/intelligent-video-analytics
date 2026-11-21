# Sprint 4 — Real-Time Video Analytics Integration

**Project:** Intelligent Video Analytics  
**Status:** Implementation and local validation complete; Git merge pending  
**Validation date:** 2026-11-21  
**Development environment:** Windows, Python 3.11, CUDA-capable NVIDIA GPU

## Summary

Sprint 4 integrated the previously trained YOLO11n and RF-DETR Nano person detectors into one configurable real-time video analytics application. The pipeline performs object detection, ByteTrack tracking, polygon-based restricted-zone intrusion detection, JSONL event logging, and per-stage performance monitoring.

Both models were exercised successfully through the refactored application entry point. The full automated test suite passed: **14 passed, 1 warning, 0 failures**.

## Deliverables

| Work item | Implementation                                                | Result   |
| --------- | ------------------------------------------------------------- | -------- |
| 4.1       | Common `BaseDetector` and `DetectionResult` interface         | Complete |
| 4.2       | YAML configuration and detector factory                       | Complete |
| 4.3       | Model-independent ByteTrack adapter                           | Complete |
| 4.4       | Restricted-zone intrusion detection using tracked foot points | Complete |
| 4.5       | JSONL intrusion event logging                                 | Complete |
| 4.6       | Detection, tracking, and analytics latency monitoring         | Complete |
| 4.7       | Reusable `AnalyticsPipeline` and integration tests            | Complete |

## Architecture

```text
Webcam / video file
       |
       v
Configurable detector (YOLO11n or RF-DETR Nano)
       |
       v
DetectionResult (xyxy boxes, scores, class IDs)
       |
       v
ByteTrack (persistent track IDs)
       |
       v
RestrictedZone (foot-point / polygon entry)
       |
       +----> JSONL intrusion events
       |
       v
Annotated video + latency / FPS overlay
```

### Key files

- `configs/pipeline.yaml` — detector, source, tracker, zone, and event configuration
- `src/video_analytics/config.py` — YAML configuration loader
- `src/video_analytics/detectors/base.py` — common detection contract
- `src/video_analytics/detectors/yolo_detector.py` — YOLO adapter
- `src/video_analytics/detectors/rfdetr_detector.py` — RF-DETR adapter
- `src/video_analytics/detectors/factory.py` — model selection
- `src/video_analytics/tracker.py` — ByteTrack adapter
- `src/video_analytics/zones.py` — polygon-based intrusion state
- `src/video_analytics/events.py` — JSONL event logger
- `src/video_analytics/monitoring.py` — rolling performance metrics
- `src/video_analytics/pipeline.py` — reusable end-to-end processing logic
- `scripts/run_pipeline.py` — live application entry point

## Validation

### Automated tests

Command:

```powershell
$env:PYTHONPATH = "src"
python -m pytest tests/ -v
```

Observed on 2026-11-21:

```text
14 passed, 1 warning in 3.14s
```

Tests cover detection result validation, event logging, performance monitoring, tracker behavior, zone geometry and entry-state handling, and pipeline integration (single intrusion event and empty detections).

### Live model checks

```powershell
python scripts/run_pipeline.py --model yolo
python scripts/run_pipeline.py --model rfdetr
```

Both commands ran successfully according to local validation. These checks confirm that both model paths execute, but no quantitative live FPS, end-to-end latency distribution, or long-duration stability measurements were recorded for Sprint 4.

## Runtime behavior

- Both detectors expose a consistent person-detection result schema.
- The tracker associates detections with persistent IDs across frames.
- Intrusion detection uses the bottom-center point of each tracked bounding box.
- A JSONL event is generated when a tracked ID newly enters the configured zone; continuous presence does not repeatedly log entry events.
- The overlay reports a rolling estimate of **pipeline processing FPS**, plus detection, tracking, and analytics times. This is **not** camera/display end-to-end FPS.

## Known warnings and technical debt

1. **ByteTrack deprecation:** The installed `supervision` package warns that `ByteTrack` was deprecated in v0.28.0 and will be removed in v0.30.0. Migrate to a supported tracker API or pin compatible dependencies before upgrading.
2. **PyTorch JIT deprecation:** Dependencies emit a `torch.jit.script` FutureWarning. Monitor upstream compatibility.
3. **RF-DETR inference optimization:** RF-DETR warns that its checkpoint-based inference path is not optimized. Investigate its supported native inference API before drawing deployment performance conclusions.
4. **Configuration flags:** `tracking.enabled` and `zones.enabled` appear in the configuration, but the current main pipeline constructs these components unconditionally. Either implement these switches or remove the unused flags.
5. **Error handling and deployment:** Consider clearer source/weights/config validation, graceful error reporting, dependency pinning, and longer-running video tests.
6. **Tracking edge cases:** ID switches or temporary missed detections can produce repeated entry events; validate behavior on realistic videos.

## Recommended follow-up

- Add reproducible prerecorded-video integration tests, including multiple people, temporary occlusions, and re-entry events.
- Record sustained pipeline FPS and latency percentiles for both models under controlled input conditions.
- Resolve ByteTrack compatibility and optimize RF-DETR inference.
- Review repository hygiene before merging: exclude generated datasets, weights, logs, MLflow databases, and secrets.

## Sprint closure

Sprint 4's planned implementation and local validation criteria have been met. Merge into `main` after reviewing the staged diff and verifying that no generated artifacts or secrets are included.
