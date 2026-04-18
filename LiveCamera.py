from __future__ import annotations

# =========================
# 📦 Imports
# =========================
import argparse
import time
import cv2

from config import Config
from reporter import create_and_save_report
from yolo_detect import detect_frame


# =========================
# 🎥 Live Monitoring Engine
# =========================
def run_live_monitor(source: str | int = 0) -> None:
    """
    Run real-time road monitoring using webcam, IP camera, or video file.

    Args:
        source (str | int): 
            - Webcam index (e.g., 0)
            - Video file path
            - IP camera stream URL
    """

    # Initialize video capture
    capture = cv2.VideoCapture(source)

    # Optimize buffer for IP streams (reduce latency)
    if isinstance(source, str) and source.startswith("http"):
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 3)

    # Check if camera opened successfully
    if not capture.isOpened():
        print(f"[CAMERA][ERROR] Unable to open source: {source}")
        return

    # =========================
    # 📊 Stats Initialization
    # =========================
    processed_frames = 0
    detection_count = 0
    start_time = time.time()
    last_stats_time = start_time

    print("[CAMERA] Monitoring started... Press 'Q' to quit.")

    # =========================
    # 🔁 Main Processing Loop
    # =========================
    while True:
        ok, frame = capture.read()

        # Break if stream ends
        if not ok or frame is None:
            print("[CAMERA] Stream ended or frame unavailable.")
            break

        processed_frames += 1
        display_frame = frame.copy()

        # =========================
        # 🧠 Run Detection (Interval Based)
        # =========================
        if processed_frames % Config.DETECTION_INTERVAL == 0:
            result = detect_frame(frame)

            if result["detected"]:
                detection_count += 1

                # Load annotated image (with bounding boxes)
                annotated = cv2.imread(result["image_path"])
                if annotated is not None:
                    display_frame = annotated

                # 📍 Create structured report
                report = create_and_save_report(
                    lat=12.9716,  # TODO: Replace with real GPS
                    lng=77.5946,
                    image_path=result["image_path"],
                    severity=result["severity"],
                    confidence=result["confidence"],
                )

                print(
                    f"[DETECTED] {report['hazard_type']} | "
                    f"{report['address']} | "
                    f"Confidence: {report['confidence']:.2f}"
                )

        # =========================
        # 🖥️ Display Output
        # =========================
        cv2.imshow("🚦 RoadWatch AI Live Feed", display_frame)

        # Exit on 'Q'
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            print("[CAMERA] Stop requested by user.")
            break

        # =========================
        # 📈 Print Stats (Every 30 sec)
        # =========================
        now = time.time()
        if now - last_stats_time >= 30:
            elapsed = max(now - start_time, 1)
            fps = processed_frames / elapsed

            print(
                f"[STATS] Frames: {processed_frames} | "
                f"Detections: {detection_count} | "
                f"Avg FPS: {fps:.2f}"
            )

            last_stats_time = now

    # =========================
    # 🧹 Cleanup
    # =========================
    capture.release()
    cv2.destroyAllWindows()
    print("[CAMERA] Monitoring stopped.")


# =========================
# ⚙️ CLI Argument Parsing
# =========================
def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for camera input.
    """
    parser = argparse.ArgumentParser(
        description="RoadWatch AI Live Camera Monitor"
    )

    parser.add_argument(
        "--source",
        default="0",
        help='Webcam index ("0"), video file path, or IP (e.g., 192.168.1.5:8080)',
    )

    return parser.parse_args()


# =========================
# 🔄 Source Normalization
# =========================
def normalize_source(raw_source: str) -> str | int:
    """
    Normalize input source:
    - Convert numeric string → webcam index
    - Convert IP:PORT → proper video URL
    """

    raw_source = raw_source.strip()

    # Webcam index
    if raw_source.isdigit():
        return int(raw_source)

    # Convert IP:PORT to stream URL
    if (
        not raw_source.startswith("http")
        and ("." in raw_source or ":" in raw_source)
        and ("/" not in raw_source and "\\" not in raw_source)
    ):
        return f"http://{raw_source}/video"

    return raw_source


# =========================
# 🚀 Entry Point
# =========================
if __name__ == "__main__":
    args = parse_args()
    source = normalize_source(args.source)

    print(f"[SYSTEM] Starting with source: {source}")
    run_live_monitor(source)