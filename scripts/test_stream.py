
import argparse
import time

from video_analytics.stream import VideoStream


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()

    source = (
        int(args.source)
        if args.source.isdecimal()
        else args.source
    )

    stream = VideoStream(
        source=source,
        loop_file=args.loop,
    )

    start = time.perf_counter()
    count = 0

    try:
        for frame in stream.frames():
            count += 1

            if count % 30 == 0:
                print(
                    f"Frames: {count} | "
                    f"Resolution: {frame.shape[1]}x{frame.shape[0]}"
                )

            if count >= args.max_frames:
                break

    finally:
        stream.stop()

    elapsed = time.perf_counter() - start

    print(
        f"Processed {count} frames "
        f"in {elapsed:.2f} seconds"
    )


if __name__ == "__main__":
    main()
