import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PaddleOCR against one benchmark case.")
    parser.add_argument(
        "--case",
        required=True,
        help="Case id under benchmarks/ocr-japanese-manga/cases/",
    )
    parser.add_argument(
        "--image",
        default="source.png",
        help="Benchmark image filename inside the case directory. Default: source.png",
    )
    parser.add_argument(
        "--lang",
        default="japan",
        help="PaddleOCR language code. Default: japan",
    )
    parser.add_argument(
        "--device",
        default="gpu:0",
        help="Paddle device string. Default: gpu:0",
    )
    parser.add_argument(
        "--det-model",
        default="PP-OCRv5_server_det",
        help="Text detection model name. Default: PP-OCRv5_server_det",
    )
    parser.add_argument(
        "--rec-model",
        default="PP-OCRv5_server_rec",
        help="Text recognition model name. Default: PP-OCRv5_server_rec",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory. Default: ./outputs",
    )
    parser.add_argument(
        "--disable-layout-postprocess",
        action="store_true",
        help="Disable Japanese vertical reading-order postprocessing.",
    )
    return parser.parse_args()
