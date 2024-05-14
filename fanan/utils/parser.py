import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Art & Creativity in JAX 🎨 💗")
    parser.add_argument(
        "--config-path",
        type=str,
        default="configs/default.yaml",
        required=True,
        help="Path to .yaml config file (default: %(default)s)",
    )
    return parser.parse_args()
