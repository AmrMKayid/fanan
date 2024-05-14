import logging

import jax
from rich.logging import RichHandler

from fanan.config import Config
from fanan.utils.parser import parse_args

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        RichHandler(
            rich_tracebacks=True,
            markup=True,
            show_time=True,
            omit_repeated_times=False,
            show_level=True,
            show_path=True,
            tracebacks_show_locals=True,
        )
    ],
)


def main() -> None:
    logging.info(f"Total devices: {jax.device_count()}, " f"Devices per task: {jax.local_device_count()}")

    args = parse_args()
    config = Config.read_config_from_yaml(args.config_path)
    logging.info(f"{config=}")


if __name__ == "__main__":
    main()
