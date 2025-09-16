# ------------------------------------------------------------------------------#
#
# File name                 : get_logger.py
# Purpose                   : Centralized logger setup (file + console) with safe handler init
# Usage                     : from utils.get_logger import open_log
#
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh,
#                             Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 24020001@lums.edu.pk,
#                             hassan.mohyuddin@lums.edu.pk
#
# Last Modified             : Sep 16, 2025
# ------------------------------------------------------------------------------#

# --------------------------- Module Imports -----------------------------------#
from pathlib import Path
import logging
from typing import Any, Dict
# ------------------------------------------------------------------------------#


def open_log(args: Any, config: Dict[str, Any]) -> logging.Logger:
    """
    Initialize logging using repo-style config dict and CLI args.

    Expected:
        - config['log_path']: directory to store logs
        - args.config       : path to the YAML used (for log filename)

    Returns:
        logging.Logger: configured root logger.
    """
    log_dir         = Path(config["log_path"])
    log_dir.mkdir(parents=True, exist_ok=True)

    yaml_name       = Path(getattr(args, "config")).stem  # e.g., bus_train
    log_file        = log_dir / f"{yaml_name}-yaml.txt"

    return _init_logging(log_file)


def _init_logging(log_file: Path, level: int = logging.INFO) -> logging.Logger:
    """
    Create a root logger that logs to both a file and the console.
    Safe against duplicate handlers on repeated calls.
    """
    logger                  = logging.getLogger()  # root
    logger.setLevel(level)
    logger.propagate        = False  # prevent double prints in some environments

    # Clear existing handlers (important if re-initializing in the same process)
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)

    fmt                     = logging.Formatter("[%(asctime)s-%(levelname)s] %(message)s", "%y-%m-%d %H:%M:%S")

    # File handler
    fh                      = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler
    ch                      = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logger.info("Logging initialized")
    logger.info(f"Log file: {log_file.resolve()}")
    return logger