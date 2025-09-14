# ------------------------------------------------------------------------------#
#
# File name                 : get_logger.py
# Purpose                   : Unified file+console logging setup for GMS
# Usage                     : from tools.get_logger import open_log
#
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh,
#                             Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 202410001@lums.edu.pk,
#                             hassan.mohyuddin@lums.edu.pk
#
# Last Modified             : August 26, 2025
# ------------------------------------------------------------------------------#

# --------------------------- Module Imports -----------------------------------#
import logging
import os
from pathlib                   import Path
from typing                    import Optional
# ------------------------------------------------------------------------------#


# --------------------------- Internal Helpers ---------------------------------#
def _ensure_parent(path: Path) -> None:
    """Create parent directory(ies) if missing."""
    path.parent.mkdir(parents=True, exist_ok=True)


def _build_logfile_path(log_dir: Path, args_config: str) -> Path:
    """Derive logfile path from args.config (strip extension, append -yaml.txt)."""
    stem = Path(args_config).stem + "-yaml"
    return log_dir / f"{stem}.txt"


def _setup_logger(logger_name: str,
                  logfile: Path,
                  level: int = logging.INFO,
                  console_level: Optional[int] = logging.INFO) -> logging.Logger:
    """Create or reset a named logger with file + optional console handlers."""
    _ensure_parent(logfile)

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # Remove old handlers to avoid duplicate logs when re-calling open_log
    if logger.hasHandlers():
        logger.handlers.clear()

    # Common formatter
    fmt      = "[%(asctime)s-%(levelname)s] %(message)s"
    datefmt  = "%y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    # File handler (overwrite each run)
    fh = logging.FileHandler(logfile, mode="w", encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler (optional)
    if console_level is not None:
        ch = logging.StreamHandler()
        ch.setLevel(console_level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # Reduce verbosity of noisy third-party loggers (optional, safe defaults)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    return logger
# ------------------------------------------------------------------------------#


# ------------------------------ Public API ------------------------------------#
def open_log(args, config) -> logging.Logger:
    """
    Create log directory if needed, derive logfile name from args.config, and
    return a configured logger. Keeps backward compatibility with your code.

    Expected:
        config['log_path']   → directory to write logs
        args.config          → path/to/config.yaml  (used only to build filename)
    """
    log_dir   = Path(config["log_path"])
    logfile   = _build_logfile_path(log_dir, args.config)

    # (Optional) remove previous logfile to mirror old behavior
    if logfile.exists():
        try:
            logfile.unlink()
        except OSError:
            # If another process holds the file, we overwrite via FileHandler(mode="w") anyway.
            pass

    # Create and return a named logger
    logger = _setup_logger(logger_name="gms", logfile=logfile,
                           level=logging.INFO, console_level=logging.INFO)
    logger.info(f"Logging to: {logfile}")
    return logger


def initLogging(logFilename: str) -> None:
    """
    Backward-compatible shim for legacy codepaths that directly call initLogging().
    Prefer open_log() which returns a logger instance.

    This sets up a 'gms_legacy' named logger with the same formatting.
    """
    logfile = Path(logFilename)
    _setup_logger(logger_name="gms_legacy", logfile=logfile,
                  level=logging.INFO, console_level=logging.INFO)
# ------------------------------------------------------------------------------#