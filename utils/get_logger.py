# ------------------------------------------------------------------------------#
#
# File name                 : get_logger.py
# Purpose                   : Utility for initializing logging to both file and console.
# Usage                     : Called during training/inference to initialize log files.
#
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh, Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 202410001@lums.edu.pk,
#                             hassan.mohyuddin@lums.edu.pk
#
# Last Modified             : June 21, 2025
# ------------------------------------------------------------------------------#


# --------------------------- Module imports -----------------------------------#
import os, logging

from configs.config             import *

# --------------------------- Open log file ------------------------------------#
def open_log(args, config):
    """
    Set up logging to file and console for Python config scripts.
    Args:
        args   : argparse.Namespace with 'config' (path to config .py)
        config : Config object (module or dict)
    """
    log_savepath = getattr(config, 'LOG_PATH', None) or config.get('LOG_PATH', None)
    if not os.path.exists(log_savepath):
        os.makedirs(log_savepath)

    log_name = os.path.splitext(os.path.basename(args.config))[0]  # no .py
    log_filename = os.path.join(log_savepath, f'{log_name}.txt')

    if os.path.isfile(log_filename):
        os.remove(log_filename)

    initLogging(log_filename)
    logging.info(f"Logging initialized. Log file: {log_filename}")
    
# --------------------------- Logger initializer -------------------------------#
def initLogging(logFilename):
    """
    Initializes Python logging to write logs to both a file and the console.

    Args:
        logFilename : Full path of the log file.

    Returns:
        None
    """

    logging.basicConfig(
        level     = logging.INFO,
        format    = '[%(asctime)s-%(levelname)s] %(message)s',
        datefmt   = '%y-%m-%d %H:%M:%S',
        filename  = logFilename,
        filemode  = 'w'
    )

    # Set up console output
    console        = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter      = logging.Formatter('[%(asctime)s-%(levelname)s] %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

# --------------------------------- End -----------------------------------------#
