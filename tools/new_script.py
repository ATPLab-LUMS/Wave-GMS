# ------------------------------------------------------------------------------#
#
# File name                 : new_script.py
# Purpose                   : Generate a new Python script with GMS style header & scaffold
# Usage                     : python3 tools/new_script.py --name valid.py --purpose "Main Inference Loop for GMS" --usage "python valid.py"
#
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh, 
#                             Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 24020001@lums.edu.pk,
#                             hassan.mohyuddin@lums.edu.pk
#
# Last Modified             : August 25, 2025
# ------------------------------------------------------------------------------#

# --------------------------- Module Imports -----------------------------------#
import argparse, sys, textwrap
from pathlib                  import Path
from datetime                 import datetime

# ------------------------------------------------------------------------------#
#                         Template & Helpers                                     #
# ------------------------------------------------------------------------------#
HEADER_TEMPLATE = """# ------------------------------------------------------------------------------#
#
# File name                 : {fname}
# Purpose                   : {purpose}
# Usage                     : {usage}
#
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh, 
#                             Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 24020001@lums.edu.pk,
#                             hassan.mohyuddin@lums.edu.pk
#
# Last Modified             : {date}
# ------------------------------------------------------------------------------#

# --------------------------- Module Imports -----------------------------------#
import torch, os, time, logging, argparse

import numpy                as np
import pandas               as pd

from tqdm                   import tqdm
from PIL                    import Image
from einops                 import rearrange
from torch.utils.data       import DataLoader

from data                   import *
from utils                  import *
from networks               import *
from diffusers              import AutoencoderTiny
from configs                import config
# ------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------#
#                         Main                                                   #
# ------------------------------------------------------------------------------#
def main() -> None:
    parser                  = argparse.ArgumentParser()
    parser.add_argument("--data_root",   type=str, required=False, help="Path to dataset root")
    args                    = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger                  = logging.getLogger("{stem}")
    logger.info("Starting {stem} ...")
    # TODO: implement your logic here

if __name__ == "__main__":
    # Example: {usage}
    main()
"""

def today() -> str:
    return datetime.now().strftime("%B %d, %Y")

def generate_script(path: Path, purpose: str, usage: str) -> None:
    date    = today()
    stem    = path.stem
    content = HEADER_TEMPLATE.format(
        fname=path.name, purpose=purpose, usage=usage, date=date, stem=stem
    )

    if path.exists():
        print(f"[ERROR] File already exists: {path}")
        sys.exit(1)

    path.write_text(content, encoding="utf-8")
    print(f"[OK] Created: {path}")

# ------------------------------------------------------------------------------#
#                         CLI                                                    #
# ------------------------------------------------------------------------------#
def parse_args():
    p = argparse.ArgumentParser(description="Generate a new script with GMS style header & scaffold")
    p.add_argument("--name",    required=True,  type=str, help="Filename to create (e.g., valid.py)")
    p.add_argument("--purpose", required=True,  type=str, help="Short purpose line")
    p.add_argument("--usage",   required=False, type=str, help="Usage line", default=None)
    return p.parse_args()

def main_cli():
    args   = parse_args()
    name   = args.name
    usage  = args.usage or f"python {name}"
    generate_script(Path(name), purpose=args.purpose, usage=usage)

if __name__ == "__main__":
    main_cli()