# STYLEGUIDE

These rules must be followed in **every Python script** in this repository.

---
## 1) Mandatory file header (top of every script)
```python
# ------------------------------------------------------------------------------#
#
# File name                 : <script_name>.py
# Purpose                   : <short summary>
# Usage                     : python <script_name>.py [--flags]
#
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh, 
#                             Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 202410001@lums.edu.pk,
#                             hassan.mohyuddin@lums.edu.pk
#
# Last Modified             : <Month DD, YYYY>
# ------------------------------------------------------------------------------#
```

- Keep the **columns aligned** exactly as shown (name/purpose/usage/authors/email/last modified).
- If a script has no CLI, keep `Usage` and show “See example in `main()`”.

---
## 2) Imports (grouped & aligned)

- Groups: **stdlib → third-party → project-local**.
- Align like a table (see example below).
- One blank line **between groups**.

```python
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
```
- Prefer explicit imports; wildcard (`*`) is acceptable for **project-local** modules when intentional.

--- 
## 3) Section breakers

Before **every class, function, or major block**, insert a 3-line breaker with a short descriptive title:
Examples:
- `Dataset: Qatar-19`
- `Model: LiteVAE Encoder-Only`
- `Main: Inference Loop`

---

## 4) Alignment

- Align `=` within a **local block**:

```python
batch_size          = 16
learning_rate       = 1e-4
num_epochs          = 100
save_every_steps    = 500
```
- Align `CLI args` similarly.

```python
parser.add_argument("--data_root",   type=str,   required=True, help="Path to Dataset/Qatar-19")
parser.add_argument("--batch_size",  type=int,   default=16,    help="Per-device batch size")
parser.add_argument("--precision",   type=str,   default="fp16",choices=["fp32","fp16","bf16"])
```
> Keep alignment **local** (don’t force alignment across unrelated blocks).

---

## 5) Main block

Every runnable script must end with:

```python
if __name__ == "__main__":
    # Example: python valid.py --data_root ./Dataset/Qatar-19
    main()
```

---

## 6) Logging, I/O, and paths

- Use `logging` for status; avoid bare prints for pipeline status.
- Use `pathlib.Path` for paths.
- No hard-coded absolute user paths; put them in configs or CLI args.

---

## 7) Config-driven, not if/elif sprawl

- Use registries/factories or config targets (e.g., `instantiate_from_config`) to select components.
- Adding new models/features must **not** add more top-level `if model == "X"` branches.

---

## 8) GPU / memory-safe patterns

- **Build on CPU → (optional) freeze → single `.to(device, dtype)` move**.
- Avoid repeated device/dtype moves.
- Use `channels_last` only on CUDA tensors that benefit from it.
- Prefer `safetensors` for checkpoints.

---

## 9) Minimal docstrings & typing

- Functions/classes should have a one-liner purpose and type hints where reasonable.

---

## 10) Last Modified

- Update the **Last Modified** date in the header whenever you edit a file.

---