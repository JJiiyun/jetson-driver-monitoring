#!/usr/bin/env python3
from __future__ import annotations

import sys

from run_video_inference import main


if __name__ == "__main__":
    sys.exit(main(use_fsm=True))
