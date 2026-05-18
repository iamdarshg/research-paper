"""Backward-compatible import surface for the offline densifier.

The implementation lives in `offline_densify.py` to keep the claims honest:
this is an offline RLVR-style verifier loop, not online RL training.
"""

from offline_densify import *  # noqa: F401,F403
