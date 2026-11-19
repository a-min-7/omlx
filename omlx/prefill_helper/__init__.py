# SPDX-License-Identifier: Apache-2.0
"""Prefill Accelerator: a helper Mac runs the first layers of each prompt chunk.

This Mac keeps the whole model and does all decoding. For each prefill chunk
the helper runs layers ``[0, k)`` and sends back the tensors that cross the cut
plus the chunk's cache state for those layers; this Mac runs ``[k, n)`` and then
applies that state, so every cache ends the chunk at the same offset.
"""

import os

# The shape version of the memory plan the two Macs exchange. The protocol
# version and a family's contract say whether a helper can serve a model at
# all; this says whether both sides read ``plan()``'s answer the same way.
# Bump it when ``shared``/``per_layer`` change meaning, not on every edit.
PLAN_API_VERSION = 1

# The source this build came from, when the packaging sets it. A helper
# built from another commit can report the same version and still plan or
# load differently, which ``version_ok`` cannot see.
COMMIT = os.environ.get("OMLX_COMMIT", "unknown")
