# SPDX-License-Identifier: Apache-2.0
"""Prefill Accelerator helper for oMLX on Linux with NVIDIA GPUs.

It shares no code with oMLX but speaks its cluster discovery, pairing and
prefill-helper wire protocols.
"""

import os

__version__ = "0.1.6"

# The shape version of the memory plan this helper's planner computes. The
# data protocol and a family's contract cannot see a plan difference; a host
# compares this against its own and flags the mismatch.
PLAN_API_VERSION = 1

# The source this helper was built from, set by the packaging step. A host
# logs it and can flag a build that differs from its own, which the version
# and contracts cannot see either; "unknown" is never a mismatch.
COMMIT = os.environ.get("OMLX_CUDA_HELPER_COMMIT", "unknown")

# Framing and ops of the prefill data channel (oMLX ``wire.PROTOCOL``).
DATA_PROTOCOL = 1
# Ops of the control channel between a paired host and this helper.
CONTROL_VERSION = 1
# A stage worker prints this before the JSON that says it is ready.
READY_PREFIX = "OMLX_CUDA_HELPER_READY "
