# EXP-026/v4 · isolated dependency gate

The [Worker report](WORKER_REPORT.md) records why this round ended at CPU preflight with **no GPU attempt**. The official matching-tag [wheel receipt](WHEEL_RECEIPT.json) and [installation log](INSTALL_LOG.md) trace the single download and isolated overlay. [ENV_PREFLIGHT.json](ENV_PREFLIGHT.json) records the real native import failure, Diffusers flash requirement result, C++ symbol comparison and CUDA-compiler mismatch. [COMPILE_DECISION.md](COMPILE_DECISION.md) explains why the allowed one-time source compile was not started.

The v3 four-card load config and patch remained unchanged; [SOURCE_REFERENCE.json](SOURCE_REFERENCE.json) pins their hashes. v3's failed GPU attempt and its 105.265 GPU seconds remain in `../memory_load_v3/`. v4 used 0 GPU seconds and produced no reader output or video.
