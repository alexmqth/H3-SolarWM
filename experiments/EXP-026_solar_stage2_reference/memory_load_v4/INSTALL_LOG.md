# Official wheel installation record

The only new download was the official Dao-AILab FlashAttention v2.8.3 release asset `flash_attn-2.8.3+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl` (256,007,522 bytes). [WHEEL_RECEIPT.json](WHEEL_RECEIPT.json) records the full official URL and SHA-256 `1c8d2fb083edea3647688fcd4bb21d4641eb01f50359d8fa6200ecd093ba1c54`.

The command used the existing `solarwm-h3` interpreter only to invoke pip; it wrote into this experiment's isolated target and did not install dependencies or modify the shared environment:

```bash
.venvs/solarwm-h3/bin/python -m pip install --no-index --no-deps \
  --target submission/experiments/EXP-026_solar_stage2_reference/memory_load_v4/overlay \
  submission/experiments/EXP-026_solar_stage2_reference/memory_load_v4/wheel/*.whl
```

Pip reported `Successfully installed flash-attn-2.8.3`. The overlay-native `.so` SHA-256 is `8db3f3c5ada7b67060fc1f89c8e14ce23b007883319899a643e4a113e5a52268`, identical to the existing failing extension. CPU import therefore still fails at the same undefined symbol. Total v4 directory size at audit was 1,259,730,459 bytes, under the 8 GiB new-disk cap; no model or dataset was downloaded.
