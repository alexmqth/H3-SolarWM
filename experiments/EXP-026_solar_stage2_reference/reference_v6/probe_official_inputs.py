"""One bounded, credential-free small-file probe per official provider."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import requests
import time

HERE = Path(__file__).resolve().parent
MAIN_REVISION = "60c85899ee5303bb3ccdd5add7b5d606c54ddaab"
TARGETS = [
    (
        "huggingface",
        "test-index.jsonl.gz",
        "https://huggingface.co/datasets/junchaoh-cs/SolarWM-Data/resolve/"
        + MAIN_REVISION
        + "/releases-v1/recipes/clean-158f-h3/latent-wds/"
        "minimax-h3-158f-768p-nomind-v1/test-index.jsonl.gz",
    ),
    (
        "modelscope",
        "encoder_contract.json",
        "https://modelscope.cn/datasets/junchao2003/"
        "SolarWM-Data_Latent-WDS_minimax-h3-158f-768p-nomind-v1/"
        "resolve/master/support/encoder_contract.json",
    ),
]


def main() -> None:
    results = []
    for provider, filename, url in TARGETS:
        result = {
            "provider": provider,
            "target_filename": filename,
            "requested_url": url,
            "attempts": 1,
            "start_unix": time.time(),
            "status": "UNKNOWN",
            "http_status": None,
            "downloaded_bytes": 0,
        }
        try:
            with requests.get(url, stream=True, timeout=(10, 50), allow_redirects=True) as response:
                result["http_status"] = response.status_code
                if response.status_code != 200:
                    result["status"] = "HTTP_DENIED_OR_UNAVAILABLE"
                else:
                    declared = response.headers.get("content-length")
                    result["declared_bytes"] = int(declared) if declared and declared.isdigit() else None
                    if result["declared_bytes"] and result["declared_bytes"] > 2 * 2**20:
                        result["status"] = "SKIPPED_OVER_2MIB_PROBE_CAP"
                    else:
                        content = bytearray()
                        for block in response.iter_content(chunk_size=64 * 1024):
                            content.extend(block)
                            if len(content) > 2 * 2**20:
                                result["status"] = "ABORTED_OVER_2MIB_PROBE_CAP"
                                break
                        if result["status"] == "UNKNOWN":
                            target = HERE / "inputs" / provider / filename
                            target.parent.mkdir(parents=True, exist_ok=True)
                            target.write_bytes(content)
                            result.update({
                                "status": "DOWNLOADED",
                                "path": str(target),
                                "downloaded_bytes": len(content),
                                "sha256": hashlib.sha256(content).hexdigest(),
                            })
        except requests.RequestException as exc:
            result.update({"status": "REQUEST_ERROR", "error_type": type(exc).__name__})
        result["elapsed_sec"] = time.time() - result["start_unix"]
        results.append(result)
        (HERE / "ACCESS_PROBES.json").write_text(json.dumps(results, indent=2) + "\n")
        print(provider, result["status"], result["http_status"], flush=True)


if __name__ == "__main__":
    main()
