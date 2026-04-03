from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
python_root = REPO_ROOT / "python"
if str(python_root) not in sys.path:
    sys.path.insert(0, str(python_root))


def emit(message: dict) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def progress(request_id: str, value: float, stage: str, message: str) -> None:
    emit(
        {
            "status": "progress",
            "request_id": request_id,
            "progress": value,
            "stage": stage,
            "message": message,
        }
    )


def handle_inspect(payload: dict) -> dict:
    from tritpack.backends.gguf_backend import GGUFBackend

    backend = GGUFBackend(payload["path"])
    meta = backend.load_metadata()
    return {
        "tensor_count": len(backend.tensor_names()),
        "metadata": {k: str(v) for k, v in list(meta.items())[:32]},
    }


def handle_estimate(payload: dict) -> dict:
    from tritpack.backends.gguf_backend import GGUFBackend

    backend = GGUFBackend(payload["path"])
    return backend.estimate_compressed_size(alpha=float(payload.get("alpha", 0.7)))


def handle_convert(request_id: str, payload: dict) -> dict:
    from integration.gguf_patcher import GGUFPatcher

    output_dir = Path(payload["output_dir"])
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    progress(request_id, 0.05, "prepare", "Inspecting GGUF and creating output directory")
    patcher = GGUFPatcher(
        alpha=float(payload.get("alpha", 0.7)),
        block_size=int(payload.get("block_size", 64)),
    )
    progress(request_id, 0.2, "convert", "Compressing tensors into TritPack format")

    def on_tensor(index: int, total: int, name: str) -> None:
        fraction = index / max(1, total)
        progress(
            request_id,
            0.2 + (fraction * 0.68),
            "convert",
            f"Compressing tensor {index}/{total}: {name}",
        )

    report = patcher.patch(
        payload["source_path"],
        str(output_dir),
        progress_callback=on_tensor,
    )
    progress(request_id, 0.92, "finalize", "Writing metadata and summarizing conversion")
    summary = report.summary()
    return {
        "output_dir": str(output_dir),
        "summary": (
            f"ratio={summary['ratio']}x | "
            f"original={summary['original_gb']} GB | "
            f"compressed={summary['compressed_gb']} GB"
        ),
    }


def handle_verify(payload: dict) -> dict:
    from integration.gguf_patcher import GGUFPatcher

    sample_tensors = int(payload.get("sample_tensors", 8))
    patcher = GGUFPatcher()
    report = patcher.verify(payload["source_path"], payload["tritpack_dir"])
    sampled = report.layers[:sample_tensors] if sample_tensors < len(report.layers) else report.layers

    if sampled:
        mean_cos = sum(layer["cos_sim"] for layer in sampled) / len(sampled)
        mean_snr = sum(layer["snr_db"] for layer in sampled) / len(sampled)
    else:
        mean_cos = None
        mean_snr = None

    metadata_complete = (Path(payload["tritpack_dir"]) / "metadata.json").exists()
    return {
        "ok": metadata_complete and (mean_cos is None or mean_cos >= 0.8),
        "tensors_verified": len(sampled),
        "metadata_complete": metadata_complete,
        "mean_cosine_similarity": mean_cos,
        "mean_snr_db": mean_snr,
        "notes": [],
    }


def handle_reconstruct(payload: dict) -> dict:
    from integration.tritpack_loader import TritPackLoader

    loader = TritPackLoader(payload["tritpack_dir"])
    loader.reconstruct_gguf(payload["output_path"])
    return {"output_path": payload["output_path"]}


def main() -> int:
    handlers = {
        "inspect": lambda request_id, payload: handle_inspect(payload),
        "estimate": lambda request_id, payload: handle_estimate(payload),
        "convert": handle_convert,
        "verify": lambda request_id, payload: handle_verify(payload),
        "reconstruct": lambda request_id, payload: handle_reconstruct(payload),
    }

    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            request = json.loads(raw_line)
            request_id = request["request_id"]
            command = request["command"]
            payload = request.get("payload", {})

            if command == "shutdown":
                emit({"status": "ok", "request_id": request_id, "payload": {"bye": True}})
                return 0

            if command not in handlers:
                raise ValueError(f"Unsupported command: {command}")

            result = handlers[command](request_id, payload)
            emit({"status": "ok", "request_id": request_id, "payload": result})
        except Exception as exc:  # pragma: no cover - worker safety net
            emit(
                {
                    "status": "error",
                    "request_id": request.get("request_id") if "request" in locals() else "unknown",
                    "message": f"{exc}\n{traceback.format_exc()}",
                }
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
