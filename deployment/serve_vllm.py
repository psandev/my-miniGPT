"""vLLM server launcher with OpenAI-compatible API.

Serves AWQ-quantized models with optional Marlin kernel and TurboQuant
KV cache compression hook.

Usage::

    python deployment/serve_vllm.py --model checkpoints/awq \\
        --host 0.0.0.0 --port 8000 --max-batch-size 32
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)


def launch_vllm_server(
    model_path: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    max_model_len: int = 4096,
    gpu_memory_utilization: float = 0.85,
    max_num_batched_tokens: int | None = None,
    tensor_parallel_size: int = 1,
    quantization: str | None = "awq_marlin",
    dtype: str = "auto",
    enable_turboquant: bool = False,
    turboquant_bits: int = 4,
    turboquant_window: int = 256,
    served_model_name: str | None = None,
    api_key: str | None = None,
    trust_remote_code: bool = True,
) -> None:
    """Launch a vLLM server with OpenAI-compatible API endpoint.

    Parameters
    ----------
    model_path:
        Path to the model (HF format, AWQ, or GPTQ).
    host:
        Server host address.
    port:
        Server port.
    max_model_len:
        Maximum model context length.
    gpu_memory_utilization:
        Fraction of GPU memory to use (0-1).
    max_num_batched_tokens:
        Maximum tokens in a batch.
    tensor_parallel_size:
        Number of GPUs for tensor parallelism.
    quantization:
        Quantization method. Options: ``"awq_marlin"``, ``"awq"``, ``"gptq"``, ``None``.
    dtype:
        Model dtype. ``"auto"`` selects based on model config.
    enable_turboquant:
        Whether to enable TurboQuant KV cache compression.
    turboquant_bits:
        TurboQuant quantization bits.
    turboquant_window:
        TurboQuant residual window size.
    served_model_name:
        Name to advertise in the API. Defaults to model directory name.
    api_key:
        Optional API key for authentication.
    trust_remote_code:
        Whether to trust remote code in model repos.
    """
    if served_model_name is None:
        served_model_name = Path(model_path).name

    # Log TurboQuant status
    if enable_turboquant:
        logger.info(
            "TurboQuant enabled: %d-bit KV cache, %d-token residual window",
            turboquant_bits, turboquant_window,
        )
        # TurboQuant is applied as a runtime hook.
        # For vLLM integration, this would require a custom attention backend.
        logger.warning(
            "TurboQuant vLLM integration requires a custom attention backend. "
            "Using standard vLLM KV cache for now."
        )

    # Build vLLM command
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--host", host,
        "--port", str(port),
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--dtype", dtype,
        "--served-model-name", served_model_name,
    ]

    if quantization:
        cmd.extend(["--quantization", quantization])

    if max_num_batched_tokens:
        cmd.extend(["--max-num-batched-tokens", str(max_num_batched_tokens)])

    if api_key:
        cmd.extend(["--api-key", api_key])

    if trust_remote_code:
        cmd.append("--trust-remote-code")

    logger.info("Launching vLLM server: %s", " ".join(cmd))
    logger.info("OpenAI-compatible API at http://%s:%d/v1", host, port)

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        logger.info("Server shut down by user")
    except subprocess.CalledProcessError as exc:
        logger.error("vLLM server exited with code %d", exc.returncode)
        raise


def _load_cfg():
    """Load central config.yaml via OmegaConf (silent fallback if missing)."""
    try:
        from omegaconf import OmegaConf
        cfg_path = Path(__file__).resolve().parent.parent / "config.yaml"
        return OmegaConf.load(cfg_path)
    except Exception:
        return None


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # Load central config — deployment parameters default from config.yaml
    cfg = _load_cfg()
    _d = cfg.deploy if cfg else None
    _p = cfg.paths  if cfg else None

    parser = argparse.ArgumentParser(description="Launch vLLM OpenAI-compatible server")
    parser.add_argument("--model",                    default=None, help="Model path (HF/AWQ/GPTQ)")
    parser.add_argument("--host",                     default=None)
    parser.add_argument("--port",                     type=int,   default=None)
    parser.add_argument("--max-model-len",            type=int,   default=None)
    parser.add_argument("--gpu-memory-utilization",   type=float, default=None)
    parser.add_argument("--max-num-batched-tokens",   type=int,   default=None)
    parser.add_argument("--tensor-parallel-size",     type=int,   default=None)
    parser.add_argument("--quantization",             default=None,
                        choices=["awq_marlin", "awq", "gptq", "none"])
    parser.add_argument("--dtype",                    default=None)
    parser.add_argument("--enable-turboquant",        action="store_true")
    parser.add_argument("--turboquant-bits",          type=int, default=None)
    parser.add_argument("--turboquant-window",        type=int, default=None)
    parser.add_argument("--served-model-name",        default=None)
    parser.add_argument("--api-key",                  default=None)
    args = parser.parse_args()

    model_path = args.model or (str(_p.awq_dir) if _p else None)
    if not model_path:
        logger.error("No model path provided. Set paths.awq_dir in config.yaml or pass --model.")
        sys.exit(1)

    quant_raw = args.quantization or (str(_d.quantization) if _d and _d.quantization else "awq_marlin")
    quant = quant_raw if quant_raw != "none" else None

    launch_vllm_server(
        model_path             = model_path,
        host                   = args.host                  or (str(_d.host)   if _d else "0.0.0.0"),
        port                   = args.port                  or (int(_d.port)   if _d else 8000),
        max_model_len          = args.max_model_len         or (int(_d.max_model_len)          if _d else 4096),
        gpu_memory_utilization = args.gpu_memory_utilization or (float(_d.gpu_memory_utilization) if _d else 0.85),
        max_num_batched_tokens = args.max_num_batched_tokens,
        tensor_parallel_size   = args.tensor_parallel_size  or (int(_d.tensor_parallel_size)   if _d else 1),
        quantization           = quant,
        dtype                  = args.dtype                 or (str(_d.dtype)  if _d else "auto"),
        enable_turboquant      = args.enable_turboquant     or (bool(_d.turboquant_enabled) if _d else False),
        turboquant_bits        = args.turboquant_bits       or (int(_d.turboquant_bits)   if _d else 4),
        turboquant_window      = args.turboquant_window     or (int(_d.turboquant_window) if _d else 256),
        served_model_name      = args.served_model_name     or (str(_d.model_name) if _d else None),
        api_key                = args.api_key               or (str(_d.api_key) if _d and _d.api_key else None),
    )


if __name__ == "__main__":
    main()
