"""SGLang server launcher with RadixAttention for prefix caching.

Serves AWQ-quantized models with optional TurboQuant KV cache hook.

Usage::

    python deployment/serve_sglang.py --model checkpoints/awq \\
        --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)


def launch_sglang_server(
    model_path: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    max_total_tokens: int = 4096,
    mem_fraction_static: float = 0.85,
    tensor_parallel_size: int = 1,
    quantization: str | None = "awq",
    enable_turboquant: bool = False,
    turboquant_bits: int = 4,
    turboquant_window: int = 256,
    served_model_name: str | None = None,
    trust_remote_code: bool = True,
    disable_radix_cache: bool = False,
) -> None:
    """Launch an SGLang server with RadixAttention prefix caching.

    Parameters
    ----------
    model_path:
        Path to the model (HF format or AWQ quantized).
    host:
        Server host address.
    port:
        Server port.
    max_total_tokens:
        Maximum total tokens in KV cache.
    mem_fraction_static:
        Static GPU memory fraction for KV cache.
    tensor_parallel_size:
        Number of GPUs for tensor parallelism.
    quantization:
        Quantization method (``"awq"`` or ``None``).
    enable_turboquant:
        Whether to enable TurboQuant KV cache compression.
    turboquant_bits:
        TurboQuant quantization bits.
    turboquant_window:
        TurboQuant residual window size.
    served_model_name:
        Model name for the API.
    trust_remote_code:
        Whether to trust remote code.
    disable_radix_cache:
        Disable RadixAttention prefix caching.
    """
    if served_model_name is None:
        served_model_name = Path(model_path).name

    if enable_turboquant:
        logger.info(
            "TurboQuant enabled: %d-bit KV cache, %d-token residual window",
            turboquant_bits, turboquant_window,
        )
        logger.warning(
            "TurboQuant SGLang integration requires custom attention backend. "
            "Using standard SGLang KV cache."
        )

    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--host", host,
        "--port", str(port),
        "--mem-fraction-static", str(mem_fraction_static),
        "--tp-size", str(tensor_parallel_size),
    ]

    if quantization:
        cmd.extend(["--quantization", quantization])

    if trust_remote_code:
        cmd.append("--trust-remote-code")

    if disable_radix_cache:
        cmd.append("--disable-radix-cache")

    logger.info("Launching SGLang server: %s", " ".join(cmd))
    logger.info("API at http://%s:%d", host, port)

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        logger.info("Server shut down by user")
    except subprocess.CalledProcessError as exc:
        logger.error("SGLang server exited with code %d", exc.returncode)
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

    parser = argparse.ArgumentParser(description="Launch SGLang server")
    parser.add_argument("--model",                  default=None, help="Model path")
    parser.add_argument("--host",                   default=None)
    parser.add_argument("--port",                   type=int,   default=None)
    parser.add_argument("--max-total-tokens",       type=int,   default=None)
    parser.add_argument("--mem-fraction-static",    type=float, default=None)
    parser.add_argument("--tensor-parallel-size",   type=int,   default=None)
    parser.add_argument("--quantization",           default=None, choices=["awq", "none"])
    parser.add_argument("--enable-turboquant",      action="store_true")
    parser.add_argument("--turboquant-bits",        type=int, default=None)
    parser.add_argument("--turboquant-window",      type=int, default=None)
    parser.add_argument("--served-model-name",      default=None)
    parser.add_argument("--disable-radix-cache",    action="store_true")
    args = parser.parse_args()

    model_path = args.model or (str(_p.awq_dir) if _p else None)
    if not model_path:
        logger.error("No model path provided. Set paths.awq_dir in config.yaml or pass --model.")
        sys.exit(1)

    quant_raw = args.quantization or (str(_d.quantization) if _d and _d.quantization else "awq")
    # SGLang only supports awq; strip the _marlin suffix if present
    quant_raw = quant_raw.replace("_marlin", "")
    quant = quant_raw if quant_raw != "none" else None

    launch_sglang_server(
        model_path           = model_path,
        host                 = args.host                or (str(_d.host)   if _d else "0.0.0.0"),
        port                 = args.port                or (int(_d.port)   if _d else 8000),
        max_total_tokens     = args.max_total_tokens    or (int(_d.max_model_len)          if _d else 4096),
        mem_fraction_static  = args.mem_fraction_static or (float(_d.gpu_memory_utilization) if _d else 0.85),
        tensor_parallel_size = args.tensor_parallel_size or (int(_d.tensor_parallel_size)  if _d else 1),
        quantization         = quant,
        enable_turboquant    = args.enable_turboquant   or (bool(_d.turboquant_enabled) if _d else False),
        turboquant_bits      = args.turboquant_bits     or (int(_d.turboquant_bits)   if _d else 4),
        turboquant_window    = args.turboquant_window   or (int(_d.turboquant_window) if _d else 256),
        served_model_name    = args.served_model_name   or (str(_d.model_name) if _d else None),
        disable_radix_cache  = args.disable_radix_cache,
    )


if __name__ == "__main__":
    main()
