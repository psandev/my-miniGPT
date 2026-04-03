"""Convert HuggingFace model to GGUF format via llama.cpp conversion scripts.

Supports quantization types: Q4_K_M (default), Q6_K, Q4_K_S.

Usage::

    python quantization/export_gguf.py --model-path checkpoints/hf_export \\
        --output checkpoints/gguf/model.gguf --quant-type Q4_K_M
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

SUPPORTED_QUANT_TYPES = ["Q4_K_M", "Q6_K", "Q4_K_S", "Q5_K_M", "Q8_0", "F16"]


def find_llama_cpp_path() -> Path | None:
    """Locate the llama.cpp installation directory.

    Searches common locations and environment variables.
    """
    # Check environment variable
    env_path = os.environ.get("LLAMA_CPP_PATH")
    if env_path and Path(env_path).is_dir():
        return Path(env_path)

    # Check common locations
    common_paths = [
        Path.home() / "llama.cpp",
        Path("/opt/llama.cpp"),
        Path("./llama.cpp"),
    ]
    for p in common_paths:
        if p.is_dir():
            return p

    # Check if llama-quantize is on PATH
    if shutil.which("llama-quantize"):
        return None  # Available on PATH, no need for a directory

    return None


def convert_hf_to_gguf(
    model_path: str,
    output_path: str,
    quant_type: str = "Q4_K_M",
    llama_cpp_path: str | None = None,
) -> Path:
    """Convert a HuggingFace model to GGUF format.

    This is a two-step process:
    1. Convert HF model to F16 GGUF using ``convert_hf_to_gguf.py``
    2. Quantize to the target type using ``llama-quantize``

    Parameters
    ----------
    model_path:
        Path to the HuggingFace model directory.
    output_path:
        Path for the output GGUF file.
    quant_type:
        Quantization type. One of: Q4_K_M, Q6_K, Q4_K_S, Q5_K_M, Q8_0, F16.
    llama_cpp_path:
        Path to llama.cpp directory. Auto-detected if not provided.

    Returns
    -------
    Path
        Path to the quantized GGUF file.
    """
    if quant_type not in SUPPORTED_QUANT_TYPES:
        raise ValueError(f"Unsupported quant type '{quant_type}'. Supported: {SUPPORTED_QUANT_TYPES}")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Find llama.cpp
    if llama_cpp_path:
        cpp_dir = Path(llama_cpp_path)
    else:
        cpp_dir = find_llama_cpp_path()

    # Step 1: Convert HF -> F16 GGUF
    f16_path = out_path.parent / f"{out_path.stem}_f16.gguf"

    if cpp_dir:
        convert_script = cpp_dir / "convert_hf_to_gguf.py"
    else:
        convert_script = None

    if convert_script and convert_script.exists():
        logger.info("Converting HF model to F16 GGUF using %s", convert_script)
        cmd = [
            sys.executable, str(convert_script),
            model_path,
            "--outfile", str(f16_path),
            "--outtype", "f16",
        ]
        subprocess.run(cmd, check=True)
    else:
        # Try using the llama-cpp-python library as fallback
        logger.info("llama.cpp convert script not found, trying llama-cpp-python")
        try:
            from llama_cpp import llama_cpp as _  # noqa: F401 (import check)
        except ImportError:
            pass

        # Alternative: use the python package directly
        cmd = [
            sys.executable, "-m", "llama_cpp.convert",
            model_path,
            "--outfile", str(f16_path),
            "--outtype", "f16",
        ]
        try:
            subprocess.run(cmd, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error(
                "Could not convert model. Install llama.cpp or set LLAMA_CPP_PATH.\n"
                "  git clone https://github.com/ggerganov/llama.cpp\n"
                "  cd llama.cpp && make"
            )
            raise

    logger.info("F16 GGUF created: %s", f16_path)

    # Step 2: Quantize (skip if target is F16)
    if quant_type == "F16":
        if f16_path != out_path:
            shutil.move(str(f16_path), str(out_path))
        return out_path

    quantize_bin = None
    if cpp_dir:
        for name in ["llama-quantize", "quantize"]:
            candidate = cpp_dir / name
            if candidate.exists():
                quantize_bin = str(candidate)
                break
            candidate = cpp_dir / "build" / "bin" / name
            if candidate.exists():
                quantize_bin = str(candidate)
                break

    if not quantize_bin:
        quantize_bin = shutil.which("llama-quantize") or shutil.which("quantize")

    if not quantize_bin:
        logger.error("llama-quantize binary not found. Build llama.cpp first.")
        raise FileNotFoundError("llama-quantize not found")

    logger.info("Quantizing F16 -> %s using %s", quant_type, quantize_bin)
    cmd = [quantize_bin, str(f16_path), str(out_path), quant_type]
    subprocess.run(cmd, check=True)

    # Clean up intermediate F16 file
    if f16_path.exists() and f16_path != out_path:
        f16_path.unlink()
        logger.info("Removed intermediate F16 file")

    logger.info("GGUF export complete: %s (%s)", out_path, quant_type)
    return out_path


def export_multiple_quants(
    model_path: str,
    output_dir: str,
    quant_types: list[str] | None = None,
    llama_cpp_path: str | None = None,
) -> dict[str, Path]:
    """Export model to multiple GGUF quantization types.

    Parameters
    ----------
    model_path:
        Path to the HuggingFace model directory.
    output_dir:
        Directory for GGUF outputs.
    quant_types:
        List of quantization types. Defaults to [Q4_K_M, Q6_K, Q4_K_S].
    llama_cpp_path:
        Path to llama.cpp directory.

    Returns
    -------
    dict[str, Path]
        Mapping of quant type to output file path.
    """
    if quant_types is None:
        quant_types = ["Q4_K_M", "Q6_K", "Q4_K_S"]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = {}

    for qt in quant_types:
        output_path = out / f"model-{qt.lower()}.gguf"
        try:
            results[qt] = convert_hf_to_gguf(
                model_path=model_path,
                output_path=str(output_path),
                quant_type=qt,
                llama_cpp_path=llama_cpp_path,
            )
        except Exception as exc:
            logger.error("Failed to export %s: %s", qt, exc)

    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Export HF model to GGUF format")
    parser.add_argument("--model-path", required=True, help="HuggingFace model directory")
    parser.add_argument("--output", required=True, help="Output GGUF file path")
    parser.add_argument(
        "--quant-type", default="Q4_K_M",
        choices=SUPPORTED_QUANT_TYPES,
        help="Quantization type (default: Q4_K_M)",
    )
    parser.add_argument("--llama-cpp-path", default=None, help="Path to llama.cpp directory")
    parser.add_argument("--all-quants", action="store_true", help="Export Q4_K_M, Q6_K, and Q4_K_S")
    args = parser.parse_args()

    if args.all_quants:
        export_multiple_quants(
            model_path=args.model_path,
            output_dir=str(Path(args.output).parent),
            llama_cpp_path=args.llama_cpp_path,
        )
    else:
        convert_hf_to_gguf(
            model_path=args.model_path,
            output_path=args.output,
            quant_type=args.quant_type,
            llama_cpp_path=args.llama_cpp_path,
        )


if __name__ == "__main__":
    main()
