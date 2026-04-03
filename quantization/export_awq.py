"""Convert HuggingFace model to AWQ 4-bit quantization via autoawq.

Uses 128-sample calibration and targets Marlin kernel compatibility
for maximum inference throughput.

Usage::

    python quantization/export_awq.py --model-path checkpoints/hf_export \\
        --output-dir checkpoints/awq
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)


def export_awq(
    model_path: str,
    output_dir: str,
    n_calibration_samples: int = 128,
    quant_bits: int = 4,
    group_size: int = 128,
    zero_point: bool = True,
    use_marlin: bool = True,
    calibration_dataset: str = "pileval",
    max_calib_seq_len: int = 512,
) -> Path:
    """Convert a HuggingFace model to AWQ 4-bit format.

    Parameters
    ----------
    model_path:
        Path to the HuggingFace model directory.
    output_dir:
        Output directory for the quantized model.
    n_calibration_samples:
        Number of calibration samples for AWQ quantization.
    quant_bits:
        Number of quantization bits (default 4).
    group_size:
        Quantization group size (default 128).
    zero_point:
        Whether to use zero-point quantization.
    use_marlin:
        If ``True``, target Marlin kernel compatibility for vLLM.
    calibration_dataset:
        Calibration dataset name (``"pileval"`` or ``"wikitext"``).
    max_calib_seq_len:
        Maximum sequence length for calibration samples.

    Returns
    -------
    Path
        Path to the output directory containing the AWQ model.
    """
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model from %s for AWQ quantization", model_path)
    model = AutoAWQForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Quantization config
    quant_config = {
        "w_bit": quant_bits,
        "q_group_size": group_size,
        "zero_point": zero_point,
        "version": "marlin" if use_marlin else "gemm",
    }

    logger.info(
        "Quantizing to AWQ %d-bit (group_size=%d, marlin=%s, samples=%d)",
        quant_bits, group_size, use_marlin, n_calibration_samples,
    )

    # Run calibration and quantization
    model.quantize(
        tokenizer,
        quant_config=quant_config,
        calib_data=calibration_dataset,
        n_samples=n_calibration_samples,
        max_calib_seq_len=max_calib_seq_len,
    )

    # Save quantized model
    model.save_quantized(str(out_path))
    tokenizer.save_pretrained(str(out_path))

    logger.info("AWQ model saved to %s", out_path)

    # Verify the export
    _verify_awq_model(str(out_path))

    return out_path


def _verify_awq_model(model_path: str) -> None:
    """Quick verification that the AWQ model loads correctly."""
    try:
        from awq import AutoAWQForCausalLM

        model = AutoAWQForCausalLM.from_quantized(model_path, fuse_layers=False)
        logger.info("AWQ model verification passed (loaded successfully)")
        del model
    except Exception as exc:
        logger.warning("AWQ model verification failed: %s", exc)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Export HF model to AWQ 4-bit")
    parser.add_argument("--model-path", required=True, help="HuggingFace model directory")
    parser.add_argument("--output-dir", required=True, help="Output directory for AWQ model")
    parser.add_argument("--calibration-samples", type=int, default=128)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--no-marlin", action="store_true", help="Disable Marlin kernel targeting")
    parser.add_argument("--calibration-dataset", default="pileval")
    parser.add_argument("--max-calib-seq-len", type=int, default=512)
    args = parser.parse_args()

    export_awq(
        model_path=args.model_path,
        output_dir=args.output_dir,
        n_calibration_samples=args.calibration_samples,
        quant_bits=args.bits,
        group_size=args.group_size,
        use_marlin=not args.no_marlin,
        calibration_dataset=args.calibration_dataset,
        max_calib_seq_len=args.max_calib_seq_len,
    )


if __name__ == "__main__":
    main()
