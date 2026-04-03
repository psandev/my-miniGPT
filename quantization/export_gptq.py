"""Convert HuggingFace model to GPTQ 4-bit quantization via auto-gptq.

Backup quantization path in case AWQ has compatibility issues.

Usage::

    python quantization/export_gptq.py --model-path checkpoints/hf_export \\
        --output-dir checkpoints/gptq
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)


def export_gptq(
    model_path: str,
    output_dir: str,
    quant_bits: int = 4,
    group_size: int = 128,
    calibration_dataset: str = "wikitext2",
    n_calibration_samples: int = 128,
    max_calib_seq_len: int = 512,
    use_triton: bool = False,
) -> Path:
    """Convert a HuggingFace model to GPTQ 4-bit format.

    Parameters
    ----------
    model_path:
        Path to the HuggingFace model directory.
    output_dir:
        Output directory for the quantized model.
    quant_bits:
        Number of quantization bits (default 4).
    group_size:
        Quantization group size (default 128).
    calibration_dataset:
        Name of the calibration dataset.
    n_calibration_samples:
        Number of calibration samples.
    max_calib_seq_len:
        Maximum sequence length for calibration.
    use_triton:
        Whether to use Triton kernels.

    Returns
    -------
    Path
        Path to the output directory.
    """
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    from transformers import AutoTokenizer
    from datasets import load_dataset

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model from %s for GPTQ quantization", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare calibration data
    logger.info("Preparing calibration data from %s (%d samples)", calibration_dataset, n_calibration_samples)
    if calibration_dataset == "wikitext2":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", trust_remote_code=True)
        text_field = "text"
    elif calibration_dataset == "c4":
        dataset = load_dataset("allenai/c4", "en", split="train", streaming=True, trust_remote_code=True)
        text_field = "text"
    else:
        dataset = load_dataset(calibration_dataset, split="train", trust_remote_code=True)
        text_field = "text"

    calib_data = []
    for example in dataset:
        text = example.get(text_field, "")
        if not text or len(text.strip()) < 10:
            continue
        tokenized = tokenizer(
            text,
            return_tensors="pt",
            max_length=max_calib_seq_len,
            truncation=True,
        )
        calib_data.append(tokenized.input_ids)
        if len(calib_data) >= n_calibration_samples:
            break

    # Quantization config
    quantize_config = BaseQuantizeConfig(
        bits=quant_bits,
        group_size=group_size,
        desc_act=False,
    )

    logger.info("Quantizing to GPTQ %d-bit (group_size=%d)", quant_bits, group_size)

    model = AutoGPTQForCausalLM.from_pretrained(
        model_path,
        quantize_config=quantize_config,
        trust_remote_code=True,
    )

    model.quantize(calib_data, use_triton=use_triton)

    # Save
    model.save_quantized(str(out_path))
    tokenizer.save_pretrained(str(out_path))

    logger.info("GPTQ model saved to %s", out_path)
    return out_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Export HF model to GPTQ 4-bit")
    parser.add_argument("--model-path", required=True, help="HuggingFace model directory")
    parser.add_argument("--output-dir", required=True, help="Output directory for GPTQ model")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--calibration-dataset", default="wikitext2")
    parser.add_argument("--calibration-samples", type=int, default=128)
    parser.add_argument("--max-calib-seq-len", type=int, default=512)
    parser.add_argument("--use-triton", action="store_true")
    args = parser.parse_args()

    export_gptq(
        model_path=args.model_path,
        output_dir=args.output_dir,
        quant_bits=args.bits,
        group_size=args.group_size,
        calibration_dataset=args.calibration_dataset,
        n_calibration_samples=args.calibration_samples,
        max_calib_seq_len=args.max_calib_seq_len,
        use_triton=args.use_triton,
    )


if __name__ == "__main__":
    main()
