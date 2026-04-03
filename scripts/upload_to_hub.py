"""Upload model and model card to HuggingFace Hub.

Usage::

    python scripts/upload_to_hub.py --model-path checkpoints/hf_export \\
        --repo-id username/minigpt-small --private
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)


def upload_to_hub(
    model_path: str,
    repo_id: str,
    private: bool = False,
    commit_message: str = "Upload MiniGPT model",
    create_model_card: bool = True,
    token: str | None = None,
) -> str:
    """Upload a model to the HuggingFace Hub.

    Parameters
    ----------
    model_path:
        Path to the HuggingFace-format model directory.
    repo_id:
        Hub repository ID (e.g. ``"username/model-name"``).
    private:
        Whether to create a private repository.
    commit_message:
        Git commit message for the upload.
    create_model_card:
        Whether to generate a model card if one doesn't exist.
    token:
        HuggingFace API token. If ``None``, uses cached credentials.

    Returns
    -------
    str
        URL of the uploaded model on the Hub.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    model_dir = Path(model_path)

    # Generate model card if needed
    if create_model_card and not (model_dir / "README.md").exists():
        logger.info("Generating model card...")
        from deployment.model_card import generate_model_card

        model_name = repo_id.split("/")[-1] if "/" in repo_id else repo_id
        generate_model_card(
            model_path=model_path,
            model_name=model_name,
            output_path=str(model_dir / "README.md"),
        )

    # Create repo if it doesn't exist
    logger.info("Creating/updating repository: %s", repo_id)
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)

    # Upload all files
    logger.info("Uploading model from %s to %s", model_path, repo_id)
    api.upload_folder(
        folder_path=model_path,
        repo_id=repo_id,
        commit_message=commit_message,
    )

    url = f"https://huggingface.co/{repo_id}"
    logger.info("Upload complete: %s", url)
    return url


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Upload MiniGPT model to HuggingFace Hub")
    parser.add_argument("--model-path", required=True, help="HF model directory path")
    parser.add_argument("--repo-id", required=True, help="Hub repo ID (user/model-name)")
    parser.add_argument("--private", action="store_true", help="Create private repo")
    parser.add_argument("--commit-message", default="Upload MiniGPT model")
    parser.add_argument("--no-model-card", action="store_true")
    parser.add_argument("--token", default=None, help="HuggingFace API token")
    args = parser.parse_args()

    upload_to_hub(
        model_path=args.model_path,
        repo_id=args.repo_id,
        private=args.private,
        commit_message=args.commit_message,
        create_model_card=not args.no_model_card,
        token=args.token,
    )


if __name__ == "__main__":
    main()
