"""
Data preparation module for fine-tuning.
"""
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from docling.utils.model_downloader import download_models
from sentence_transformers import SentenceTransformer
from transformers import BlipProcessor, BlipForConditionalGeneration   # >=4.26


from huggingface_hub import login
import os
#logger
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Read token from config file
def read_token_from_file():
    # Find the project root by looking for config/token.txt
    current_dir = Path(__file__).parent
    project_root = None
    
    # Search upwards for the config directory
    for parent in [current_dir] + list(current_dir.parents):
        if (parent / "config" / "token.txt").exists():
            project_root = parent
            break
    
    if project_root is None:
        # Fallback: try to find the project root by looking for common project files
        for parent in [current_dir] + list(current_dir.parents):
            if (parent / ".gitignore").exists() or (parent / "src").exists():
                project_root = parent
                break
    
    if project_root is None:
        raise FileNotFoundError("Could not find project root directory")
    
    token_file = project_root / "config" / "token.txt"
    try:
        with open(token_file, 'r') as f:
            token = f.read().strip()
        return token
    except FileNotFoundError:
        logger.error(f"Token file not found at {token_file}")
        raise
    except Exception as e:
        logger.error(f"Error reading token file: {e}")
        raise

# Login with token from file
token = read_token_from_file()
login(
  token=token,
  add_to_git_credential=False
)


def save_basic_doc_ling_models(output_dir):
    path_dir = Path(output_dir)
    downloaded = download_models(output_dir=path_dir)

def save_gen_model(model_id: str, output_dir: str):
    """Main training function."""
    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model saved to {output_dir}")

def save_embedding_model(model_id: str, output_dir: str):
    model = SentenceTransformer(model_id)
    model.save(output_dir)  
    logger.info(f"Model saved to {output_dir}")


def save_image_caption_model(model_id: str = "Salesforce/blip-image-captioning-base",
                             output_dir: str | Path = "models/blip_base"):
    """
    Download the BLIP image-captioning checkpoint + processor once and cache
    them locally so that DocLing (or any Transformers code) can be run with
    TRANSFORMERS_OFFLINE=1 later on.

    Args:
        model_id  – HF repo to pull.  Defaults to the BLIP base captioner.
        output_dir – Where the weights + configs will be written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1️⃣ Fetch from the Hub (will respect your `huggingface_hub.login()` token).
    processor = BlipProcessor.from_pretrained(model_id)
    model     = BlipForConditionalGeneration.from_pretrained(model_id)

    # 2️⃣ Write a *self-contained* copy for offline use.
    processor.save_pretrained(output_dir)
    model.save_pretrained(output_dir)

    logger.info(f"BLIP image-captioning model saved to {output_dir.resolve()}")