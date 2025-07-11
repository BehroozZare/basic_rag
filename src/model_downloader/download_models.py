"""
Data preparation module for fine-tuning.
"""
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from docling.utils.model_downloader import download_models
from sentence_transformers import SentenceTransformer
from transformers import BlipProcessor, BlipForConditionalGeneration   # >=4.26
import json
from datetime import datetime

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



def load_status():
    """Load status from JSON file"""
    status_path = os.path.join(os.path.dirname(__file__), '..', '..', 'status', 'status.json')
    try:
        with open(status_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Status file not found at {status_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing status file: {e}")
        raise

def update_status(status_data, updates):
    """Update status with new information"""
    status_path = os.path.join(os.path.dirname(__file__), '..', '..', 'status', 'status.json')
    
    # Apply updates
    for key_path, value in updates.items():
        keys = key_path.split('.')
        current = status_data
        for key in keys[:-1]:
            current = current[key]
        current[keys[-1]] = value
    
    # Update system status
    status_data['system_status']['last_run'] = datetime.now().isoformat()
    
    # Save updated status
    try:
        with open(status_path, 'w') as f:
            json.dump(status_data, f, indent=4)
        logger.info(f"Status updated: {updates}")
    except Exception as e:
        logger.error(f"Error saving status: {e}")
        
def load_config():
    """Load configuration from JSON file"""
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'status', 'config.json')
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Config file not found at {config_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing config file: {e}")
        raise


if __name__ == "__main__":
    config_dict = load_config()
    status_data = load_status()
    save_models_dir = os.path.join(config_dict["base_address"], config_dict["save_models_dir_name"])
    docling_models_save_dir = os.path.join(save_models_dir, config_dict["docling_models_dir_name"])
    embedding_model_save_dir = os.path.join(save_models_dir, config_dict["embedding_model_dir_name"])
    image_caption_model_save_dir = os.path.join(save_models_dir, config_dict["image_caption_model_dir_name"])
    gen_model_save_dir = os.path.join(save_models_dir, config_dict["gen_model_dir_name"])



    if config_dict["download_docling_models"]:
        # Save docling models
        logger.info(f"Saving basic docling models to {docling_models_save_dir}")
        save_basic_doc_ling_models(docling_models_save_dir)
        logger.info("Docling models saved")
        
        # Update status
        update_status(status_data, {
            'models.docling_models.downloaded': True,
            'models.docling_models.last_updated': datetime.now().isoformat(),
            'models.docling_models.path': docling_models_save_dir
        })
    
    if config_dict["download_embedding_model"]:
        # Save embedding model
        logger.info(f"Saving embedding model to {embedding_model_save_dir}")
        save_embedding_model(model_id=config_dict["embedding_model_config"]["model_id"], output_dir=embedding_model_save_dir)
        logger.info("Embedding model saved")
        
        # Update status
        update_status(status_data, {
            'models.embedding_model.downloaded': True,
            'models.embedding_model.last_updated': datetime.now().isoformat(),
            'models.embedding_model.path': embedding_model_save_dir
        })

    if config_dict["download_image_caption_model"]:
        # Save image caption model
        logger.info(f"Saving image caption model to {image_caption_model_save_dir}")
        save_image_caption_model(model_id=config_dict["image_caption_model_config"]["model_id"], output_dir=image_caption_model_save_dir)
        logger.info("Image caption model saved")
        
        # Update status
    
    if config_dict["download_gen_model"]:
        # Save generation model
        logger.info(f"Saving model to {gen_model_save_dir}")
        save_gen_model(model_id=config_dict["gen_model_config"]["model_id"], output_dir=gen_model_save_dir)
        logger.info("Gen model saved")
        
        # Update status
        update_status(status_data, {
            'models.generation_model.downloaded': True,
            'models.generation_model.last_updated': datetime.now().isoformat(),
            'models.generation_model.path': gen_model_save_dir
        })