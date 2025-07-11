import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from step1_extraction import PDFExtractor
from step2_chunking_and_embedding import ChunkingAndEmbedding
from model_downloader.download_models import save_basic_doc_ling_models, save_gen_model, save_embedding_model, save_image_caption_model

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


if __name__ == "__main__":
    # Load configuration and status
    config_dict = load_config()
    status_data = load_status()
    
    # Update system status to indicate start
    update_status(status_data, {
        'system_status.overall_status': 'initializing'
    })
    
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
    
    # Step 1: Initialize PDF extractor
    source_address = os.path.join(config_dict["base_address"], config_dict["src_config"]["source_address"])
    source_name = config_dict["src_config"]["source_name"]
    page_range = config_dict["src_config"]["page_range"]
    docling_checkpoint_dir = os.path.join(config_dict["base_address"], config_dict["docling_checkpoint_dir_name"])
    
    logger.info(f"Initializing PDF extractor for source file {source_name}")
    pdf_extractor = PDFExtractor(source_address, source_name, page_range, docling_checkpoint_dir, docling_models_save_dir)
    pdf_extractor.extract_pdf_pages_and_save_as_json()
    logger.info("PDF pages extracted and saved as JSON")
    
    # Update status for PDF extraction and checkpoint creation
    update_status(status_data, {
        'processing.pdf_extraction.completed': True,
        'processing.pdf_extraction.last_updated': datetime.now().isoformat(),
        'checkpoints.docling_checkpoint.created': True,
        'checkpoints.docling_checkpoint.last_updated': datetime.now().isoformat(),
        'checkpoints.docling_checkpoint.path': docling_checkpoint_dir
    })

    # Step 2: Create vector database
    logger.info(f"Initializing ChunkingAndEmbedding for source file {source_name}")
    vector_database_address = os.path.join(config_dict["base_address"], config_dict["vector_database_dir_name"], "milvus.db")
    chunking_and_embedding = ChunkingAndEmbedding(docling_checkpoint_dir, source_name, embedding_model_save_dir, vector_database_address)
    chunking_and_embedding.create_vector_database()
    logger.info("Vector database created")
    
    # Update status for vector database creation and chunking/embedding
    update_status(status_data, {
        'databases.vector_database.created': True,
        'databases.vector_database.last_updated': datetime.now().isoformat(),
        'processing.chunking_and_embedding.completed': True,
        'processing.chunking_and_embedding.last_updated': datetime.now().isoformat()
    })

    logger.info("RAG system initialized successfully")
    
    # Update final system status
    update_status(status_data, {
        'system_status.overall_status': 'completed'
    })

    # Create the retrieval database 

