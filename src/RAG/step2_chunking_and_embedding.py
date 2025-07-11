import os
from pathlib import Path
from tempfile import mkdtemp
import logging
import torch
import torch.backends as tback

from langchain_docling.loader import ExportType
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline  # Import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_docling import DoclingLoader
from docling_core.transforms.chunker.hierarchical_chunker import DocMeta
from docling.datamodel.document import DoclingDocument
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from langchain_core.prompts import PromptTemplate
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_milvus import Milvus

#logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




class ChunkingAndEmbedding:
    def __init__(self, data_path, source_name, embed_model_address, vector_database_uri, max_tokens=256):
        self.data_path = data_path
        self.source_name = source_name
        self.source_file_name = source_name + ".docling.json"
        self.source_address = os.path.join(self.data_path, self.source_file_name)
        self.export_type = ExportType.DOC_CHUNKS
        self.embed_model_address = embed_model_address
        self.vector_database_uri = vector_database_uri
        self.chunker = None
        self.tokenizer = None
        self.docling_loader = None
        self.max_tokens = max_tokens

        #Setup the embedding model
        logger.info(f"Setting up embedding model from {self.embed_model_address}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.embed_model_address, local_files_only=True)
        self.docling_tokenizer = HuggingFaceTokenizer(tokenizer=self.tokenizer, max_tokens=self.max_tokens)
        self.chunker = HybridChunker(tokenizer=self.docling_tokenizer)
        self.embedding = HuggingFaceEmbeddings(model_name=self.embed_model_address)


    def read_and_chunk_data(self):
        # Step 1: Read the docling data from docling_checkpoint
        logger.info("Step 1: Reading docling data from checkpoint...")
        self.docling_loader = DoclingLoader(
            file_path=self.source_address,
            export_type=self.export_type,
            chunker=self.chunker,
        )
        # Step 2: Chunck the data
        logger.info("Step 2: Chuncking the data...")
        docs = self.docling_loader.load()
        return docs


    def show_samples(self, docs):
        # Check the max number of tokens in the document
        max_tokens = 0
        avg_tokens = 0
        for i, chunk in enumerate(docs):
            text = chunk.page_content
            text_tokens = len(self.tokenizer.tokenize(text, max_length=None))
            max_tokens = max([max_tokens, text_tokens])
            avg_tokens += text_tokens
        avg_tokens /= len(docs)
        logger.info(f"Max tokens: {max_tokens}")
        logger.info(f"Avg tokens: {avg_tokens}")

        for d in docs[:3]:
            logger.info(f"- {d.page_content=}")
        logger.info("...")


    def setup_vector_database(self, splits):
        vectorstore = Milvus.from_documents(
            documents=splits,
            embedding=self.embedding,
            collection_name="docling_demo",
            connection_args={"uri": self.vector_database_uri},
            index_params={"index_type": "FLAT"},
            drop_old=True,
        )

    def create_vector_database(self):
        splits = self.read_and_chunk_data()
        self.show_samples(splits)
        self.setup_vector_database(splits)