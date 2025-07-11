from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_milvus import Milvus
import logging
import json
import os


class Retrieval:
    def __init__(self, gen_model_dir, embed_model_dir, vector_database_dir, image_caption_model_dir):
        self.gen_model_dir = gen_model_dir
        self.embed_model_dir = embed_model_dir
        self.vector_database_dir = vector_database_dir
        self.image_caption_model_dir = image_caption_model_dir
        self.logger = logging.getLogger(__name__)
        self.logger.info("Retrieval system initialized")
        self.logger.info(f"Loading embedding model from {self.embed_model_dir}")
        #Embeding model
        self.embedding = HuggingFaceEmbeddings(model_name=self.embed_model_dir)
        self.logger.info(f"Embedding model loaded")
        #Generation model
        self.logger.info(f"Loading generation model from {self.gen_model_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.gen_model_dir, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.gen_model_dir,
            device_map="auto",           # put layers on available GPU(s) if any
            torch_dtype="auto",          # FP16/FP32 automatically
        )
        self.logger.info(f"Generation model loaded")
        #Generation pipeline
        self.logger.info(f"Creating generation pipeline")
        self.generator_llm = self._create_generation_pipeline()
        self.logger.info(f"Generation pipeline created")
        #Vector database
        self.logger.info(f"Loading vector database from {self.vector_database_dir}")
        self.vectorstore = Milvus(embedding_function=self.embedding, collection_name="docling_demo", connection_args={"uri": self.vector_database_dir}, index_params={"index_type": "FLAT"})
        self.logger.info(f"Vector database loaded")
        #Retrieval chain
        self.logger.info(f"Creating retrieval chain")
        self.top_k = 3
        self.prompt = PromptTemplate.from_template(
            "Context information is below.\n---------------------\n{context}\n---------------------\nGiven the context information and no prior knowledge, answer the query.\nQuery: {input}\n",
        )
        self.rag_chain = self._create_retrieval_chain()
        self.logger.info(f"Retrieval chain created")


    def _clip_text(self, text, threshold=100):
        return f"{text[:threshold]}..." if len(text) > threshold else text

    def _extract_answer(self, text, question):
        # find the question text and return everything after that
        if question in text:
            return text[text.index(question) + len(question):]
        else:
            return text

    def _create_generation_pipeline(self):
        # 2. Build a transformers generation pipeline
        generation_pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            top_k=10,
            top_p=0.95,
            typical_p=0.95,
            temperature=0.01,
            repetition_penalty=1.03,
        )
        generator_llm = HuggingFacePipeline(pipeline=generation_pipe)
        return generator_llm

    def _create_retrieval_chain(self):
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.top_k})
        question_answer_chain = create_stuff_documents_chain(self.generator_llm, self.prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        return rag_chain



def load_status():
    """Load status from JSON file"""
    #logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
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



if __name__ == "__main__":
    #Create a retrieval class
    status_dict = load_status()
    gen_model_dir = status_dict["models"]["generation_model"]["path"]
    embed_model_dir = status_dict["models"]["embedding_model"]["path"]
    vector_database_dir = status_dict["databases"]["vector_database"]["path"]
    image_caption_model_dir = status_dict["models"]["image_caption_model"]["path"]
    rag_system = Retrieval(gen_model_dir, embed_model_dir, vector_database_dir, image_caption_model_dir)


    QUESTION = "How much operational income rose in 2025?"


    #test the model
    print("\n\n\n--------------------------------")
    print("Test the model without RAG...")
    print(rag_system.generator_llm.invoke(QUESTION))

    print("\n\n\n--------------------------------")
    print("Test the model with RAG...")
    resp_dict = rag_system.rag_chain.invoke({"input": QUESTION})
    # print(resp_dict)
    # clipped_answer = clip_text(resp_dict["answer"], threshold=1024)
    answer_with_context = resp_dict["answer"]
    answer = rag_system._extract_answer(answer_with_context, QUESTION)

    print(f"Question:\n{resp_dict['input']}\nLLM answer:\n{answer}")
    print("\n\n\n--------------------------------")
    print("Printing the context...")
    for i, doc in enumerate(resp_dict["context"]):
        print()
        print(f"Source {i + 1}:")
        print(f"  text: {json.dumps(rag_system._clip_text(doc.page_content, threshold=350))}")
        # for key in doc.metadata:
        #     if key != "pk":
        #         val = doc.metadata.get(key)
        #         clipped_val = clip_text(val) if isinstance(val, str) else val
        #         print(f"  {key}: {clipped_val}")

