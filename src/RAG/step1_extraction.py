import os
from pathlib import Path
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
import json


class PDFExtractor:
    def __init__(self, source_address, source_name, page_range, output_dir, docling_models_dir):
        self.source_address = source_address
        self.source_name = source_name
        self.source_file_name = source_name + ".pdf"
        self.page_range = page_range
        self.output_dir = output_dir
        self.docling_result = None
        self.docling_models_dir = docling_models_dir

    def extract_pdf_pages(self):
        opts = PdfPipelineOptions(artifacts_path=self.docling_models_dir)
        converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
        self.docling_result = converter.convert(os.path.join(self.source_address, self.source_file_name), page_range=self.page_range)


    def save_as_json(self):
        if self.docling_result is not None:
            #Save the result as docling json
            output_file_name = os.path.join(self.output_dir, self.source_name + ".docling.json")
            doc_json = self.docling_result.document.model_dump_json(indent=2)  # pydantic helper
            #Create the folder if it doesn't exist
            os.makedirs(self.output_dir, exist_ok=True)
            Path(output_file_name).write_text(doc_json, encoding="utf-8")
        else:
            raise ValueError("No docling result to save")


    def extract_pdf_pages_and_save_as_json(self):
        self.extract_pdf_pages()
        self.save_as_json()

