import gradio as gr
import PyPDF2
from PyPDF2 import PdfReader
from io import BytesIO
import pytesseract
from PIL import Image
import spacy
import json

from transformers import pipeline
from PyPDF2 import PdfReader


# initiate model
ner_model = pipeline("token-classification", model="dslim/bert-large-NER")
summarization_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")
ner_models = {
    "bert-large-NER": "dslim/bert-large-NER",
    "bioNER": "d4data/biomedical-ner-all",
    "SpaCy English NER": "en_core_web_trf",
}
spacy_ner_model = spacy.load(ner_models["SpaCy English NER"])
ner_model_bio = pipeline("token-classification", model="d4data/biomedical-ner-all")

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
from spacy import displacy


# Extracting text from pdf & image file


def extract_text_from_pdf(pdf_bytes):
    text = ""
    pdf_file = BytesIO(pdf_bytes)
    pdf_reader = PdfReader(pdf_file)

    for page_number in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_number]
        text += page.extract_text()
    return text


def extract_text_from_image_or_pdf(file_bytes):
    try:
        if file_bytes.startswith(b"%PDF"):
            text = extract_text_from_pdf(file_bytes)
            print(text)
        else:
            image = Image.open(BytesIO(file_bytes))
            text = pytesseract.image_to_string(image)

        return text
    except Exception as e:
        return f"Error extracting file"


def image_ner_tool(file, model_name):
    reformatted_ner_output = ""
    try:
        if isinstance(file, str):
            with open(file, "rb") as file_stream:
                file_bytes = file_stream.read()
        else:
            file_bytes = file.getvalue()
        text = extract_text_from_image_or_pdf(file_bytes)
        return text
    except Exception as e:
        error_message = f"Error processing file:{str(e)}"
        return error_message, "", reformatted_ner_output


# Gradio
with gr.Blocks() as demo:
    gr.Markdown(
        """
        <p style="text-align: center; font-weight: bold; font-size: 44px;">
        Intelligent Document Processing
        </p>
        <p style="text-align: center;">
        Upload a PDF or an image file to extract text and identify named entities
        </p>
        """
    )

    with gr.Row() as row:
        with gr.Column():
            text1 = gr.File(label="Upload File")
            model = gr.Dropdown(list(ner_models.keys()), label="Select NER Model")
            btn = gr.Button("Submit")

        with gr.Column():
            with gr.Tab("Extract Text"):
                output1 = gr.Textbox(label="Extracted Text", container=True)

    btn.click(
        image_ner_tool,
        [text1, model],
        [output1],
    )

demo.launch(share=True)
