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


# Performs Named Entity Recognition (NER) on given text
def perform_ner(text, model_name):
    try:
        if model_name == "SpaCy English NER":
            doc = spacy_ner_model(text)
            extracted_entities = [
                {
                    "text": ent.text,
                    "type": ent.label_,
                    "start_index": ent.start_char,
                    "end_index": ent.end_char,
                }
                for ent in doc.ents
            ]
        elif model_name == "bert-large-NER":
            entities = ner_model(text)
            extracted_entities = [
                {
                    "text": entity["word"],
                    "type": entity["entity"],
                    "start_index": entity["start"],
                    "end_index": entity["end"],
                }
                for entity in entities
            ]
        else:
            entities = ner_model_bio(text)
            extracted_entities = [
                {
                    "text": entity["word"],
                    "type": entity["entity"],
                    "start_index": entity["start"],
                    "end_index": entity["end"],
                }
                for entity in entities
            ]

        return extracted_entities
    except Exception as e:
        return f"Error Performing NER: {str(e)}"


# this function takes row text , a list of entities with their start and end indices and maps with the assigned color
def highlight_entities_with_colors_and_label_tokenized(text, entities, color_mapping, tokenizer):
    highlighted_text = ""
    current_pos = 0

    for ent in entities:
        start, end, label = (
            ent.get("start_index", 0),
            ent.get("end_index", 0),
            ent.get("type", "0"),
        )
        entity_text = text[start:end]

        # tokenize the text
        encoded_entity = tokenizer.encode(entity_text, add_special_tokens=False)
        tokenized_entity_text = tokenizer.convert_ids_to_tokens(encoded_entity)
        tokenized_entity_length = len(tokenized_entity_text)

        # adding non entity text
        highlighted_text += text[current_pos:start]

        # adding highlighted entity text with color and label on the same time
        color = color_mapping.get(label, "#4D94FF")
        highlighted_text += f"<mark style='background-color:{color}' title='{label}'>{entity_text} ({label})</mark>"

        # Update current position
        current_pos = end

        # add any non remaining non-entity text
        highlighted_text += text[current_pos:]
        return highlighted_text


# Highlight named entities in the given color maping
def highlight_entities(text, entities, model_name):
    try:
        if model_name == "SpaCy English NER":
            doc = spacy_ner_model(text)
            color_mapping = {
                "DATE": "#4D94FF",  # Blue
                "PERSON": "#4CAF50",  # Green
                "EVENT": "#FF6666",  # Salmon
                "FAC": "#66B2FF",  # Sky Blue
                "GPE": "#FFCC99",  # Light Apricot
                "LANGUAGE": "#FF80BF",  # Pink
                "LAW": "#66FF99",  # Mint
                "LOC": "#809FFF",  # Lavender Blue
                "MONEY": "#FFFF99",  # Light Yellow
                "NORP": "#808000",  # Olive Green
                "ORDINAL": "#FF9999",  # Misty Rose
                "ORG": "#FFB366",  # Light Peach
                "PERCENT": "#FF99FF",  # Orchid
                "PRODUCT": "#FF6666",  # Salmon
                "QUANTITY": "#CC99FF",  # Pastel Purple
                "TIME": "#FFD54F",  # Amber
                "WORK_OF_ART": "#FFC266",  # Light Orange
                "CARDINAL": "#008080",  # Teal
            }

            options = {
                "ents": [entity["type"] for entity in entities],
                "colors": color_mapping,
            }
            html = displacy.render(doc, style="ent", options=options, page=True)
            colored_text = html
            return colored_text
        else:
            color_mapping = {
                "O": "pink",
                "B-MIS": "red",
                "I-MIS": "brown",
                "B-PER": "green",
                "I-PER": "#FFD54F",
                "B-ORG": "orange",
                "I-ORG": "#FF6666",
                "B-LOC": "purple",
                "I-LOC": "#FFCC99",
            }
            highlighted_example = highlight_entities_with_colors_and_label_tokenized(
                text, entities, color_mapping, tokenizer
            )
            return highlighted_example
    except Exception as e:
        return f"Error highlighted entities: {str(e)}"


# Summarize text
def summarized_text(input_text):
    summarized_text = summarization_pipeline(
        input_text,
        max_length=150,
        min_length=50,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True,
    )
    summarized_text = summarized_text[0]["summary_text"]

    return summarized_text


def image_ner_tool(file, model_name):
    reformatted_ner_output = ""
    try:
        if isinstance(file, str):
            with open(file, "rb") as file_stream:
                file_bytes = file_stream.read()
        else:
            file_bytes = file.getvalue()
        text = extract_text_from_image_or_pdf(file_bytes)
        entities = perform_ner(text, model_name)
        highlighted_text = highlight_entities(text, entities, model_name)

        reformatted_ner_output = json.dumps(entities, indent=2)

        summary = summarized_text(text)

        return text, highlighted_text, reformatted_ner_output, summary
    except Exception as e:
        error_message = f"Error processing file:{str(e)}"
        return error_message, "", reformatted_ner_output


# Gradio

# # adding custom css
# css ="""

# """

# # adding custom js
# js ="""

# """

with gr.Blocks(theme='shivi/calm_seafoam') as demo:
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
            with gr.Tab("Highlighted Entitiled"):
                output2 = gr.HTML("Summarize Text")
            with gr.Tab("Summarized Text"):
                output3 = gr.HTML("Summarize Text")
            with gr.Tab("Named Entities Extracted"):
                output4 = gr.HTML(label="Named Entities")

    btn.click(
        image_ner_tool,
        [text1, model],
        [output1, output2, output4, output3],
    )

demo.launch(share=True)
