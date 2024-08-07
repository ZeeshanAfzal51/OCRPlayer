# Streamlit code for PDF text extraction and sending to Gemini AI

import streamlit as st
import fitz  # PyMuPDF
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import os
import google.generativeai as genai

# Set up the Google Generative AI client
os.environ["GEMINI_API_KEY"] = "AIzaSyDI2DelJZlGyXEPG3_b-Szo-ixRvaB0ydY"  # Replace with your actual API key
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Define the prompt
prompt = ("the following is OCR extracted text from a single invoice PDF. "
          "Please use the OCR extracted text to give a structured summary. "
          "The structured summary should consider information such as PO Number, Invoice Number, Invoice Amount, Invoice Date, CGST Amount, SGST Amount, IGST Amount, Total Tax Amount, Taxable Amount, TCS Amount, IRN Number, Receiver GSTIN, Receiver Name, Vendor GSTIN, Vendor Name, Remarks and Vendor Code. "
          "If any of this information is not available or present then NA must be denoted next to the value. "
          "Please do not give any additional information.")

# Streamlit file uploader
st.title("Digital Invoicer V1")
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# Function to extract text using PyMuPDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text_data = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        text_data.append(text)
    return text_data

# Function to convert PDF to images and perform OCR
def convert_pdf_to_images_and_ocr(pdf_path):
    images = convert_from_path(pdf_path)
    ocr_results = [pytesseract.image_to_string(image) for image in images]
    return ocr_results

# Function to combine text data and OCR results
def combine_text_and_ocr_results(text_data, ocr_results):
    combined_results = []
    for text, ocr_text in zip(text_data, ocr_results):
        combined_results.append(text + "\n" + ocr_text)
    combined_text = "\n".join(combined_results)
    return combined_text

# Process the uploaded file
if uploaded_file is not None:
    # Save uploaded file temporarily
    pdf_path = "uploaded_file.pdf"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Extract text and OCR results
    text_data = extract_text_from_pdf(pdf_path)
    ocr_results = convert_pdf_to_images_and_ocr(pdf_path)
    combined_text = combine_text_and_ocr_results(text_data, ocr_results)
    
    # Send combined_text to Gemini AI
    input_text = f"{prompt}\n\n{combined_text}"
    
    # Create the model configuration
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    # Initialize the model
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
    )

    # Start a chat session
    chat_session = model.start_chat(
        history=[]
    )

    # Send the combined text as a message
    response = chat_session.send_message(input_text)

    # Display the response from the Gemini AI
    st.header("Gemini API Output")
    st.text(response.text)
