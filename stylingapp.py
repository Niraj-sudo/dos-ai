import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain import  LLMChain, PromptTemplate
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline
import base64
from docx import Document

def get_pdf_text(pdf_docs):
    text = ""

    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
        text=text.replace('\n', ' ')
    return text
def model():
    # Define Model ID
    model_id = "OpenAssistant/llama2-13b-orca-8k-3319"
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Build HF Transformers pipeline 
    pipelines = pipeline(
        "text-generation", 
        model=model_id,
        tokenizer=tokenizer,
        device_map="auto",
        max_length=400,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id
    )
    llm = HuggingFacePipeline(pipeline=pipelines)
    return llm
def style_llm(text):
    template = """You will reword my response. Replace slang with formal language and fix grammar error. 
    Human: {human_input}
    Chatbot:"""
    prompt = PromptTemplate(
        input_variables=["human_input"], template=template
    )
    llm_chain = LLMChain(
    llm=model(),
    prompt=prompt,
    verbose=True,
    )
    llm_chain(text)
    return text
def create_word_document():
    doc = Document()
    doc.add_heading('Hello World', 0)
    return doc

def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{bin_file}">Download {file_label}</a>'
    return href
def main():
    load_dotenv()
    st.set_page_config(page_title="Deloitte",
                       page_icon=":books:")
    st.header("Professor Chicago")
    pdf_docs = st.file_uploader(
        "Upload your PDFs here and click on 'Translate'", accept_multiple_files=True)
    if st.button("Translate"):
        with st.spinner("Processing"):
            text= get_pdf_text(pdf_docs)
            text=style_llm(text)
            doc = Document()
            style = doc.styles['Normal']
            font = style.font
            font.name = 'TimeRoman'
            doc.add_heading('Gen Z human right PDF to Chicago format Word Styling document', 0)
            doc.add_paragraph(text)
            doc.add_page_break()
            name='Chicago format Word Styling document'
            doc.save(f'{name}.docx')
            st.write('Click the link to download the document with Chicago style formating')
            st.markdown(get_binary_file_downloader_html(f'{name}.docx', f'{name}'), unsafe_allow_html=True)
            st.write('Here is a sample of AI translation:')
            st.write(text)
if __name__ == '__main__':
    main()