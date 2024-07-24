import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_pdf_text(pdf_docs):
    logging.info("Starting to extract text from PDFs")
    start_time = time.time()
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        except Exception as e:
            logging.error(f"Error processing PDF file {pdf.name}: {e}")
    end_time = time.time()
    logging.info(f"Extracted text from PDFs in {end_time - start_time:.2f} seconds")
    return text

def get_text_chunks(text):
    logging.info("Starting to split text into chunks")
    start_time = time.time()
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    end_time = time.time()
    logging.info(f"Split text into chunks in {end_time - start_time:.2f} seconds")
    return chunks

def get_vectorstore(text_chunks):
    logging.info("Starting to create vector store")
    start_time = time.time()
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    end_time = time.time()
    logging.info(f"Created vector store in {end_time - start_time:.2f} seconds")
    return vectorstore

def get_conversation_chain(vectorstore):
    logging.info("Starting to create conversation chain")
    start_time = time.time()
    llm = ChatOpenAI()
    logging.info(f"Created LLM: {llm}") 
    
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    logging.info(f"Created memory: {memory}")
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    end_time = time.time()
    logging.info(f"Created conversation chain in {end_time - start_time:.2f} seconds")
    return conversation_chain

def handle_userinput(user_question):
    logging.info(f"Handling user input: {user_question}")
    start_time = time.time()
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response.get('chat_history', [])
    end_time = time.time()
    logging.info(f"Handled user input in {end_time - start_time:.2f} seconds")

    for i, message in enumerate(st.session_state.chat_history):
        if isinstance(message, dict):
            content = message.get('content', '')
        else:
            # Jika message bukan dictionary, sesuaikan dengan jenis objeknya
            content = getattr(message, 'content', '')

        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing"):
                    logging.info("Processing started")
                    overall_start_time = time.time()
                    
                    # get pdf text
                    raw_text = get_pdf_text(pdf_docs)
                    logging.info(f"Extracted text from PDFs: {raw_text[:100]}...")

                    # get the text chunks
                    text_chunks = get_text_chunks(raw_text)
                    logging.info(f"Split text into chunks: {len(text_chunks)} chunks")

                    # create vector store
                    vectorstore = get_vectorstore(text_chunks)
                    logging.info(f"Created vector store: {vectorstore}")

                    # create conversation chain
                    st.session_state.conversation = get_conversation_chain(vectorstore)

                    overall_end_time = time.time()
                    logging.info(f"Processing completed in {overall_end_time - overall_start_time:.2f} seconds")
            else:
                st.warning("Please upload at least one PDF file.")

if __name__ == '__main__':
    main()
