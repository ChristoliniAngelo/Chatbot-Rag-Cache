import openai
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from htmlTemplates import user_template, bot_template
import json
import re
import logging
import time
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CACHE_FILE = 'cache.json'

def ensure_cache_file():
    if not os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'w') as f:
            json.dump({}, f)

def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logging.error(f"JSONDecodeError: {e} - The cache file might be empty or corrupted.")
            return {}
        except Exception as e:
            logging.error(f"An error occurred while loading the cache: {e}")
            return {}
    return {}

def save_cache(cache):
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f, indent=4)
    except Exception as e:
        logging.error(f"An error occurred while saving the cache: {e}")

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
    llm = ChatOpenAI(
        model="gpt-4o-mini",
    )
    logging.info(f"Created LLM: {llm}")

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    logging.info(f"Created memory: {memory}")

    prompt_template = """
    You are Esti, a helpful assistant.
    Answer the question with Bahasa Indonesia.
    Use the following context to answer the question at the end.
    If you don't know the answer, just say you don't know. Do not try to fabricate an answer.
    If the question is not related to the context, politely inform that you are designed to answer only questions related to the context.

    Context:
    {context}

    Question: {question}

    Helpful answer in markdown.
    """
    PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={'k': 5}),
        memory=memory,
        combine_docs_chain_kwargs={'prompt': PROMPT}
    )
    end_time = time.time()
    logging.info(f"Created conversation chain in {end_time - start_time:.2f} seconds")
    return conversation_chain

def normalize_question(question):
    # Convert to lowercase
    question = question.lower()
    # Remove punctuation and extra spaces
    question = re.sub(r'[^\w\s]', '', question)
    question = re.sub(r'\s+', ' ', question).strip()
    return question

def rephrase_question_and_get_response(question):
    cache = load_cache()
    
    normalized_question = normalize_question(question)
    # Check if the normalized question is already in the cache
    if normalized_question in cache:
        return cache[normalized_question]

    # Rephrase the question using OpenAI
    openai.api_key = os.getenv("OPENAI_API_KEY")
    rephrase_question_prompt = f"""
    I need you to rephrase the following question to make it clearer and more precise.
    The rephrased question should convey the same meaning but be structured differently. 
    Make sure it is a standalone question that maintains the core intent of the original question.

    Original Question: {question}
    Rephrased Question:
    """

    
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": rephrase_question_prompt}
        ],
        temperature=0.2
    )
    
    rephrased_question = response.choices[0].message['content'].strip()
    print(f"Rephrased question: {rephrased_question}")
    rephrased_normalized_question = normalize_question(rephrased_question)

    # Get the response from the conversation chain
    response = st.session_state.conversation({'question': rephrased_normalized_question})
    answer = response.get('answer', "No answer found.")
    
    # Update cache
    cache[normalized_question] = answer
    save_cache(cache)
    
    return answer

def handle_userinput(user_question):
    logging.info(f"Handling user input: {user_question}")
    start_time = time.time()

    # Rephrase the user question and get the response
    response = rephrase_question_and_get_response(user_question)

    # Display the response
    st.markdown(bot_template.replace("{{MSG}}", response), unsafe_allow_html=True)

    end_time = time.time()
    logging.info(f"Handled user input in {end_time - start_time:.2f} seconds")


def main():
    ensure_cache_file()
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

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

                    # Get PDF text
                    raw_text = get_pdf_text(pdf_docs)
                    logging.info(f"Extracted text from PDFs: {raw_text[:100]}...")

                    # Get the text chunks
                    text_chunks = get_text_chunks(raw_text)
                    logging.info(f"Split text into chunks: {len(text_chunks)} chunks")

                    # Create vector store
                    vectorstore = get_vectorstore(text_chunks)
                    logging.info(f"Created vector store: {vectorstore}")

                    # Create conversation chain
                    st.session_state.conversation = get_conversation_chain(vectorstore)

                    overall_end_time = time.time()
                    logging.info(f"Processing completed in {overall_end_time - overall_start_time:.2f} seconds")
            else:
                st.warning("Please upload at least one PDF file.")

if __name__ == '__main__':
    main()
