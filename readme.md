# Chatbot with PDFs

## Description

This project is a chatbot application built with Streamlit and Langchain that interacts with users by processing and analyzing PDF documents. It uses OpenAI for natural language understanding and FAISS for vector storage and retrieval. Additionally, it includes a caching mechanism to enhance performance by storing and retrieving previously answered questions.

## Installation

1. **Clone the Repository**

```bash
git clone https://github.com/ChristoliniAngelo/Chatbot-Rag-Cache.git
cd Chatbot-Rag-Cache
```

2. **Set Up Environment**

Create and activate a virtual environment:

```bash
python -m venv venv

source venv/bin/activate 
or 
# On Windows use 
venv\Scripts\activate
```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```
4. **Set Up Environment Variables**

Create a `.env` file in the project root with the following content:

```.env
OPENAI_API_KEY=your_openai_api_key
```

## Usage

1. **Run the Application**

   ```bash
   streamlit run app.py
   ```
2. **Upload PDF Documents**
   Use the sidebar in the Streamlit app to upload your PDF files and click on 'Process'.
3. **Interact with the Chatbot**
   Enter your questions in the text input field to get answers based on the content of the uploaded PDFs.
4. **Caching Mechanism**
   The application includes a caching method to store and retrieve responses to previously asked questions. The cache is saved in a JSON file (`cache.json`). When a question is asked, the system first checks if it has already been answered and stored in the cache. If so, it retrieves the cached response. Otherwise, it rephrases the question using OpenAI's GPT-4o-mini model, processes it, and then stores the response in the cache for future use.

## Dependencies

The project requires the following Python packages:

* `streamlit`
* `openai`
* `PyPDF2`
* `langchain`
* `python-dotenv`
* `faiss-cpu` (or `faiss-gpu` if using GPU)

These dependencies are listed in `requirements.txt`.

## Contributing

Feel free to fork the repository and submit pull requests with improvements or fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
