# Chatbot with PDFs

## Description

This project is a chatbot application built with Streamlit and Langchain that interacts with users by processing and analyzing PDF documents. It uses OpenAI for natural language understanding and FAISS for vector storage and retrieval.

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
   ```
2. **Set Up Environment**

   - Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```
4. **Set Up Environment Variables**

   Create a `.env` file in the project root with the following content:

   ```env
   OPENAI_API_KEY=your_openai_api_key
   ```

## Usage

1. **Run the Application**

   ```bash
   streamlit run app.py
   ```
2. **Upload PDF Documents**

   - Use the sidebar in the Streamlit app to upload your PDF files and click on 'Process'.
3. **Interact with the Chatbot**

   - Enter your questions in the text input field to get answers based on the content of the uploaded PDFs.

## Dependencies

The project requires the following Python packages:

- `streamlit`
- `openai`
- `PyPDF2`
- `langchain`
- `python-dotenv`
- `faiss-cpu` (or `faiss-gpu` if using GPU)

These dependencies are listed in `requirements.txt`.

## Contributing

Feel free to fork the repository and submit pull requests with improvements or fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
