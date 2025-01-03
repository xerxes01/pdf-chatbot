# PDF Question-Answering System

This system allows you to upload PDF documents and ask questions about their content. It uses OpenAI's language models for generating accurate answers based on the document context.

## Features

- PDF text extraction and processing
- Semantic search using document embeddings
- Question answering using OpenAI GPT models
- FastAPI-based REST API
- Production-grade code structure

## Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Modify .env to add your OpenAI API key
4. Run the server: `python main.py`

## Usage

Send a POST request to `/upload-and-query` with:
- PDF file in form data
- JSON body containing list of questions

Example curl request:
```bash
curl -X POST "http://localhost:8000/api/upload-and-query" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@handbook.pdf" \
     -F 'questions_json={"questions":["What is the company name?","Who is the CEO?"]}'

```

## Project Structure

```
pdf-qa-system/
├── src/
│   ├── core/
│   │   ├── document_processor.py
│   │   ├── vector_store.py
│   │   ├── question_answerer.py
│   │   └── pdf_qa_system.py
│   └── api/
│       └── routes.py
├── main.py
├── requirements.txt
└── README.md
```