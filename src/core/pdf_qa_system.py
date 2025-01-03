import os
from typing import List, Dict
import json
from dotenv import load_dotenv
import openai
from .document_processor import DocumentProcessor
from .vector_store import VectorStore
from .question_answerer import QuestionAnswerer, Question


class PDFQuestionAnsweringSystem:
    def __init__(self):
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")

        self.doc_processor = DocumentProcessor()
        self.vector_store = VectorStore()
        self.qa = QuestionAnswerer()

    def load_document(self, pdf_path: str):
        """Process PDF and prepare for questioning."""
        chunks = self.doc_processor.process_pdf(pdf_path)
        self.vector_store.add_documents(chunks)

    def answer_questions(self, questions: List[str]) -> str:
        """Answer a list of questions and return JSON response."""
        results = []

        for q_text in questions:
            context = self.vector_store.find_relevant_context(q_text)
            question = Question(text=q_text, context=context)
            answer = self.qa.answer_question(question)
            results.append({
                "question": answer.question,
                "answer": answer.answer
            })

        return json.dumps(results, indent=2)