from dataclasses import dataclass
from typing import Optional
import openai


@dataclass
class Question:
    text: str
    context: Optional[str] = None


@dataclass
class Answer:
    question: str
    answer: str
    confidence: float


class QuestionAnswerer:
    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.model = model

    def answer_question(self, question: Question) -> Answer:
        """Generate answer using OpenAI API."""
        # debug
        print("Context: ", question.context)

        system_prompt = """You are a helpful assistant that answers questions based on the provided context.
        If you cannot find a clear answer in the context, respond with "Data Not Available".
        Provide direct, concise answers without additional explanation."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context: {question.context}\n\nQuestion: {question.text}"}
        ]

        response = openai.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            max_tokens=150
        )

        answer_text = response.choices[0].message.content.strip()
        confidence = 1.0 if answer_text != "Data Not Available" else 0.0

        return Answer(
            question=question.text,
            answer=answer_text,
            confidence=confidence
        )