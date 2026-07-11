import os
from enum import Enum
from typing import cast

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langsmith import Client, traceable
from pydantic import BaseModel, Field

load_dotenv()


class ConfidenceLevel(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"


class QAResponse(BaseModel):
    answer: str = Field(description="The answer to the user question")
    confidence: ConfidenceLevel = Field(
        description="Confidence level: high, medium or low"
    )
    reasoning: str = Field(description="The reasoning behind the answer provided")
    follow_up_questions: list[str] = Field(
        description="A list of follow up questions related to the topic",
        default_factory=list,
    )
    sources_needed: bool = Field(
        description="Indicates whether sources are needed for the answer",
        default=False,
    )


class SmartBotQA:
    def __init__(
        self, *, model_name: str = "gpt-4o-mini", temperature: float = 0.3
    ) -> None:
        self.model = init_chat_model(
            model_name, temperature=temperature
        ).with_structured_output(QAResponse)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are SmartBot, an intelligent question-answering assistant.

Your job is to answer user questions as accurately and clearly as possible.

Always return your response using the provided structured schema.

Guidelines:

- Answer the user's question directly.
- If you are uncertain, state that uncertainty rather than inventing facts.
- Keep answers concise but complete.
- Explain the reasoning behind your answer in the "reasoning" field.
- Estimate your confidence as one of:
  - high
  - medium
  - low
- Suggest 2-5 relevant follow-up questions whenever they would naturally help the user explore the topic further.
- Set "sources_needed" to:
    - true if the answer depends on recent events, statistics, laws, medical information, financial advice, scientific research, or any information that should be verified.
    - false if the answer is based on stable, general knowledge.
- Never fabricate sources or claim to have verified information when you have not.

Your goal is to be accurate, honest, and helpful.
""",
                ),
                ("human", "{question}"),
            ]
        )
        self.chain = self.prompt | self.model

    @traceable(name="ask_question", run_type="chain", client=Client())
    def ask(self, question: str) -> QAResponse:
        try:
            res = self.chain.invoke({"question": question})
            return cast(QAResponse, res)
        except Exception as e:
            print(f"Error during BOT response: {e}")
            return QAResponse(
                answer=(
                    "I'm sorry, but I couldn't process your question due to an "
                    "internal error. Please try again in a moment."
                ),
                confidence=ConfidenceLevel.low,
                reasoning=(
                    "The request could not be completed because an unexpected "
                    "error occurred while generating the response."
                ),
                follow_up_questions=[
                    "Could you try asking the question again?",
                    "Would you like to rephrase your question?",
                ],
                sources_needed=False,
            )

    @traceable(
        name="ask_batch",
        run_type="chain",
    )
    def ask_batch(self, questions: list[str]) -> list[QAResponse]:
        input = [{"question": q} for q in questions]
        responses = self.chain.batch(input)
        return cast(list[QAResponse], responses)


def main():
    bot = SmartBotQA()
    questions = [
        "What is Python?",
        "Explain dependency injection in simple terms.",
        "How does HTTPS protect data in transit?",
        "What are the latest AI trends in 2026?",
        "Should I invest all my savings in a single stock?",
    ]
    print("=" * 50)
    print("SMART QA Bot DEMO")
    print("=" * 50)
    for question in questions:
        print(f"\nQuestion: {question}")
        print("-" * 40)
        response = bot.ask(question)
        print(f"Answer: {response.answer}")
        print(f"Confidence: {response.confidence.value}")
        print(f"Reasoning: {response.reasoning}")
        print(f"Sources Needed: {'Yes' if response.sources_needed else 'No'}")
        if response.follow_up_questions:
            print("Follow-up Questions:")
            for i, question in enumerate(response.follow_up_questions, start=1):
                print(f"  {i}. {question}")
    print("=" * 50)


@traceable(name="error_handling_demo", run_type="chain")
def demo_error_handling():
    bot = SmartBotQA()
    print("\n" + ("=") * 60)
    print("Error handling demo")
    print("\n" + ("=") * 60)
    long_question = "What is " + "very" * 100 + "important"
    response = bot.ask(long_question)
    print(response)


if __name__ == "__main__":
    if (
        not os.getenv("LANGSMITH_API_KEY")
        and not os.getenv("LANGSMITH_TRACING") == "true"
    ):
        exit()
    os.environ.setdefault("LANGSMITH_PROJECT_NAME", "Smart Q&A Bot Project")
    print(f"LANGSMITH is configured - {os.getenv('LANGSMITH_PROJECT_NAME')}")
    try:
        # main()
        demo_error_handling()
    finally:
        Client().flush()
