import cohere
import uuid
import os
from dotenv import load_dotenv

load_dotenv()

class Chatbot:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.reset_conversation()

        cohere_api_key = os.getenv("COHERE_API_KEY")
        if not cohere_api_key:
            raise ValueError("COHERE_API_KEY not found in .env file")

        self.co = cohere.Client(cohere_api_key)

    def reset_conversation(self):
        self.conversation_id = str(uuid.uuid4())

    def respond(self, user_message: str):
        response = self.co.chat(
            message=user_message,
            model="command-r",
            search_queries_only=True
        )

        retrieved_docs = []
        seen = set()  # 👈 prevent duplicates

        if response.search_queries:
            for query in response.search_queries:
                chunks = self.vectorstore.retrieve(query.text)

                for chunk in chunks:
                    if chunk not in seen:
                        retrieved_docs.append({"text": chunk})
                        seen.add(chunk)

            response = self.co.chat_stream(
                message=user_message,
                model="command-r",
                documents=retrieved_docs,
                conversation_id=self.conversation_id,
            )
        else:
            response = self.co.chat_stream(
                message=user_message,
                model="command-r",
                conversation_id=self.conversation_id,
            )

        return response, retrieved_docs
