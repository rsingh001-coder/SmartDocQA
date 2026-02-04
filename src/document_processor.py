import os
import cohere
import fitz
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv


load_dotenv()


class VectorStore:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path

        cohere_api_key = os.getenv("COHERE_API_KEY")
        pinecone_api_key = os.getenv("PINECONE_API_KEY")

        if not cohere_api_key or not pinecone_api_key:
            raise ValueError("COHERE_API_KEY or PINECONE_API_KEY not found in .env file")

        self.co = cohere.Client(cohere_api_key)
        self.pinecone_api_key = pinecone_api_key

        self.chunks = []
        self.embeddings = []

        self.retrieve_top_k = 10
        self.rerank_top_k = 3
        self.index_name = "rag-qa-bot"

        self.load_pdf()
        self.split_text()
        self.embed_chunks()
        self.index_chunks()

    def load_pdf(self):
        self.pdf_text = self.extract_text_from_pdf(self.pdf_path)

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        text = ""
        with fitz.open(pdf_path) as pdf:
            for page in pdf:
                text += page.get_text("text")
        return text

    def split_text(self, chunk_size=1000):
        sentences = self.pdf_text.split(". ")
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += sentence + ". "
            else:
                self.chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "

        if current_chunk:
            self.chunks.append(current_chunk.strip())

    def embed_chunks(self, batch_size=90):
        for i in range(0, len(self.chunks), batch_size):
            batch = self.chunks[i:i + batch_size]
            embeddings = self.co.embed(
                texts=batch,
                model="embed-english-v3.0",
                input_type="search_document"
            ).embeddings

            self.embeddings.extend(embeddings)

    def index_chunks(self):
        pc = Pinecone(api_key=self.pinecone_api_key)

        dimension = len(self.embeddings[0])

        if self.index_name not in pc.list_indexes().names():
            pc.create_index(
                name=self.index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )

        self.index = pc.Index(self.index_name)

        vectors = [
            (str(i), self.embeddings[i], {"text": self.chunks[i]})
            for i in range(len(self.chunks))
        ]

        self.index.upsert(vectors=vectors)

    def retrieve(self, query: str) -> list:
        query_embedding = self.co.embed(
            texts=[query],
            model="embed-english-v3.0",
            input_type="search_query"
        ).embeddings[0]

        results = self.index.query(
            vector=query_embedding,
            top_k=self.retrieve_top_k,
            include_metadata=True
        )

        docs = [match["metadata"]["text"] for match in results["matches"]]

        reranked = self.co.rerank(
            query=query,
            documents=docs,
            top_n=self.rerank_top_k,
            model="rerank-v3.5"
        )

        return [docs[item.index] for item in reranked.results]
