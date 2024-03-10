import logging

from dataclasses import dataclass
from langchain_mongodb.vectorstores import MongoDBDocumentType
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import DirectoryLoader
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_core.documents import Document
from pymongo import MongoClient
from pymongo.collection import Collection
from typing import List, Tuple
from langchain_openai import OpenAIEmbeddings, OpenAI
from llm import LLM


@dataclass
class RAGConfig:
    """Data class for configuration settings."""
    index_name: str
    db_name: str
    collection_name: str
    openai_api_key: str
    mongo_uri: str


def connect_to_collection(config: RAGConfig) -> Collection[MongoDBDocumentType]:
    client = MongoClient(config.mongo_uri)
    logging.info('Connected to MongoDB')
    return client[config.db_name][config.collection_name]


class OpenAILLM(LLM):

    def __init__(self, config: RAGConfig, save_new=False) -> None:
        self.config = config
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.config.openai_api_key)
        self.collection = connect_to_collection(config)
        self.vec_store = None
        self.save_new = save_new
        if save_new:
            data = DirectoryLoader('data', glob='./*.txt', show_progress=True).load()
            MongoDBAtlasVectorSearch.from_documents(data, self.embeddings, collection=self.collection)
            logging.info('saved new embeddings to MongoDB')
        else:
            self.vec_store = MongoDBAtlasVectorSearch(
                self.collection,
                embedding=self.embeddings,
                index_name=self.config.index_name,
            )

    def tokenize(self, text: str) -> List[float]:
        self._validate()
        return self.embeddings.embed_query(text)

    def answer(self, query: str) -> str:
        self._validate()
        _ = self._search(query)  # might be useful for a hallucination detection
        llm = OpenAI(openai_api_key=self.config.openai_api_key, temperature=0.5)
        qa = RetrievalQA.from_chain_type(llm, chain_type='stuff', retriever=self.vec_store.as_retriever())
        response = qa.invoke({"query": query}, return_only_outputs=True)
        return response['result'].strip()

    def _search(self, query: str) -> Tuple[Document, float]:
        docs = self.vec_store.similarity_search_with_score(query=query, k=1)
        document, result = docs[0]
        logging.info(f'Answer: {document}')
        logging.info(f'Result: {result}')
        return document, result

    def _validate(self) -> None:
        if self.save_new:
            raise Exception("OpenAILLM has been initialized withing save mode")
