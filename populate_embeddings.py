import logging
import os

from dotenv import load_dotenv

from openaillm import OpenAILLM, RAGConfig


def main() -> None:
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    load_dotenv()
    rag_config = RAGConfig(
        index_name='openai_idx',
        db_name='langchain_demo',
        collection_name='collection_of_blobs',
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        mongo_uri=os.getenv('MONGO_DB_URI'),
    )
    OpenAILLM(config=rag_config, save_new=True)


if __name__ == '__main__':
    main()
