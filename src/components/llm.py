import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

embedding_model = OpenAIEmbeddings(
    model='text-embedding-3-small',
    dimensions=1536
)


if __name__ == "__main__":
    print('llm.py')