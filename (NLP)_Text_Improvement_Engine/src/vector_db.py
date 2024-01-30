from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
# from langchain.embeddings.base import Embeddings
from langchain.embeddings import HuggingFaceEmbeddings


class VectorDataBase:
    # change for "intfloat/multilingual-e5-large" if you have enough RAM
    EMBEDDING_HF_NAME = "intfloat/e5-base"

    def __init__(self, data_path: str = "standard_phrases.txt"):
        self.data_path = data_path
        self.embeddings = HuggingFaceEmbeddings(model_name=self.EMBEDDING_HF_NAME)
        self.db: FAISS = self._vectorize_docs()

    def _vectorize_docs(self) -> FAISS:
        """
        Create vector database from files.
        :return: FAISS DB
        """
        loader = TextLoader(self.data_path)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(separator="\n", chunk_overlap=0, length_function=len, chunk_size=0)
        texts = text_splitter.split_documents(documents)
        db = FAISS.from_documents(texts, self.embeddings)
        db.as_retriever()
        return db
