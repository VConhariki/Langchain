from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

CAMINHO_RAG_PADRAO = "rag.md"

def obter_rag(caminho_arquivo: str | None = None) -> BM25Retriever:
    documentos = TextLoader(caminho_arquivo or CAMINHO_RAG_PADRAO, encoding="utf-8").load()
    divisor = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
    )

    documentos_divididos = divisor.split_documents(documentos)
    retriever = BM25Retriever.from_documents(documentos_divididos, k=4)
    return retriever