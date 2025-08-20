import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import HuggingFaceEmbeddings

os.environ["GROQ_API_KEY"] = "---"
os.environ["PINECONE_API_KEY"] = "---"

loader = PyPDFLoader("sample.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

texts = [doc.page_content for doc in chunks]

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

index_name = "pdf-learner"

vectorstore = PineconeVectorStore.from_texts(
    texts,
    embedding,
    index_name=index_name,
    pinecone_api_key=os.environ["PINECONE_API_KEY"]
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatGroq(model="llama3-70b-8192", api_key=os.environ["GROQ_API_KEY"])

qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

while True:
    query = input("\nAsk a question (or 'exit'): ")
    if query.lower() == "exit":
        break
    answer = qa_chain.run(query)
    print(f"\n📘 Answer: {answer}")
