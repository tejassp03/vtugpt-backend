from fastapi import FastAPI, HTTPException
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Anyscale
import boto3
from botocore.exceptions import ClientError
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import aiohttp
import os

os.environ["ANYSCALE_API_BASE"] = "https://api.endpoints.anyscale.com/v1"
os.environ["ANYSCALE_API_KEY"] = 'esecret_u2sbcvxgegri9x31reyagyurgb'


app = FastAPI()
origins = [
    "http://localhost:3000",
    "http://localhost:8080",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def read_root():
    return {"message": "Welcome to the Langchain API"}

def load_vectorstore():
        embeddings=CohereEmbeddings(cohere_api_key="PtVIxuqTBJYMP1GAhjsXbFso9hX08lGVZjIcwBVT")
        db = FAISS.load_local('vectordb', embeddings, allow_dangerous_deserialization=True)
        return db

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

@app.post("/query")
async def query(question_request: QuestionRequest):
    template =  """Answer the question based on the context below. If the
question cannot be answered using the information provided answer
with "I don't know" Do not hallucinate.

Context:
{context}

Question:
{question}

Answer:
"""
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = Anyscale(model_name='meta-llama/Meta-Llama-3-8B-Instruct')

    try:
        db = load_vectorstore()
        qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=db.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        question = question_request.question
        
        result = qa_chain({"query": question})
        query_with_vtu = f"{question} VTU"

        # Asynchronous call to YouTube Data API
        async with aiohttp.ClientSession() as session:
            async with session.get("https://www.googleapis.com/youtube/v3/search", params={
                "key": "AIzaSyAfMQm5XhQa21QIGz52e9YlBPjEpgGG8pg",
                "part": "snippet",
                "q": query_with_vtu,
                "maxResults": 3,
                "type": "video"
            }) as response:
                youtube_data = await response.json()
        videos = []
        for item in youtube_data.get("items", []):
            video_id = item["id"]["videoId"]
            title = item["snippet"]["title"]
            thumbnail_url = item["snippet"]["thumbnails"]["default"]["url"]
            videos.append({"video_id": video_id, "title": title, "thumbnail_url": thumbnail_url})

        # Add extracted video information to the result
        result['youtube_videos'] = videos
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

def ingest_documents_from_local(local_directory):
    vector_dir = "vectordb/"
    try:
        # Load documents from local directory
        documents = []
        for file in os.listdir(local_directory):
            if file.endswith('.pdf'):
                pdf_path = os.path.join(local_directory, file)
                loader = PyPDFLoader(pdf_path, extract_images=False)
                documents.extend(loader.load())

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=10)
        chunked_documents = text_splitter.split_documents(documents)

        embeddings = CohereEmbeddings(cohere_api_key="PtVIxuqTBJYMP1GAhjsXbFso9hX08lGVZjIcwBVT")

        # Create vector store and save it
        vector_store = FAISS.from_documents(chunked_documents, embeddings)
        vector_store.save_local(vector_dir)

        return {"message": f"Vector DB stored in {vector_dir}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

def download_files_from_s3(bucket_name, local_directory, aws_access_key_id, aws_secret_access_key):
    s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
    try:
        os.makedirs(local_directory, exist_ok=True)
        response = s3.list_objects_v2(Bucket=bucket_name)
        if 'Contents' in response:
            for obj in response['Contents']:
                filename = obj['Key']
                local_file_path = os.path.join(local_directory, os.path.basename(filename))
                if not os.path.exists(local_file_path):
                    s3.download_file(bucket_name, filename, local_file_path)
    except ClientError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ingest")
async def ingest_documents():
    bucket_name = "vtugpt-docs"  # Replace with your S3 bucket name
    local_directory = "documents"
    aws_access_key_id = "AKIA47CRXS6UGRY3IMOQ"  # Replace with your AWS access key ID
    aws_secret_access_key = "imwie3mCuWgbcJQX0TYFDEjA9KRhEj4YDe9rTq8G"  # Replace with your AWS secret access key

    try:
        # Download files from S3 to local directory
        download_files_from_s3(bucket_name, local_directory, aws_access_key_id, aws_secret_access_key)

        # Ingest documents from local directory
        return ingest_documents_from_local(local_directory)

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

