import os
import fitz  # PyMuPDF
from docx import Document
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
import gradio as gr

# Load environment variables
load_dotenv()

# Retrieve the OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Define a template for the chat prompt
template = """
Answer the question based on the context below. If you can't 
answer the question, reply "I don't know".

Context: {context}

Question: {question}
"""
parser = StrOutputParser()
prompt = ChatPromptTemplate.from_template(template)

# Cache for storing processed documents and their indexes
document_cache = {}

def load_pdf(filepath):
    """Extract text from a PDF file."""
    text = ""
    doc = fitz.open(filepath)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

def load_docx(filepath):
    """Extract text from a DOCX file."""
    doc = Document(filepath)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def load_text(filepath):
    """Extract text from a text file."""
    with open(filepath, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def process_query(filepaths, model_name, chunk_size, chunk_overlap, num_of_chunks, temperature, max_token_to_use, embedding_model, question):
    # Initialize the ChatOpenAI model
    model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model=model_name, temperature=temperature, max_tokens=max_token_to_use)

    all_context_docs = []

    # Create the faiss_indexes folder if it doesn't exist
    os.makedirs('faiss_indexes', exist_ok=True)

    # Process each file
    for filepath in filepaths:
        if filepath in document_cache:
            print(f"Loading FAISS index from cache for {filepath}")
            db = document_cache[filepath]
        else:
            faiss_index_file = os.path.join('faiss_indexes', f"{os.path.splitext(os.path.basename(filepath))[0]}_faiss_index")
            
            # Check for existing FAISS index
            if not os.path.exists(faiss_index_file):
                print(f"Creating FAISS index for {filepath}")
                if filepath.endswith(".pdf"):
                    text = load_pdf(filepath)
                elif filepath.endswith(".docx"):
                    text = load_docx(filepath)
                elif filepath.endswith(".txt"):
                    text = load_text(filepath)
                else:
                    loader = TextLoader(filepath, encoding="utf-8")
                    text_documents = loader.load()
                    text = "".join([doc.page_content for doc in text_documents])

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                chunks = text_splitter.split_text(text)
                embeddings = OpenAIEmbeddings(model=embedding_model)
                db = FAISS.from_texts(chunks, embeddings)
                db.save_local(faiss_index_file)
            else:
                print(f"Loading existing FAISS index for {filepath}")
                embeddings = OpenAIEmbeddings(model=embedding_model)
                db = FAISS.load_local(faiss_index_file, embeddings, allow_dangerous_deserialization=True)
            
            # Cache the processed document
            document_cache[filepath] = db
        
        # Retrieve relevant chunks
        context_docs = db.similarity_search(query=question, k=num_of_chunks)
        all_context_docs.extend(context_docs)
    
    # Combine the relevant chunks into a single context string
    relevant_chunks = [doc.page_content for doc in all_context_docs]
    context = "\n\n".join(relevant_chunks)
    
    # Prepare the prompt with context and question
    formatted_prompt = prompt.format_prompt(context=context, question=question).to_string()
    
    # Get the answer from the model
    answer = model(formatted_prompt)
    
    # Extract the content part of the answer
    answer_content = answer.content
    answer_tokens = answer.usage_metadata

    return context, answer_content, answer_tokens

# Create Gradio interface
def gradio_interface(filepaths, model_name, chunk_size, chunk_overlap, num_of_chunks, temperature, max_token_to_use, embedding_model, question):
    # Process the query and get the relevant chunks and answer
    context, answer_content, answer_tokens = process_query(filepaths, model_name, chunk_size, chunk_overlap, num_of_chunks, temperature, max_token_to_use, embedding_model, question)
    # Return only the relevant chunks and the answer
    return context, answer_content, answer_tokens

model_options = ["gpt-4o-mini", "gpt-4o-large", "gpt-3.5-turbo", "gpt-4-32k"]
embedding_model_options = ["text-embedding-ada-002", "text-embedding-babbage-001", "text-embedding-curie-001"]

demo = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Files(label="Upload Documents", type="filepath", file_count="multiple"),
        gr.Dropdown(choices=model_options, label="Model Name", value="gpt-4o-mini"),
        gr.Slider(label="Chunk Size", minimum=0, maximum=2000, step=100, value=1000, info="Size of each text chunk"),
        gr.Slider(label="Chunk Overlap", minimum=0, maximum=500, step=50, value=100, info="Overlap between chunks"),
        gr.Slider(label="Number of Chunks", minimum=1, maximum=20, step=1, value=10, info="Number of chunks to retrieve"),
        gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, step=0.1, value=0.7, info="Model response creativity"),
        gr.Slider(label="Max Tokens to Use", minimum=10, maximum=4000, step=10, value=1000, info="Maximum tokens to use in the response"),
        gr.Dropdown(choices=embedding_model_options, label="Embedding Model", value="text-embedding-ada-002"),
        gr.Textbox(label="Question", placeholder="Enter your question here...")
    ],
    outputs=[
        gr.Textbox(label="Relevant Context", placeholder="This will show the relevant chunks used in the query."),
        gr.Textbox(label="Answer", placeholder="This will show the model's response based on the context."),
        gr.Textbox(label="Answer Token Cost", placeholder="This will show the model's cost of tokens.")
    ],
    title="Simple RAG Project",
    description="Customize the parameters and ask questions based on the provided context."
)
demo.launch(share = True)