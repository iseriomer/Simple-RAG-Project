import tkinter as tk
from tkinter import scrolledtext
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from operator import itemgetter
import os

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize model
model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
parser = StrOutputParser()
template = """
Answer the question based on the context below. If you can't 
answer the question, reply "I don't know".

Context: {context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model | parser

# Load and process documents
loader = TextLoader("transcription.txt")
text_documents = loader.load()

def process_query(query, chunk_size, chunk_overlap, num_chunks):
    # Adjust chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(text_documents)
    
    embeddings = OpenAIEmbeddings()
    vectorstore1 = DocArrayInMemorySearch.from_texts(
        texts=[chunk.page_content for chunk in chunks],
        embedding=embeddings,
    )
    retriever1 = vectorstore1.as_retriever()

    embedded_query = embeddings.embed_query(query)
    retrieved_chunks = retriever1.invoke(query)
    
    # Combine the retrieved chunks into a single context
    combined_context = "\n".join(chunk.page_content for chunk in retrieved_chunks[:num_chunks])
    full_query = f"""
    Answer the question based on the context below. If you can't 
    answer the question, reply "I don't know".

    Context: {combined_context}

    Question: {query}
    """
    
    response = model.invoke(full_query)
    
    # Extract and return the content from the response
    if hasattr(response, 'content'):
        return response.content
    return str(response)

# Create the UI
class ChatbotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Chatbot Interface")

        # Frame for chat area
        self.chat_frame = tk.Frame(root, bg="#f0f0f0")
        self.chat_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.chat_area = scrolledtext.ScrolledText(self.chat_frame, state='disabled', wrap='word', height=20, width=80, bg="#ffffff", font=("Arial", 12))
        self.chat_area.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Frame for input and settings
        self.input_frame = tk.Frame(root, bg="#e0e0e0")
        self.input_frame.pack(padx=10, pady=10, fill=tk.X)

        self.input_area = tk.Entry(self.input_frame, width=70, font=("Arial", 12))
        self.input_area.pack(side=tk.LEFT, padx=(0, 10))

        self.send_button = tk.Button(self.input_frame, text="Send", command=self.send_message, bg="#4CAF50", fg="white", font=("Arial", 12, "bold"))
        self.send_button.pack(side=tk.LEFT)

        # Input fields for chunk settings
        self.settings_frame = tk.Frame(root, bg="#e0e0e0")
        self.settings_frame.pack(padx=10, pady=10, fill=tk.X)

        self.chunk_size_label = tk.Label(self.settings_frame, text="Chunk Size:", bg="#e0e0e0", font=("Arial", 12))
        self.chunk_size_label.grid(row=0, column=0, sticky='e')
        self.chunk_size_input = tk.Entry(self.settings_frame, width=10, font=("Arial", 12))
        self.chunk_size_input.grid(row=0, column=1, sticky='w')
        self.chunk_size_input.insert(0, "1000")  # Default value

        self.chunk_overlap_label = tk.Label(self.settings_frame, text="Chunk Overlap:", bg="#e0e0e0", font=("Arial", 12))
        self.chunk_overlap_label.grid(row=1, column=0, sticky='e')
        self.chunk_overlap_input = tk.Entry(self.settings_frame, width=10, font=("Arial", 12))
        self.chunk_overlap_input.grid(row=1, column=1, sticky='w')
        self.chunk_overlap_input.insert(0, "100")  # Default value

        self.num_chunks_label = tk.Label(self.settings_frame, text="Number of Chunks:", bg="#e0e0e0", font=("Arial", 12))
        self.num_chunks_label.grid(row=2, column=0, sticky='e')
        self.num_chunks_input = tk.Entry(self.settings_frame, width=10, font=("Arial", 12))
        self.num_chunks_input.grid(row=2, column=1, sticky='w')
        self.num_chunks_input.insert(0, "5")  # Default value

        # Add tags for user and bot messages
        self.chat_area.tag_config('user', foreground='black')
        self.chat_area.tag_config('bot', foreground='blue')

        # Bind Enter key to send_message method
        self.input_area.bind("<Return>", self.send_message_event)

    def send_message(self):
        user_message = self.input_area.get()
        if user_message.lower() in ["0", "exit"]:
            self.root.quit()
            return

        if user_message:
            # Get chunk settings
            chunk_size = int(self.chunk_size_input.get())
            chunk_overlap = int(self.chunk_overlap_input.get())
            num_chunks = int(self.num_chunks_input.get())

            self.chat_area.config(state='normal')
            self.chat_area.insert(tk.END, "You: " + user_message + '\n', 'user')
            response = process_query(user_message, chunk_size, chunk_overlap, num_chunks)
            self.chat_area.insert(tk.END, "Bot: " + response + '\n', 'bot')
            self.chat_area.config(state='disabled')
            self.chat_area.yview(tk.END)
            self.input_area.delete(0, tk.END)

    def send_message_event(self, event):
        self.send_message()

# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = ChatbotApp(root)
    root.mainloop()
