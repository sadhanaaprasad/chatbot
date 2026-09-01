import os
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from docx import Document
import pypdf


# -----------------------------
# Streamlit Page Configuration
# -----------------------------

st.set_page_config(
    page_title="AI Assistant with RAG",
    page_icon="🩵",
    layout="wide"
)


# -----------------------------
# Custom CSS
# -----------------------------

st.markdown("""
<style>
.stApp {
    background-color: #f0f8ff;
}

h1 {
    color: #1e3d59;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------
# API Initialization
# -----------------------------

try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    GROQ_API_KEY = None


if GROQ_API_KEY:
    client = Groq(api_key=GROQ_API_KEY)
else:
    client = None


# -----------------------------
# Load Embedding Model
# -----------------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


embedding_model = load_embedding_model()


# -----------------------------
# Initialize Session State
# -----------------------------

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "document_chunks" not in st.session_state:
    st.session_state.document_chunks = []

if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------
# Extract Text from File
# -----------------------------

def extract_text_from_file(uploaded_file):

    file_name = uploaded_file.name
    extension = os.path.splitext(file_name)[1].lower()

    text = ""

    if extension in [".txt", ".csv"]:

        text = uploaded_file.getvalue().decode(
            "utf-8",
            errors="ignore"
        )

    elif extension == ".pdf":

        reader = pypdf.PdfReader(uploaded_file)

        for page in reader.pages:
            text += (page.extract_text() or "") + "\n"

    elif extension == ".docx":

        doc = Document(uploaded_file)

        text = "\n".join(
            [paragraph.text for paragraph in doc.paragraphs]
        )

    return text


# -----------------------------
# Split Text into Chunks
# -----------------------------

def chunk_text(text, chunk_size=500, overlap=50):

    words = text.split()

    chunks = []

    for i in range(
        0,
        len(words),
        chunk_size - overlap
    ):

        chunk = " ".join(
            words[i:i + chunk_size]
        )

        if chunk:
            chunks.append(chunk)

    return chunks


# -----------------------------
# Process Uploaded File
# -----------------------------

def process_file(uploaded_file):

    raw_text = extract_text_from_file(uploaded_file)

    document_chunks = chunk_text(raw_text)

    if not document_chunks:
        return False, "Could not extract text from the file."

    embeddings = embedding_model.encode(
        document_chunks
    )

    embeddings = np.array(
        embeddings
    ).astype("float32")

    dimension = embeddings.shape[1]

    vector_store = faiss.IndexFlatL2(
        dimension
    )

    vector_store.add(embeddings)

    st.session_state.vector_store = vector_store
    st.session_state.document_chunks = document_chunks

    return True, (
        f"Processed {len(document_chunks)} "
        "context chunks successfully!"
    )


# -----------------------------
# Retrieve Relevant Context
# -----------------------------

def retrieve_context(query, k=3):

    vector_store = st.session_state.vector_store
    document_chunks = st.session_state.document_chunks

    if (
        vector_store is None
        or len(document_chunks) == 0
    ):
        return ""

    query_vector = embedding_model.encode(
        [query]
    )

    query_vector = np.array(
        query_vector
    ).astype("float32")

    distances, indices = vector_store.search(
        query_vector,
        k
    )

    retrieved_chunks = []

    for i in indices[0]:

        if i < len(document_chunks):
            retrieved_chunks.append(
                document_chunks[i]
            )

    return "\n---\n".join(
        retrieved_chunks
    )


# -----------------------------
# Page Title
# -----------------------------

st.title("🩵 AI Assistant with RAG File Upload")

st.write(
    "Upload a document and ask questions about it!"
)


# -----------------------------
# Sidebar File Upload
# -----------------------------

with st.sidebar:

    st.header("📁 Upload File")

    uploaded_file = st.file_uploader(

        "Upload Context File",

        type=[
            "pdf",
            "txt",
            "docx",
            "csv"
        ]

    )

    if uploaded_file is not None:

        if st.button("Process File"):

            with st.spinner(
                "Processing your file..."
            ):

                success, message = process_file(
                    uploaded_file
                )

            if success:
                st.success(message)

            else:
                st.error(message)


# -----------------------------
# Display Chat History
# -----------------------------

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# -----------------------------
# User Input
# -----------------------------

if prompt := st.chat_input(
    "Ask anything about your document..."
):

    if client is None:

        st.error(
            "Groq API key is missing. "
            "Please add it to Streamlit Secrets."
        )

    else:

        # Display user message

        st.session_state.messages.append({

            "role": "user",

            "content": prompt

        })


        with st.chat_message("user"):

            st.markdown(prompt)


        # Retrieve document context

        context = retrieve_context(
            prompt
        )


        messages = [

            {

                "role": "system",

                "content": (
                    "You are a helpful AI assistant. "
                    "Answer using the provided context "
                    "when available."
                )

            }

        ]


        # Add previous messages

        for message in st.session_state.messages[:-1]:

            messages.append({

                "role": message["role"],

                "content": message["content"]

            })


        # Add RAG context

        if context:

            final_prompt = f"""
Relevant Context from Uploaded File:

{context}

User Question:

{prompt}
"""

        else:

            final_prompt = prompt


        messages.append({

            "role": "user",

            "content": final_prompt

        })


        # Generate response

        with st.chat_message("assistant"):

            response_placeholder = st.empty()

            full_response = ""


            try:

                response_stream = (
                    client.chat.completions.create(

                        model="llama-3.1-8b-instant",

                        messages=messages,

                        temperature=0.7,

                        stream=True

                    )
                )


                for chunk in response_stream:

                    content = (
                        chunk.choices[0]
                        .delta.content
                        or ""
                    )

                    full_response += content

                    response_placeholder.markdown(
                        full_response + "▌"
                    )


                response_placeholder.markdown(
                    full_response
                )


                st.session_state.messages.append({

                    "role": "assistant",

                    "content": full_response

                })


            except Exception as e:

                st.error(
                    f"Error: {str(e)}"
                )