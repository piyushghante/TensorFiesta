# import streamlit as st
# import io
# import os
# from langchain.prompts import PromptTemplate  # Changed import statement
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.llms import CTransformers
# from langchain.chains import RetrievalQA
# import tempfile
# from PyPDF2 import PdfReader

# DB_FAISS_PATH = "db_faiss/"
# cache = {}  # Create a cache dictionary

# def set_custom_prompt():
#     """
#     Prompt template for QA retrieval for each vector store
#     """
#     custom_prompt_template = """ Use the following pieces of information to answer the user's question. IF you don't know the answer, please just say that you don't know the answer, don't try to make up an answer.

#  Context : {context}
#  Question: {question}

#  Only returns the helpful answer below and nothing else.
#  Helpful answer:

#  """
#     prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
#     return prompt

# def load_llm():
#     llm = CTransformers(
#         model="llama-2-7b-chat.ggmlv3.q8_0.bin",
#         model_type="llama",
#         max_new_tokens=1024,
#         temperature=0.7
#     )
#     return llm

# def retrieval_qa_chain(llm, prompt, db):
#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=db.as_retriever(search_kwargs={'k': 2}),
#         return_source_documents=True,
#         chain_type_kwargs={'prompt': prompt}
#     )
#     return qa_chain

# def qa_bot():
#     embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
#     db = FAISS.load_local(DB_FAISS_PATH, embeddings)
#     llm = load_llm()
#     qa_prompt = set_custom_prompt()
#     qa_chain = retrieval_qa_chain(llm, qa_prompt, db)
#     return qa_chain

# def cache_result(key, func, *args, **kwargs):
#     """
#     Caches the result of a function to avoid recomputation for the same inputs.
#     """
#     if key in cache:
#         return cache[key]
#     result = func(*args, **kwargs)
#     cache[key] = result
#     return result

# async def get_answer(chain, user_input):
#     res = await chain.acall(user_input)
#     answer = res["result"]
#     sources = res["source_documents"]
#     if sources:
#         answer += f"\nSources:" + str(sources)
#     else:
#         answer += f"\nNo Sources Found"
#     return answer

# async def main():
#     st.title("Mines Bot")
#     cache_result("qa_chain", qa_bot)  # Initialize the QA chain
#     chain = cache["qa_chain"]  # Retrieve the initialized QA chain

#     # File upload UI
#     st.sidebar.title("Upload PDF")
#     uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")
#     if uploaded_file is not None:
#         # Process the PDF file
#         with st.spinner('Processing PDF...'):
#             texts = extract_text_from_pdf(uploaded_file)
#             for text in texts:
#                 answer = await get_answer(chain, text)
#                 st.write(answer)

# def extract_text_from_pdf(uploaded_file):
#     """
#     Extract text content from a PDF file.
#     """
#     texts = []
#     with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
#         tmp_file.write(uploaded_file.read())
#         tmp_file.close()
#         with open(tmp_file.name, "rb") as file:
#             reader = PdfReader(file)
#             num_pages = len(reader.pages)
#             for page_num in range(num_pages):
#                 page = reader.pages[page_num]
#                 text = page.extract_text()
#                 # Split text into chunks of maximum context length
#                 chunk_size = 1024
#                 chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
#                 texts.extend(chunks)
#     os.unlink(tmp_file.name)  # Delete temporary file
#     return texts

# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main())


import streamlit as st
import os
import io
import tempfile
from PyPDF2 import PdfReader
from langchain.prompts import PromptTemplate  
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA

# Constants
DB_FAISS_PATH = "db_faiss/"
cache = {}

# Define custom prompt template for QA retrieval
def set_custom_prompt():
    custom_prompt_template = """
    Use the following pieces of information to answer the user's question. If you don't know the answer, please just say that you don't know the answer, don't try to make up an answer.

    Context: {context}
    Question: {question}

    Only returns the helpful answer below and nothing else.
    Helpful answer:
    """
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
    return prompt

# Load Language Model
def load_llm():
    llm = CTransformers(
        model="llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens=1024,
        temperature=0.7
    )
    return llm

# Setup retrieval question answering chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain

# Initialize QA bot
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa_chain = retrieval_qa_chain(llm, qa_prompt, db)
    return qa_chain

# Cache function results
def cache_result(key, func, *args, **kwargs):
    if key in cache:
        return cache[key]
    result = func(*args, **kwargs)
    cache[key] = result
    return result

# Asynchronous function to get answer from QA chain
async def get_answer(chain, user_input):
    res = await chain.acall(user_input)
    answer = res["result"]
    sources = res["source_documents"]
    if sources:
        answer += f"\nSources:" + str(sources)
    else:
        answer += f"\nNo Sources Found"
    return answer

# Main function
async def main():
    st.title("Mines Bot")
    cache_result("qa_chain", qa_bot)  
    chain = cache["qa_chain"]  

    # File upload UI
    st.sidebar.title("Upload PDF")
    uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        # Process the PDF file
        with st.spinner('Processing PDF...'):
            texts = extract_text_from_pdf(uploaded_file)
            for text in texts:
                answer = await get_answer(chain, text)
                st.write(answer)

# Extract text from PDF
def extract_text_from_pdf(uploaded_file):
    texts = []
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file.close()
        with open(tmp_file.name, "rb") as file:
            reader = PdfReader(file)
            num_pages = len(reader.pages)
            for page_num in range(num_pages):
                page = reader.pages[page_num]
                text = page.extract_text()
                # Split text into chunks of maximum context length
                chunk_size = 1024
                chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
                texts.extend(chunks)
    os.unlink(tmp_file.name)  
    return texts

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

