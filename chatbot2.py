import streamlit as st
from langchain.prompts import PromptTemplate  # Changed import statement
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA

DB_FAISS_PATH = "db_faiss/"
cache = {}  # Create a cache dictionary

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vector store
    """
    custom_prompt_template = """ Use the following pieces of information to answer the user's question. IF you don't know the answer, please just say that you don't know the answer, don't try to make up an answer.

 Context : {context}
 Question: {question}

 Only returns the helpful answer below and nothing else.
 Helpful answer:

 """
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
    return prompt

def load_llm():
    llm = CTransformers(
        model="llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm

def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain

def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa_chain = retrieval_qa_chain(llm, qa_prompt, db)
    return qa_chain

def cache_result(key, func, *args, **kwargs):
    """
    Caches the result of a function to avoid recomputation for the same inputs.
    """
    if key in cache:
        return cache[key]
    result = func(*args, **kwargs)
    cache[key] = result
    return result

async def get_answer(chain, user_input):
    res = await chain.acall(user_input)
    answer = res["result"]
    sources = res["source_documents"]
    if sources:
        answer += f"\nSources:" + str(sources)
    else:
        answer += f"\nNo Sources Found"
    return answer

async def main():
    st.title("Mines Bot")
    cache_result("qa_chain", qa_bot)  # Initialize the QA chain
    chain = cache["qa_chain"]  # Retrieve the initialized QA chain

    user_input = st.text_input("Enter your query here:")
    if st.button("Submit"):
        answer = await get_answer(chain, user_input)
        st.write(answer)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
