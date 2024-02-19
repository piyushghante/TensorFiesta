import asyncio
from langchain.prompts import PromptTemplate  # Changed import statement
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl

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

# Chainlit
@cl.on_chat_start
async def start():
    cache_result("qa_chain", qa_bot)  # Initialize the QA chain
    chain = cache["qa_chain"]  # Retrieve the initialized QA chain
    msg = cl.Message(content="Starting the bot.....")
    await msg.send()
    msg.content = "Hi, Welcome to the Mines Bot. What is your query?"
    await msg.update()

@cl.on_message
async def main(message):
    # Create a new chain object
    chain = cache["qa_chain"]

    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    
    # Extract text content from message if it's a Message object
    text_content = message.content if isinstance(message, cl.Message) else message
    
    res = await chain.acall(text_content, callbacks=[cb])  # Pass text content instead of message
    answer = res["result"]
    sources = res["source_documents"]

    if sources:
        answer += f"\nSources:" + str(sources)
    else:
        answer += f"\nNo Sources Found"

    await cl.Message(content=answer).send()

if __name__ == "__main__":
    cl.run_forever()
