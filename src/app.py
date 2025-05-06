import os
from dotenv import load_dotenv, find_dotenv
import tempfile
import streamlit as st
from streamlit_feedback import streamlit_feedback
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Enable to supports multiple file formats
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.word_document import Docx2txtLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders.wikipedia import WikipediaLoader

# callbacks
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.tracers.context import collect_runs

# RAG chain related
from langchain_core.prompts import ChatMessagePromptTemplate, MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain_community.llms import openai
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever
from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx
from langsmith import Client

# Ponecone free tier sometimes runs into unpected Auth error, so use chroma insted. 
#from langchain_pinecone import PineconeVectorStore
#import pinecone
#from pinecone import ServerlessSpec
from langchain_chroma import Chroma
import chromadb
from chromadb.config import Settings

import pandas as pd

# import from components/llm
from components.llm import embedding_model


load_dotenv(find_dotenv(), override=True)
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
LANGCHAIN_API_KEY= os.environ.get('LANGCHAIN_API_KEY')
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
TAVILY_API_KEY = os.environ.get('TAVILY_API_KEY')

# Initialize Streamlit session
if 'costs' not in st.session_state:
    st.session_state.costs = []
if 'latest_run_id' not in st.session_state:
    st.session_state.latest_run_id = None
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'feedback_submitted' not in st.session_state:
    st.session_state.feedback_submitted = False

# Get session_id by Streamlit
ctx = get_script_run_ctx()

# Instanciate StreamlitChatMessageHistory, which can keep chathistory with streamlit function

# default key = "langchain_messages"
chat_history = StreamlitChatMessageHistory(key="langchain_messages")


def get_db_conn()->chromadb.HttpClient:
    """
    The function that secure connect to DB and return the client
    Return:
     chromadb.HttpClient Object
    """
    try:
        client = chromadb.HttpClient(
            host='db',
            port=8000,
            settings=Settings(allow_reset=True, anonymized_telemetry=False),
        )
        print("✅ Connected to ChromaDB")
        return client
    except Exception as e:
        print("❌ Failed to connect to ChromaDB:", e)

client = get_db_conn()



# Initizalize Start Page
def init_page():
    st.set_page_config(
        page_title="Ask My Documents",
        page_icon="📄"
    )
    st.sidebar.title("Nav")


# load and parse document and returned parsed oned.
def load_document(file):
    name, extension = os.path.splitext(file.name)

    # create temporary file path  using `tmpfile(file object)`b
    # NamedTemporaryFile ... The standard function which create temporary file path. (delete=False) is option which keeps file even after `with statement`
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        # getvalue() ... get binary content of the file(file object) and write it on temporary file path
        tmp_file.write(file.getvalue())
        # assign variable(tmp_file_path) to use the later process.
        tmp_file_path = tmp_file.name

    if extension == '.pdf':
        print(f'Loading {file}...')
        loader = PyPDFLoader(file_path=tmp_file_path)
    elif extension == '.docx':
        print(f'Loading {file}...')
        loader = Docx2txtLoader(file_path=tmp_file_path)
    elif extension == '.txt':
        print(f'Loading {file}...')
        loader = TextLoader(file_path=tmp_file_path)
    else:
        print('Document format is not supported')
        return None
    
    doc = loader.load()
    return doc
    
# will add Wikipedia Mode 
def load_from_wikipedia(query, lang='en', load_max_docs=2):
    from langchain.document_loaders import WikipediaLoader
    loader = WikipediaLoader(query=query, lang=lang, load_max_docs=load_max_docs)
    doc = loader.load()
    return doc

def create_chunks(docs, chunk_size=1024, chunk_overlap=256):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        #separators=["\n\n", "\n", ". ", "。", "．", "! ", "? ", "！", "？", " ", ""]
        )
    chunks = text_splitter.split_documents(docs)
    return chunks

def file_uploader():
    uploaded_file = st.file_uploader(
        label='Upload your Documents here😇',
        type=['pdf', 'docx', 'txt'],
        accept_multiple_files=True
    )
    print(f'DEBUG... uploadef_file: {uploaded_file}')

    if uploaded_file:
        if isinstance(uploaded_file, list):
            doc_list = []
            for file in uploaded_file:
                doc = load_document(file)
                print(f'DEBUG... doc = load_document(file): {doc}')
                if doc:
                    doc_list.extend(doc)
                    print(f'DEBUG... chunks = create_chunks(doc_list)')
            if doc_list:
                chunks = create_chunks(doc_list)
                return chunks
        else:
            doc = load_document(uploaded_file)
            if doc:
                chunks = create_chunks(doc)
                return chunks
        
    return None



def build_vector_store(chunks):

    # create IDs  ... if creats 
    original_ids = []
    for chunk in chunks:
        #source_ = os.path.splitext(chunk.metadata['source'])[0]
        source_ = os.path.splitext(chunk.metadata.get('source','unknown'))[0]
        page_ = chunk.metadata.get('page', 0)
        #start_ = chunk.metadata['start_index']
        start_ = chunk.metadata.get('start_index', 0)
        #id_str = f"{source_}_{start_:08}"
        id_str = f"{source_}_{page_}_{start_}"
        original_ids.append(id_str)
        print(f'DEBUG... chunk/id_str: {chunk} / {id_str}')
        print(f'DEBUG... originas_ids: {original_ids}')
        print(f'DEBUG... Num of IDs: {len(original_ids)}')

    client = get_db_conn()

    # create chorma_db_server
    vector_store_server = Chroma(
        collection_name = "collection_name_server",
        embedding_function = embedding_model, # call from another file
        #persist_directory = "./chromadb_server"
        client = client
    )

    # add chunks to vector_store (add_documents = upsert)
    vector_store_server.add_documents(
        documents = chunks,
        ids=original_ids
    )
    return vector_store_server

           
def page_documents_uploader_and_build_vector_db():
    st.title("Documents Upload")
    container = st.container()
    with container:
        st.markdown("## Upload Documents ##")

        # upload documents and return chunks
        chunks = file_uploader()

        if chunks:
            vector_store_server = build_vector_store(chunks)
            client = get_db_conn()
            st.write("DEBUG: st.session =", (st.session_state))
            #st.write("DEBUG: st.session =", (st.session_state))

            vector_store_server = Chroma(
                collection_name = "collection_name_server",
                embedding_function = embedding_model, # call from another file
                #persist_directory = "./chromadb_server"
                client = client

            )
            st.session_state.vector_store = vector_store_server

        else:
            st.warning('file is not uploaded or unsupported format')


def select_model():
    model = st.sidebar.radio('Choose a model: ', ("GPT-4o-mini","GPT-4o"))
    print(f'DEBUG... model: {model} ')
    if model == "GPT-4o-mini":
        st.session_state.model_name = "gpt-4o-mini"
    else:
        st.session_state.model_name = "gpt-4o"

    model = ChatOpenAI(model=st.session_state.model_name, temperature=0)
    return model

    
# if user post query, get answer from model with chatHsitory based on query
def get_answer_with_history(vector_store_server, model, query, session_id='unused'):
    ### Build "Instance of 'create_history_aware_retriever'"
    ## from langchain.chains import create_history_aware_retriever
    #  history_aware_retriever = create_history_aware_retriever(llm, retriever, contexualize_q_prompt)

    ### Build retriever 
    ## retriever = vector_store_server.as_retriever()
    client = get_db_conn()

    vector_store_server = Chroma(
        collection_name = "collection_name_server",
        embedding_function = embedding_model, # call from another file
        #persist_directory = "./chromadb_server"
        client = client
    )
    retriever = vector_store_server.as_retriever(search_type='similarity', search_kwargs={'k':10})

    ### Build chat_prmopt
    ## create contexualize_q_system_prompt
    # crate contexualize_q_prompt (system: contexualize_q_system_prompt, MessagesPlaceholder(chat_history), human:'{input}' )
    contexualize_q_system_prompt = (
        "Reformulate the latest user question into a standalone question using relevant context from the chat history." 
        "Make the question as clear and specific as possible."
        "Do not answer the question."
        "If no reformulation is needed, return the question as is."
    )
    contexualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ('system', contexualize_q_system_prompt),
            MessagesPlaceholder('chat_history'),
            ('human', '{input}')
        ]
    )

    system_message = """
    You are an AI assistant. Use only the provided reference information to answer the user's question. 
    If a complete answer cannot be found, try to provide a helpful partial answer based on what is available. 
    If no useful information is available, suggest how the user could rephrase or clarify their question to get a better answer. 
    Do not use outside knowledge beyond the provided context.
    {context}
    """
    human_message = "Question: {input}"
        # create chat prompt
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            MessagesPlaceholder("chat_history"),
            ("human", human_message)

        ]
    )
    # {context} ... Python の文字列フォーマットのプレースホルダ
    
    # the order of process...
    # - get relevant doc with retriever
    # - Instanciate create_history_aware_retriever (histor_aware_retiever)   <-- history_aware_retriever = create_history_aware_retiever(model...)
    # - place history_aware_retriever to {context}   <--  add_context = RunnablePassThrough.assign(context=history_aware_retriever) 
    # - chain -- add_context > chat_prompt > model 


    # create_history_aware_retriever ... build-in function of langchain   -- from langchain.chains import create_history_aware_retriever
    # it creats a chain that takes conversation history and returns documents
    # if there is no chat_history, then the input is just passed directly to the retiever. If there is chat_history then the prompt and LLM will be used to geerate a search history.
    # that serch query is then passed to the retriever. 
    history_aware_retriever = create_history_aware_retriever(
        model, retriever, contexualize_q_prompt
    )

    # Add the document serached by Retriever to {context}
    add_context = RunnablePassthrough.assign(context=history_aware_retriever)
    # Define Chain
    # process add_context > chat_prompt > model > StrOutputParser() in order. 
    rag_chain = add_context | chat_prompt | model | StrOutputParser()

    runnable_with_history = RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: chat_history,
        input_messages_key='input',
        history_messages_key='chat_history'
    )

    # get_openai_callback() .... Record OpenAI usage fee (Token, fees, etc.)
    # collect_runs() ... Record execution history in LangSimith (can be analyzed and visualized later)
    # .invoke*()  ... Excecute Chain and to get the answer (and cost)
    # run_id ... Use User Feedback function
    with get_openai_callback() as cb:
        with collect_runs() as runs_cb:
            answer = runnable_with_history.invoke({'input': query}, config={'configurable': {'session_id': session_id}})
            run_id = runs_cb.traced_runs[0].id
            st.session_state.latest_run_id = run_id
    return answer, cb.total_cost



def page_aks_my_docs():
    # Crate Chat UI Title
    st.title('📄 Ask My PDF(s)')
    """
    Messages are automatically saved in the Session State across iterations.
    You can view the contents of the Sesstion State ih the Expander below.
    """

    view_messages = st.expander("View the message contents in session state")
    # select models
    model = select_model()

    # Display the chathistory 
    if "vector_store" in st.session_state:
        st.write("DEBUG: st.session =", (st.session_state))

        vector_store_server = st.session_state.vector_store
    else:
        vector_store_server = None

    if vector_store_server:
        for msg in chat_history.messages:
            st.chat_message(msg.type).write(msg.content)

        # If user input question, invoke chain to get answer
        if query := st.chat_input():
            st.chat_message('human').write(query)
            with st.spinner('ChatGPT is typing...'):

                # Get answer from query
                answer, cost = get_answer_with_history(vector_store_server, model, query, session_id=ctx.session_id)

            st.chat_message('ai').write(answer)
            # display cost
            st.session_state.costs.append(cost)

            # Rest Feedback status
            st.session_state.feedback_submitted = False

            # Keep the latest "run_id"
            #st.session_state.latest_run_id = st.session_state.feedback_submitted:

        # Display Feedback Form with 'streamlit_feedback' and call 'send_feckback' function.
        if st.session_state.get('latest_run_id') and not st.session_state.feedback_submitted:
            run_id = st.session_state.latest_run_id
            feedback = streamlit_feedback(
                feedback_type="thumbs",
                optional_text_label="[Optional] Please provide an explanation",
                on_submit=send_feedback,
                key=f"feedback_key_{run_id}",
                args=[run_id]
            )

# Send Feedback to LangSmith
def send_feedback(user_feedback, run_id):
    scores = {"👍":1, "👎":0}
    score_key = user_feedback['score']
    score = scores[score_key]
    comment = user_feedback.get('text')

    client = Client()
    client.create_feedback(
        run_id=run_id,
        key="thumbs",
        score=score,
        comment=comment,
    )

    st.session_state.feedback_submitted = True
    st.success("Thank you for your feedback")


def main():
    # Initialize Page
    init_page()

    selection = st.sidebar.radio("Go to", ["Documents Upload","Ask My Docs"])
    if selection == "Documents Upload":
        page_documents_uploader_and_build_vector_db()
    if selection == "Ask My Docs":
        page_aks_my_docs()
    
    costs = st.session_state.get('costs', [])
    st.sidebar.markdown("## Costs")
    st.sidebar.markdown(f'**Total Cost: $ {sum(costs):.5f} **')
    for cost in costs:
        st.sidebar.markdown(f'cost: - $ {cost:.5f}')


if __name__ == '__main__':
    main()


