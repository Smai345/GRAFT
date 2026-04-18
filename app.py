import streamlit as st
import re
import spacy
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

# --- API Setup ---
api_key = "AIzaSyAbXR2j3O0ZF6e_gZAfmN1Qf4LNx8CR-zc"
genai.configure(api_key=api_key)

# --- Cache Setup (Runs once so the UI is fast) ---
@st.cache_resource
def initialize_system():
    doc_text = """
    Whitepaper: AuraFlow - A Decentralized Framework for Adaptive AI
    1. Introduction
    In the landscape of artificial intelligence, traditional monolithic systems present significant challenges in scalability, adaptability, and resilience. AuraFlow is a novel, decentralized framework designed to overcome these limitations by enabling the creation of stateful, multi-agent AI systems. It is built on the core principles of decentralization, modularity, and emergent intelligence. By distributing tasks across specialized, independent agents, AuraFlow creates robust systems that can adapt to new information in real-time without requiring complete model retraining. This document outlines the core components, architecture, and primary use cases of the AuraFlow framework.
    2. Core Components
    The AuraFlow framework is composed of four primary components that work in synergy. Each component is a specialized agent with a distinct role.
    2.1 The Cognition Core
    The Cognition Core is the central reasoning engine of an AuraFlow instance. Unlike traditional neural networks, it utilizes a proprietary Probabilistic Logic Network (PLN) for decision-making. This allows the Core to handle uncertainty, reason with incomplete information, and provide transparent, explainable outputs. The Cognition Core is responsible for high-level task decomposition, planning, and final response synthesis. Its performance is heavily dependent on the quality of processed data it receives from the Data Weavers.
    2.2 Data Weavers
    Data Weavers are specialized agents tasked with data ingestion, preprocessing, and normalization. Each Weaver can be configured to handle specific data modalities, such as unstructured text, images, streaming time-series data, or structured database records. They clean and transform raw data into a standardized format that the Cognition Core can efficiently process. This modular approach allows an AuraFlow system to seamlessly integrate new data sources by simply deploying a new, appropriately configured Data Weaver. Communication between Data Weavers and the Cognition Core is managed by the Synapse Bridge.
    2.3 The Synapse Bridge
    The Synapse Bridge is the high-bandwidth, low-latency communication backbone of the AuraFlow framework. It facilitates interaction between all components, primarily managing the flow of information from the Data Weavers to the Cognition Core and broadcasting the Core's directives to other agents. It uses a custom, lightweight data exchange protocol called the Neuro-Link Protocol, which ensures secure and efficient data transmission, a critical feature for real-time applications.
    2.4 The Sentinel Layer
    The Sentinel Layer acts as the ethical and security guardian of the system. It is a specialized validation agent that monitors the outputs and behavior of the Cognition Core in real-time. The Sentinel Layer is responsible for applying ethical constraints, enforcing operational boundaries, and preventing the generation of harmful or biased outputs. It can veto or flag a decision made by the Cognition Core if it violates pre-defined rules, ensuring that the system operates safely and responsibly.
    3. System Architecture and Data Flow
    The architecture of AuraFlow is inherently decentralized. A typical workflow for processing a user query follows these steps:
    1. A query is received by the system.
    2. Relevant Data Weavers are activated to gather and process external or internal data related to the query.
    3. The processed data is transmitted securely via the Synapse Bridge to the Cognition Core.
    4. The Cognition Core uses its PLN to analyze the data, reason about the query, and formulate a plan or response.
    5. Before being finalized, the proposed response is sent to the Sentinel Layer for validation.
    6. If approved, the final response is generated and delivered.
    This modular data flow ensures that each component can be independently upgraded or scaled. For instance, if processing speed becomes a bottleneck, more Data Weaver instances can be deployed without altering the Cognition Core.
    4. Key Applications
    The unique architecture of AuraFlow makes it suitable for complex, dynamic environments.
    Real-time Market Analysis: An AuraFlow system can deploy multiple Data Weavers to monitor financial news, social media sentiment, and stock market data simultaneously. The Cognition Core can then synthesize this information to identify trends and risks, while the Sentinel Layer ensures that trading recommendations adhere to regulatory compliance.
    Autonomous Scientific Research: In this scenario, a network of AuraFlow instances can collaborate on research. One instance could use its Data Weavers to analyze experimental data from lab equipment, while another analyzes existing scientific literature. The Cognition Cores could then exchange findings via their Synapse Bridges to formulate new hypotheses, accelerating the pace of discovery.
    """
    cleaned_doc = re.sub(r'\n+', '\n', doc_text)
    cleaned_doc = re.sub(r' +', ' ', cleaned_doc).strip()

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(cleaned_doc)
    sentences = [sent.text.strip() for sent in doc.sents]

    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) > 500 and current_chunk:
            chunks.append(current_chunk)
            overlap_text = ' '.join(current_chunk.split()[-30:])
            current_chunk = overlap_text + " " + sentence
        else:
            current_chunk += " " + sentence
    if current_chunk:
        chunks.append(current_chunk.strip())

    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    embeddings = model.encode(chunks)

    client = chromadb.Client()
    try:
        client.delete_collection("auraflow_docs")
    except:
        pass
    collection = client.create_collection(name="auraflow_docs")
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(embeddings=embeddings, documents=chunks, ids=ids)

    graph = {i: [] for i in range(len(chunks))}
    for i in range(len(chunks) - 1):
        graph[i].append(i + 1)

    similarity_matrix = cosine_similarity(embeddings)
    similar_pairs = np.where(similarity_matrix > 0.80)
    for i, j in zip(*similar_pairs):
        if i != j and j not in graph[i]:
            graph[i].append(j)

    return model, collection, chunks, graph

# --- Core Functions ---
def query_vector_db_with_scores(query, collection, model, n_results=3):
    query_embedding = model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
    return results['documents'][0], results['distances'][0]

def expand_with_graph(initial_chunks, graph, all_chunks):
    expanded_chunks_set = set(initial_chunks)
    chunk_to_id = {chunk: i for i, chunk in enumerate(all_chunks)}
    for chunk_text in initial_chunks:
        node_id = chunk_to_id.get(chunk_text)
        if node_id is None: continue
        for neighbor_id in graph.get(node_id, []):
            expanded_chunks_set.add(all_chunks[neighbor_id])
    return list(expanded_chunks_set)

def rerank_chunks(query, chunks_to_rank, model, top_n=5):
    query_embedding = model.encode(query)
    chunk_embeddings = model.encode(chunks_to_rank)
    similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
    scored_chunks = list(zip(chunks_to_rank, similarities))
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    return [chunk for chunk, score in scored_chunks[:top_n]]

def merge_chunks(chunks):
    return "\n\n---\n\n".join(chunks)

def generate_llm_answer(query, context):
    prompt_template = f"""
    Answer the following query based only on the provided context.
    If the context does not contain the answer, state that the information is not available in the document.
    Be concise and do not add any information that is not present in the context.
    CONTEXT: {context}
    QUERY: {query}
    ANSWER:
    """
    try:
        llm = genai.GenerativeModel('gemini-2.5-flash')
        response = llm.generate_content(prompt_template)
        return response.text
    except Exception as e:
        return f"An error occurred: {e}"

# Insert this at the very top of your "UI Layout" section
# Add a sidebar to explain the technical stack and project status. This keeps the main screen clean for the live demo.
st.set_page_config(page_title="GRAFT: AuraFlow", page_icon="🧠", layout="wide")

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100) # Generic AI icon
    st.title("Project GRAFT")
    st.markdown("""
    **Methodology:** RAFT + GraphRAG [cite: 31, 33]
    **Model:** Gemini 2.5 Flash
    **Vector DB:** ChromaDB
    ---
    **Key Features:**
    - Noise Robustness [cite: 44]
    - Multi-Document Reasoning [cite: 53]
    - Semantic Graph Retrieval
    """)
    st.status("System Online", state="running")

# --- UI Layout ---

st.title("🧠 AuraFlow GraphRAG Demo")
st.write("Ask questions about the decentralized AuraFlow framework.")

with st.spinner("Initializing Database and Graph (Runs once)..."):
    model, collection, chunks, graph = initialize_system()

query = st.text_input("Enter your question:")

if st.button("Generate Answer"):

    if query:
        with st.spinner("Searching and Reasoning..."):
            initial_chunks, _ = query_vector_db_with_scores(query, collection, model)
            expanded_chunks = expand_with_graph(initial_chunks, graph, chunks)
            reranked_chunks = rerank_chunks(query, expanded_chunks, model, top_n=5)

            # Step 4: Metrics [cite: 3]
            col1, col2, col3 = st.columns(3)
            col1.metric("Retrieved Chunks", len(initial_chunks))
            col2.metric("Graph Neighbors", len(expanded_chunks) - len(initial_chunks))
            col3.metric("Reranked Output", "Top 5")

            context = merge_chunks(reranked_chunks)
            answer = generate_llm_answer(query, context)
            
            st.divider()
            st.subheader("💡 Answer from Agent")
            with st.chat_message("assistant"):
                st.write(answer)
            
            with st.expander("🔍 System Trace (The 'Proof' for Judges)"):
                st.write("**The system retrieved these specific documents to ground the answer:**")
                for i, c in enumerate(reranked_chunks):
                    st.info(f"**Source Document {i+1}**: {c}")
    else:
        st.warning("Please enter a question to start the retrieval process.")