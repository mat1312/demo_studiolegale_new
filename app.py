import os
import streamlit as st
import requests
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain import PromptTemplate, LLMChain
import streamlit.components.v1 as components

# Carica le variabili d'ambiente dal file .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("La variabile OPENAI_API_KEY non è stata caricata correttamente.")
    st.stop()

elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
if not elevenlabs_api_key:
    st.error("La variabile ELEVENLABS_API_KEY non è stata caricata correttamente.")
    st.stop()

# Imposta il percorso del vector DB persistente
persist_directory = "vectordb"
if not os.path.exists(persist_directory):
    st.error("Il vector DB non è stato trovato. Esegui prima l'ingestione con 'ingest.py'.")
    st.stop()

# Carica il vector DB abilitando la deserializzazione per file pickle
embeddings = OpenAIEmbeddings()
vector_store = FAISS.load_local(persist_directory, embeddings, allow_dangerous_deserialization=True)

# Definisci il system prompt (messaggio di sistema)
SYSTEM_PROMPT = """
Sei "Avvocato Virtuale" dello studio legale Di Pietro in Italia. Il tuo compito è fornire informazioni e orientamenti preliminari su questioni legali, mantenendo un tono professionale, chiaro ed empatico e rivolgendoti sempre con il "Lei".

Rispondi sia a domande legali che a domande generali. Se necessario, consulta la tua knowledge base per informazioni su normative, contratti, ecc.
. Se il caso richiede approfondimenti, informa l’utente che un avvocato lo contatterà.
Fai una domanda alla volta e guida l’utente in modo naturale.
Chiudi la conversazione solo dopo aver raccolto l'email e il numero di telefono dell'utente.
"""

# Inizializza il modello senza passare 'system_message'
llm = ChatOpenAI(temperature=0.5, model="gpt-4o-mini")  # oppure "gpt-4" se disponibile

# Inizializza la catena RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(),
    return_source_documents=True
)

# Funzioni per recuperare conversazioni da ElevenLabs
def get_last_conversation(agent_id, api_key):
    url = "https://api.elevenlabs.io/v1/convai/conversations"
    headers = {"xi-api-key": api_key}
    params = {"agent_id": agent_id, "page_size": 1}
    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
         st.error(f"Errore nel recuperare le conversazioni: {response.status_code}")
         return None
    data = response.json()
    conversations = data.get("conversations", [])
    if not conversations:
         st.info("Nessuna conversazione trovata.")
         return None
    return conversations[0].get("conversation_id")

def get_conversation_details(conversation_id, api_key):
    url = f"https://api.elevenlabs.io/v1/convai/conversations/{conversation_id}"
    headers = {"xi-api-key": api_key}
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
         st.error(f"Errore nel recuperare i dettagli della conversazione: {response.status_code}")
         return None
    return response.json()

# Configurazione della pagina Streamlit
st.set_page_config(page_title="Assistente Legale", layout="wide")
st.title("Assistente Legale")

# SEZIONE: Q&A tramite LangChain e OpenAI
st.subheader("Fai una domanda su questioni legali, contratti, normative, ecc.")
user_input = st.text_input("Inserisci la tua domanda qui")
if st.button("Invia") and user_input:
    with st.spinner("Generazione della risposta..."):
        # Crea il prompt concatenando il messaggio di sistema e la domanda dell'utente
        prompt_text = SYSTEM_PROMPT.strip() + "\n\nDomanda: " + user_input.strip()
        result = qa_chain.invoke(prompt_text)
        # Estrai la risposta e le fonti (se presenti)
        answer = result["result"] if isinstance(result, dict) and "result" in result else result
        source_docs = result.get("source_documents", [])
    st.markdown(f"**Q:** {user_input}")
    st.markdown(f"**A:** {answer}")
    if source_docs:
        st.markdown("**Fonti:**")
        sources_dict = {}
        for doc in source_docs:
            metadata = doc.metadata
            if "source" in metadata:
                source = metadata["source"].replace("\\", "/")
                page = metadata.get("page", None)
                line = metadata.get("start_index", None)
                sources_dict.setdefault(source, []).append((page, line))
        for source, occurrences in sources_dict.items():
            file_name = os.path.basename(source)
            occ_list = []
            for p, l in occurrences:
                occ_str = ""
                if p is not None and p != 0:
                    occ_str += f"pagina {p}"
                if l is not None and l != 0:
                    occ_str += f", riga {l}" if occ_str else f"riga {l}"
                if occ_str:
                    occ_list.append(occ_str)
            occ_text = " - ".join(occ_list) if occ_list else ""
            st.markdown(f"- [{file_name} ({occ_text})]({source})" if occ_text else f"- [{file_name}]({source})")
    else:
        st.markdown("*Nessuna fonte disponibile.*")

# SEZIONE: Agent Conversazionale ElevenLabs (embedding del widget)
# SEZIONE: Agent Conversazionale ElevenLabs (embedding del widget)
# SEZIONE: Agent Conversazionale ElevenLabs (embedding del widget)
st.subheader("Agent Conversazionale ElevenLabs")
widget_html = """
<style>
  /* Container per il widget (desktop) */
  .widget-container {
    position: fixed;
    top: 50%;
    left: 20%;  /* Posizionato a sinistra su desktop */
    transform: translate(0, -50%);
    width: 600px;
    height: 600px;
    z-index: 9999;
  }
  /* Adattamento per dispositivi mobili */
  @media only screen and (max-width: 768px) {
    .widget-container {
      top: 5%;  /* Più in alto per dispositivi mobili */
      left: 50%;  /* Centra il widget orizzontalmente */
      transform: translate(-50%, 0);  /* Solo centratura orizzontale */
      width: 90%;  /* Occupa il 90% della larghezza dello schermo */
      height: auto; /* L'altezza si adatterà al contenuto oppure impostala fissa se necessario */
    }
  }
  /* Assicura che il widget usi tutto lo spazio del container */
  elevenlabs-convai {
    width: 100%;
    height: 100%;
  }
</style>
<div class="widget-container">
  <elevenlabs-convai agent-id="vE96ET0MG8Jlv2jZA5cq"></elevenlabs-convai>
</div>
<script src="https://elevenlabs.io/convai-widget/index.js" async></script>
"""
components.html(widget_html, height=600)



# SEZIONE: Transcript e Estrazione Contatti
st.subheader("Transcript e Estrazione Contatti")
col1, col2 = st.columns(2)

with col1:
    if st.button("Recupera conversazione"):
        with st.spinner("Recupero conversazione..."):
            agent_id = "vE96ET0MG8Jlv2jZA5cq"
            conv_id = get_last_conversation(agent_id, elevenlabs_api_key)
            if conv_id:
                details = get_conversation_details(conv_id, elevenlabs_api_key)
                if details:
                    transcript = details.get("transcript", [])
                    st.session_state["transcript"] = transcript  # Salva il transcript in sessione
                    if transcript:
                        st.markdown("#### Transcript")
                        for msg in transcript:
                            role = msg.get("role", "unknown")
                            time_in_call_secs = msg.get("time_in_call_secs", "")
                            message = msg.get("message", "")
                            st.markdown(f"**{role.capitalize()} [{time_in_call_secs}s]:** {message}")
                    else:
                        st.info("Nessun transcript disponibile")
            else:
                st.error("Nessuna conversazione trovata")

with col2:
    if st.button("Estrai contatti e informazioni"):
        transcript = st.session_state.get("transcript", [])
        if transcript:
            # Filtra solo i messaggi dell'utente
            user_messages = [msg.get("message", "") for msg in transcript if msg.get("role", "").lower() == "user"]
            transcript_text = "\n".join(user_messages)
            if transcript_text.strip():
                prompt_template = """
Analizza la seguente trascrizione di una conversazione tra un utente e un agente virtuale.
Estrai, se presenti, l'indirizzo email e il numero di telefono dell'utente e riassumi dettagliatamente in maniera strutturata con tutti i dettagli rilevanti per la richiesta di assistenza legale.
Rispondi nel seguente formato:
Email: <indirizzo email>
Telefono: <numero di telefono>
Riassunto: <riassunto dettagliato>

Se non trovi alcun dato, indica "Non trovato".
Se vedi qualche termine simile a "chiocciola" si tratta di un'email e cambiala con il carattere "@".

Trascrizione:
{transcript}
"""
                template = PromptTemplate(input_variables=["transcript"], template=prompt_template)
                contact_chain = LLMChain(llm=llm, prompt=template)
                with st.spinner("Analizzando la trascrizione per estrarre contatti..."):
                    contact_info = contact_chain.run(transcript=transcript_text)
                st.markdown("#### Contatti estratti")
                st.markdown(contact_info)
            else:
                st.info("Nessun messaggio utente trovato per l'analisi dei contatti.")
        else:
            st.info("Nessuna trascrizione disponibile per l'analisi dei contatti.")
