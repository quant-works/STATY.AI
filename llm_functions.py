import datetime
import glob
import json
import matplotlib.pyplot as plt
from multiprocessing import Pool
import ollama
import os
import platform
import psutil
import shutil
import time
import streamlit as st
from tqdm import tqdm
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_community.document_loaders import (
    CSVLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredHTMLLoader,     
    UnstructuredWordDocumentLoader,
)
#from langchain_community.document_loaders.powerpoint import UnstructuredPowerPointLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import nltk
print(nltk.data.path)


ai_avatar='default_data/logo.png'
hs_avatar = 'default_data/hs_logo1.png'

#----------------------------------------------------------------------------------------------
# Light Theme
def theme_func_light():
    st.markdown(
        f"""
    <style>
    div.stApp {{ background-color: white !important; }}
    </style> 
    
    """,
        unsafe_allow_html=True,
    )

#----------------------------------------------------------------------------------------------
# Dark Theme    
def theme_func_dark():
    st.markdown(
        f"""
    <style>
    div.stApp {{ background-color: rgb(114, 114, 114) !important; }}
    div[data-testid="stBottomBlockContainer"]{{ background-color: rgb(114, 114, 114) !important; }}
    </style> 
    
    """,
        unsafe_allow_html=True,
    )
#----------------------------------------------------------------------------------------------
# Settings sidebar    
def settings_sidebar():    
       
    # Display settings
    with st.sidebar.expander("Settings"):     
        sett_theme = st.selectbox('Theme', ["Light", "Dark"], help="Select app theme")        
        output_format=st.selectbox('Output format', [".txt", ".json"], help="Specify the format of saved chats.")

    # Check theme
    if sett_theme == "Dark":
        theme_func_dark()
    if sett_theme == "Light":
        theme_func_light()
    return sett_theme,output_format
#----------------------------------------------------------------------------------------------
# Select model sidebar   
def select_model_sidebar(): 
    menu_option=st.session_state['main_navigation']

    # Read avaiable models from the Ollama default folder
    available_ollama_models=ollama_check()  
    
    if available_ollama_models is None:
        model_names=[]    
    else:
        model_names = [model["name"] for model in available_ollama_models]
      
   
    # User selection of the LLM
    llm_name = st.sidebar.selectbox("Choose model", [""] + model_names)
    if llm_name == "" and menu_option =="Chat":    
        if model_names !=[]:
            st.info("Select LLM model from the 'Choose model' menu on the sidebar.")
        
        chat_info_expander=st.expander("More Info on Model Selection",expanded=False)
        
        with chat_info_expander:
                st.markdown("")
                st.markdown(
                "\n\n For chats in English, consider employing 'gemma' built by Google DeepMind (https://ollama.com/library/gemma), 'llama3' developed by Meta Inc (https://ollama.com/library/llama3) or 'mistral' built by a french comapny Mistral AI (https://ollama.com/library/mistral)."  
                "\n\n For chats in German, consider employing 'discolm-mfto-german' (https://huggingface.co/Blizado/discolm-mfto-7b-german-v0.1) or 'SauerkrautLM' (https://huggingface.co/VAGOsolutions/SauerkrautLM-7b-HerO)."  
                "\n\n You can download and delete models from the 'Manage LLMs' menu.") 
    elif llm_name == "" and menu_option =="Chat with my data":    
        if model_names !=[]:
            st.info("Select LLM model from the 'Choose model' menu on the sidebar." )
        
        chat_info_expander=st.expander("More Info on Chatting with Your Data",expanded=False)
        with chat_info_expander:
                st.markdown("")
                st.markdown("**Knowledge Base and RAG**")
                st.markdown("""
                    Imagine having a conversation with your data files like PDFs, Word files or Power-Point slides – all easily accessible through a chat interface. This is now possible with the power of Large Language Models (LLMs) and a technique called **Retrieval-Augmented Generation (RAG)**.

                    **Here's how it works:**

                    In order to chat with your own documents, you need to make them understandable for LLMs. Regular documents like PDFs or Word files aren't directly interpretable by these models. You need to transform them into a format the LLMs can understand and store them in a database accessible by LLMs. For data transformation, a technique called **embeddings** is used. A database where this data is stored is typically called a **knowledge base**.  

                    **The Magic of Embeddings:** An embedding model is essentially used to convert words into numerical codes that capture their meaning and relationships within the document. You can use here different embedding models, all known for their good performance and speed. 

                    **Chunking for Efficiency:**  While embedding is powerful, processing entire documents at once can be computationally expensive. That's where **chunking** comes in. You need to break down your documents into smaller, manageable pieces called chunks. One simple chunking strategy is the so called recursive chunking applied here. You simple specify the chunk size as the number of characters (e.g. 1500) and the text splitter tries to split your text into chuks of appximatelly the same size, while trying to keep the sentences together. This allows for faster processing and memory efficiency.
                    Smaller chunk size provide more granular detail but can lead to loss of context. Conversely, larger chunks are faster to process but may miss important nuances.

                    **Chunk overlap:**. Overlap allows some content to be included in both adjacent chunks, ensuring a smooth transition and preserving context across the document.

                    
                    Once your knowledge base is built with these embedded and chunked documents, the LLM acts as your intelligent assistant. It can understand your questions in natural language and delve into your knowledge base using RAG technology. 

                    """)
                
                st.markdown("")
                st.write("**Model Selection**")
                st.markdown( """
                For chats in English, consider employing 'gemma' built by Google DeepMind (https://ollama.com/library/gemma), 'llama3' developed by Meta Inc (https://ollama.com/library/llama3) or 'mistral' built by a french comapny Mistral AI (https://ollama.com/library/mistral).  
                
                For chats in German, consider employing 'discolm-mfto-german' (https://huggingface.co/Blizado/discolm-mfto-7b-german-v0.1) or 'SauerkrautLM' (https://huggingface.co/VAGOsolutions/SauerkrautLM-7b-HerO).  
                
                You can download and delete models from the 'Manage LLms' menu.
                
                
                """)  
    
    return menu_option, available_ollama_models,llm_name


#-----------------------------------------------------------------------------------------------------------
def staty_ai_info():
    st.markdown("**LLMs**")
    st.markdown("""                    
         LLMs are downloaded from Ollama. For a full list of available models, please check https://ollama.com/library.   
                    
        Once Ollama is set up, you might be wondering which model to download first. Here's why **"gemma:2b"** is a good choice:
                      
        **"gemma:2b"** is a relatively lightweight model, making it ideal for getting started. This translates to faster download times, lower resource consumption, and potentially smoother operation, especially if you have limited computational power.   
       
        While "gemma:2b" is an excellent starting point, consider exploring other models once you're comfortable. Ollama's library offers a diverse range of models tailored to specific use cases. The best choice ultimately depends on your requirements (e.g., task complexity, desired accuracy, CPU/GPU power).        
        
                  """)


#-----------------------------------------------------------------------------------------------------------

# model details settings expander
def model_settings(available_ollama_models,llm_name):
   
    # llm details object
    llm_details = [model for model in available_ollama_models if model["name"] == llm_name][0]
        
    # convert size in llm_details from bytes to GB (human-friendly display)
    if type(llm_details["size"]) != str:
        llm_details["size"] = f"{round(llm_details['size'] / 1e9, 2)} GB"

    # display llm details
    with st.sidebar.expander("LLM Info"):
        st.markdown(
            f"Family: {llm_details['details']['family']}\n\n"
            f"Model size: {llm_details['size']}\n\n"
            f"Parameter number: {llm_details['details']['parameter_size']}\n\n"
            f"Quantization level: {llm_details['details']['quantization_level']}")
 

    # display llm settings   
    with st.sidebar.expander("LLM Settings"):
        system_promt="."+st.text_area("Specify session promt",
                value="Use friendly and informative tone in your responses.", 
                help="A system prompt is a secret instruction that AI model use to understand your requests better. If you change the default promt, press 'Clear chat' to reset the 'memory' of the model.")
        model_temp=st.number_input("Specify model temperature",
                                    min_value=0.0,
                                    max_value=1.0,
                                    step=0.1,
                                    value=0.2,
                                    help="Higher temperature = more creative (but potentially strange) outputs, lower temperature = more coherent (but potentially less surprising) results. If you change the default value, press 'Clear chat' to reset the 'memory' of the model.")
     
    return model_temp,system_promt
#----------------------------------------------------------------------------------------------
# Check if Ollama is installed and running (_home)
def ollama_check_home():
    
    try:
        available_ollama_models = ollama.list()["models"]
    except Exception as e:
        st.error("""
                Oops! It seems there's an issue with Ollama.  
                Please ensure that Ollama is installed and running. 
                You can download Ollama from https://ollama.com.  
                After starting Ollama, don't forget to reload this page.                  
                 """)
    
     
#----------------------------------------------------------------------------------------------
# Check if Ollama is installed and running
def ollama_check():

    available_ollama_models=[]

    try:
        available_ollama_models = ollama.list()["models"]
    except Exception as e:
        st.error("Please make sure you have Ollama installed and running!  You can download Ollama from https://ollama.com.  \n Please reload this page after starting Ollama.   ")
        st.stop()

    
    if available_ollama_models==[]:
        st.error("**You don't have any models yet.** Install some suitable for your RAM from the menu 'Manage LLMs/Download LLMs'. \n\n Here's why **'gemma:2b'** is a good choice:**'gemma:2b'** is a relatively lightweight model, making it ideal for getting started. This translates to faster download times, lower resource consumption, and potentially smoother operation, especially if you have limited computational power.")
               
    return available_ollama_models  



#----------------------------------------------------------------------------------------------
# Download model from Ollama    
def pull_model(pull_model_name):
    import json
    ollama_stream_info = st.empty()  
    # Display a message indicating that the download process has started
    ollama_stream_info.info("Downloading model...")          
    pull_request=ollama.pull(pull_model_name,stream=True)

    # Add progress bar:
    my_bar = st.progress(0.0)                    
    for response in pull_request:
        #ollama_stream_info.info(json.dumps(response["status"]))
        if "status" in response: 
            status = response["status"]
            ollama_stream_info.info(f"Status: {status} ")
        if "completed" in response and "total" in response:
            completed = response["completed"]
            total = response["total"]

            if isinstance(completed, int) and isinstance(total, int):  # Check if both are numbers
                # Calculate percentage with 2 decimal places
                percentage = (100 * completed / total)
                formatted_percentage = f"{percentage:.2f}%"  # Format with 2 decimal places

                ollama_stream_info.info(f"Main model file download progress: {formatted_percentage} completed")
                my_bar.progress(percentage/100)
    # Once the download completes, display a success message
    ollama_stream_info.success("Model loaded successfully!")

#--------------------------------------------------------------------------------------------------
# Delete model from PC
def delete_model(delete_model_name):
    import json
    ollama_stream_info = st.empty()  
    # Display a message indicating that the download process has started
    ollama_stream_info.info("Deleting model...")          
    delete_request=ollama.delete(delete_model_name)
                       
    for response in delete_request:
        st.write(response)
   

#--------------------------------------------------------------------------------------------------
# Check PC info (OS, RAM, disk space)
def confuguration_check():
        
    # Check the operating system
    os_name = platform.system()

    if os_name == "Windows":
        print("Operating system: Windows")
    elif os_name == "Darwin":
        print("Operating system: macOS")
    elif os_name == "Linux":
        print("Operating system: Linux")    
    else:
        st.error("Operating system not supported! You can try using the tool, but it probably won't work properly!")
        
    # Get disk space information
    disk_usage = psutil.disk_usage("/")  
    total_disk_space = disk_usage.total / (1024**3)  # Convert to GB
    used_disk_space = disk_usage.used / (1024**3)
    free_disk_space = disk_usage.free / (1024**3)


    # Get RAM information
    total_ram = psutil.virtual_memory().total / (1024**3)
    available_ram = psutil.virtual_memory().available / (1024**3)
    used_ram = total_ram - available_ram
    
    print(f"Free disk space: {free_disk_space:.2f} GB")
    print(f"Available RAM: {available_ram:.2f} GB")
    

    return(os_name, total_ram,available_ram,total_disk_space,free_disk_space )

#--------------------------------------------------------------------------------------------------
# Select models based on RAM
def ram_based_models(total_ram):

    models_dictionary = {
    "sauerkrautlm-7b-hero": "German text generation model with 7 billion parameters, trained on a massive dataset of German text (https://huggingface.co/VAGOsolutions/SauerkrautLM-7b-HerO).",
    "gemma:2b": "A 2-billion parameter lightweight text model from Google DeepMind, ideal for various tasks where a smaller, efficient model is preferred (https://ollama.com/library/gemma).",
    "gemma:7b": "A member of the Gemma family from Google DeepMind, featuring 7 billion parameters for more complex text processing tasks (https://ollama.com/library/gemma).",
    "llama3:8b": "A member of the Llama 3 family of models developed by Meta Inc (https://ollama.com/library/llama3).",
    "qwen:0.5b": "A compact text model with only 0.5 billion parameters, suitable for tasks on resource-constrained environments (https://ollama.com/library/qwen).",
    "llava:7b": "Combines a vision encoder and Vicuna for general-purpose visual and language understanding (https://ollama.com/library/llava).",
    "llava:13b": "Combines a vision encoder and Vicuna for general-purpose visual and language understanding (https://ollama.com/library/llava).",
    "llama2:latest": "Llama 2 is released by Meta Platforms, Inc., featuring 7 billion parameters (https://ollama.com/library/llama2)",
    "llama2:13b": "Llama 2 is released by Meta Platforms, Inc., featuring 13 billion parameters  (https://ollama.com/library/llama2)",
    "llama:70b": "Llama 2 is released by Meta Platforms, Inc., featuring 70 billion parameters  (https://ollama.com/library/llama2)",
    "mistral:latest": "Mistral AI brings the strongest open generative models to the developers (https://mistral.ai/news/la-plateforme/)",
    "neural-chat": "A model specifically trained for engaging in chatbot conversations and responding to user queries in a conversational manner (https://ollama.com/library/neural-chat).",
    "wizardcoder": "A code generation model based on Code Llama (https://ollama.com/library/wizardcoder).",
    "codellama:latest": "A model for generating and discussing code, built on top of Llama 2 from Meta (https://ollama.com/library/codellama).",
    "codellama:13b": "Code generation model, boasting 13 billion parameters built on top of Llama 2 from Meta (https://ollama.com/library/codellama).",
    "codellama:34b": "Powerful code generation model, boasting 34 billion parameters built on top of Llama 2 from Meta (https://ollama.com/library/codellama)",
    "codellama:70b": "Very powerful code generation model, boasting 70 billion parameters built on top of Llama 2 from Meta (https://ollama.com/library/codellama).",
    "cas/discolm-mfto-german": "Experimental merge of pre-trained language models (https://ollama.com/cas/discolm-mfto-german).",
    }

       
    recommended_models = []
    if total_ram < 4:
        pass
    else: #4 <= total_ram < 7.9
        recommended_models = ["gemma:2b"]#, "wizard-math","llama-pro:latest"]  
    
    if total_ram >= 7.9:
        recommended_models.extend(["llama3:8b","gemma:7b","mistral:latest","cas/discolm-mfto-german","sauerkrautlm-7b-hero","llama2:latest","codellama:latest"])#, "wizard-math","llama-pro:latest"]  
    if total_ram >= 15.9:
        recommended_models.extend(["llama2:13b", "codellama:13b"])  
    if total_ram >= 60:
        recommended_models.extend(["llama:70b","codellama:34b", "codellama:70b" ])  # Larger models for 64GB+ RAM

    return recommended_models,models_dictionary 
 
#--------------------------------------------------------------------------------------------------
# Small donut chart  
def create_donut_chart(total_val, free_val,fig_size,font_size, plot_para):
    # Calculate used disk space
    used_val = total_val - free_val

    # Data to plot
    sizes = [free_val, used_val]
    labels = ['Free: '  + str(round(free_val, 2))+ " GB", 'Used: ' + str(round(used_val, 2))]
    colors = ['#D0F0C0', '#38bcf0']  # Light green for free space, white for used space
    
    # Plot
    plt.figure(figsize=(fig_size, fig_size))  
    plt.pie(sizes, labels=labels, colors=colors, startangle=90,textprops = {"fontsize": font_size})
   # plt.rcParams.update({'font.size': font_size})
    # Draw a white circle at the center to create a donut chart effect
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    # Add a text at the center showing total disk space
    plt.text(0, 0, f'{plot_para}\n Total:\n{round(total_val,2)} GB', ha='center', va='center', fontsize=font_size)
  
    return(fig)

# --------------------------------------------------------------------------------------------------------------
def knowledge_base_info():
    
    st.info("""
                Select option from the 'Knowledge base' menu on the sidebar.
            
                To create a knowledge base, select option  'Create knowledge base'.
                
                If you already have a knowledge base, select 'Read knowledge base' followed by the specific knowledge base name to start chatting.""")
    
    chat_info_basic_expander=st.expander("Create knowledge base - quick start",expanded=False)
    with chat_info_basic_expander:
        st.markdown("")
        # Creating a Knowledge Base
        st.markdown("**Create knowledge base - quick start:**")
        st.markdown("1. From the sidebar menu, navigate to **'Knowledge base'** and select **'Create knowledge base'**")
        st.markdown("2. Simply **drag and drop** your documents into the designated area.")
        st.markdown("3. Click the **'Create embeddings'** button to process your documents.")

        st.markdown("The app will automatically transform your data into a format suitable for large language models (LLMs). All processed documents are stored in a special database called a **'knowledge base'** located within the app's internal folder ('knowledge_bases').  This allows you to chat with your documents and access information easily.")

        # Chatting with an Existing Knowledge Base
        st.markdown("**If you already have a knowledge base:**")
        st.markdown("1. From the sidebar menu, navigate to **'Knowledge base'** and select **'Read knowledge base'**")
        st.markdown("2. Choose the specific **knowledge base name** you want to chat with from the dropdown menu (or list of options).")
        st.markdown("Once you've selected a knowledge base, you'll be able to start chatting with your documents!")
                                      
                      
    chat_info_advanced_expander=st.expander("Create knowledge base - for advanced users",expanded=False)
    with chat_info_advanced_expander:
        st.markdown("")
        st.markdown("""
        To chat with your own documents, large language models (LLMs) need to understand them first. 
        Regular documents like PDFs or Word files aren't directly interpretable. 
        Therefore, your documents need to be transformed into a format the LLM can understand and stored in a database accessible by LLMs. 
        """)    
        st.markdown("1. From the sidebar menu, navigate to **'Knowledge base'** and select **'Create knowledge base'**")
        st.markdown("2. Simply **drag and drop** your documents into the designated area.")
        
        st.markdown("3. Within the '**Knowledge base settings**' menu you have the flexibility to customize various settings for optimal results. This includes choosing the embedding model, splitter settings, and even checking the splitter performance. The splitter acts like a smart document separator, ensuring the LLM understands individual sections within your documents.") 
        st.markdown("4. When you are satisfied with the splitter performance click **'Create embeddings'** button to process your documents.")
        st.markdown("""            
       You can create separate knowledge bases for different document types. This can be beneficial for several reasons:

        **Improved Search Accuracy**: By grouping similar documents together, you can enhance the accuracy of search results within the knowledge base.
        
        **Enhanced Privacy**: If you have documents containing sensitive information, storing them in separate knowledge bases can provide an extra layer of control.
        
        **Better Organization**: Separating documents by category (e.g., legal documents, financial reports) makes it easier to navigate and find what you need quickly.

        All processed documents are stored in a special database called a 'knowledge base' located within the apps folder 'knowledge_bases'. This allows you to chat with your documents and access information easily.
            
                        
                        """)
        
#--------------------------------------------------------------------------------------------------
# Print chat history together with an with an avatar
def print_chat_history_timeline(chat_history_key):
    for message in st.session_state[chat_history_key]:
        role = message["role"]
        if role == "user":
            with st.chat_message("user", avatar=hs_avatar):                 
                st.markdown(message["content"], unsafe_allow_html=True)
        elif role == "assistant":
            with st.chat_message("assistant", avatar=ai_avatar):
                st.markdown(message["content"], unsafe_allow_html=True)

#--------------------------------------------------------------------------------------------------
# Main Ollama chat function
def llm_stream(model_name, messages): 
    '''
    #Messages general form for the Ollama call:
    messages=[{'role': 'user', 
               'content': 'Erzähl mir einen Witz',
               "options": { "seed": 42,
                            "temperature": 0.8 }}]
    '''
    response = ollama.chat(model_name, messages, stream=True)

    for chunk in response:
        yield chunk['message']['content']

#--------------------------------------------------------------------------------------------------
# Save chat history to a json file
def save_chat_to_json(file_prefix,llm_name, conversation_key):

    output_dir_name = "chat_history"
    output_dir = os.path.join(os.getcwd(), output_dir_name)

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{output_dir}/{file_prefix}_{timestamp}_{llm_name.replace(':', '-')}"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(f"{filename}.json", "w") as f:
        json.dump(st.session_state[conversation_key], f, indent=4)
        st.success(f"Conversation saved to {filename}.json")

#--------------------------------------------------------------------------------------------------
#Save chat  as a semicolon-separated txt file.
def save_chat_to_txt(file_prefix, llm_name, conversation_key):
 
    output_dir_name = "chat_history"
    output_dir = os.path.join(os.getcwd(), output_dir_name)

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{output_dir}/{file_prefix}_{timestamp}_{llm_name.replace(':', '-')}"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    chat_data = st.session_state[conversation_key]
    
    chat_data = st.session_state[conversation_key]
    
    # Convert JSON to semicolon-separated text
    with open(f"{filename}.txt", "w", encoding="utf-8") as f:
        for entry in chat_data:
            sender = entry.get('role', '')
            message = entry.get('content', '')
            f.write(f"{sender};{message}\n")
    
    st.success(f"Conversation saved to {filename}.txt")

#-----------------------------------------------------------------------------
# write embedding model name to the persist directory
def write_embedding_model_name(embeddings_model_name, persist_directory):
    model_name_file = os.path.join(persist_directory, "embedding_model.txt") 
    with open(model_name_file, "w") as f:
        f.write(embeddings_model_name)

#-----------------------------------------------------------------------------
# read model name from the persist directiry 
def read_embedding_model_name(persist_directory):
    model_name_file = os.path.join(persist_directory, "embedding_model.txt") 
    if os.path.exists(model_name_file):
        with open(model_name_file, "r") as f:
            try:
                return f.read().strip()
            except Exception as e:
                st.error(f"The persist directory seems to be corrupted. You should close the app and delete the directory  {persist_directory} manually from your PC.")
                st.stop()
        

#-----------------------------------------------------------------------------
# Langchain loader names

langchain_loader = {
    ".csv": (CSVLoader, {}), 
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),   
    ".html": (UnstructuredHTMLLoader, {}),    
    ".pdf": (PyPDFLoader, {}),
    #".ppt": (UnstructuredPowerPointLoader, {}),
    #".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),    
    }

#-----------------------------------------------------------------------------
# Load a document using the langchain loader    
def load_single_document(file_path): 
    
    ext = "." + file_path.rsplit(".", 1)[-1]
    
    if ext in langchain_loader:
        loader_class, loader_args = langchain_loader[ext]
        loader = loader_class(file_path, **loader_args)
        
        return loader.load()

    raise ValueError(f"Unsupported file extension '{ext}'")

#-----------------------------------------------------------------------------
# Loads mutiple documents from the directory
def load_documents(source_dir):
  """
  Loads documents from the source directory sequentially.

  Args:
      source_dir: Path to the directory containing the documents.

  Returns:
      A list of loaded documents.
  """
  
  all_files = []
  
  for ext in langchain_loader:
    all_files.extend(
      glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
    )
  
  filtered_files = [file_path for file_path in all_files]
  results = []
  
  # Sequential processing
  for file_path in filtered_files:  

    document = load_single_document(file_path)  # Call load_single_document for each file path sequentially
    results.extend(document)
  
  return results


#------------------------------------------------------
# Loads mutiple documents from the directory
# doesn't work in the exe version due to parallel processing!
def load_documents_ini(source_dir):
    
    all_files = []
        
    for ext in langchain_loader:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )
    
    filtered_files = [file_path for file_path in all_files]
    
    # Parallel processing
    with Pool(processes=os.cpu_count()) as pool:
        results = []
        
        with tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) as pbar:
           
            for i, docs in enumerate(pool.imap_unordered(load_single_document, filtered_files)):    # Call load_single_document for each file path in parallel
                                
                results.extend(docs)
                pbar.update()
    
    return results

#-----------------------------------------------------------------------------
# file selector for the major app directory
def file_selector():
    folder_path = os.path.join(os.getcwd(), 'knowledge_bases') 
    
    
    filenames = os.listdir(folder_path)
    folder_names = [filename for filename in filenames 
                    if os.path.isdir(os.path.join(folder_path, filename)) and filename not in [ ".venv", ".streamlit","default_data","temp_data","__pycache__"]]
    selected_filename = st.selectbox('Select a folder', 
                                     folder_names,
                                     help="Select a folder with your knowledge base",
                                     key='selected_knowledge_base')

    return os.path.join(folder_path, selected_filename)

#--------------------------------------------------------------------------------------------------
# Load documents and split in chunks
def experimental_process_documents():
    source_directory = st.session_state['source_directory']
    chunk_size = st.session_state['chunk_size']
    chunk_overlap = st.session_state['chunk_overlap']
    separators=st.session_state['separators']    
    is_separator_regex=st.session_state['is_separator_regex']
    
    documents = load_documents(source_directory)   
  
    #print(f"Loaded {len(documents)} new documents from {source_directory}")
       
    #text splitter 
    if st.session_state['splitter_function']=='char_text_splitter':
        text_splitter = CharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap,
            separator=separators, 
            is_separator_regex=is_separator_regex,
            length_function=len)  
    else:
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap,
            separators=separators, 
            is_separator_regex=is_separator_regex,
            length_function=len)  
    
    #list of text chunks
    texts = text_splitter.split_documents(documents)
        
    return texts    

#-----------------------------------------------------------------------------
# Load documents and split in chunks
def process_documents():
    source_directory = st.session_state['source_directory']
    chunk_size = st.session_state['chunk_size']
    chunk_overlap = st.session_state['chunk_overlap']
    separators=st.session_state['separators']
    

    is_separator_regex=st.session_state['is_separator_regex']

    print(f"Loading documents from {source_directory}")
    documents = load_documents(source_directory)
    if not documents:
        print("No documents to load")
        exit(0)
    #print(f"Loaded {len(documents)} new documents from {source_directory}")
    
    

    #text splitter     
    if st.session_state['splitter_function']=='char_text_splitter':
        text_splitter = CharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap,
            separator=separators, 
            is_separator_regex=is_separator_regex,
            length_function=len)  
    else:
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap,
            separators=separators, 
            is_separator_regex=is_separator_regex,
            length_function=len)  
    
    #list of text chunks
    texts = text_splitter.split_documents(documents)
    st.success(f"Your data are split into {len(texts)} chunks of text")
    return texts    


#----------------------------------------------------------------------------------------------
# check if Chroma for the filename exists
def chroma_exists(persist_directory: str) -> bool:
 
  return os.path.exists(persist_directory)

#------------------------------------------------------------------------------------------------------
#reset key 'knowledge_base_selection' after embedding
def chat_after_embedding():    
    st.session_state['knowledge_base_selection']= "Read knowledge base"
    st.session_state['selected_knowledge_base']=os.path.basename(os.path.normpath(st.session_state['persist_directory']))
    
    #delete temp_dir if not already deleted!
    temp_dir = os.path.join(os.getcwd(), "temp_data")  
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir) 

#-----------------------------------------------------------------------------------------------------------
#check if a file already exists in a default knowledge base 'knowledge_base'
def check_for_duplicates(persist_directory,embeddings_model_name):
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings) 

    # get metadata from Chroma    
    data=db.get()["metadatas"]   
    sources = [item.get("source") for item in data if isinstance(item, dict)]

    # Get unique sources in the knowledge base
    unique_sources = set(sources)
    dict_filenames = [os.path.basename(key) for key in unique_sources]
        
    # check the content of the temp directory
    temp_dir = os.path.join(os.getcwd(), "temp_data") 
    temp_dir_content=os.listdir(temp_dir)
    
    common_elements = [element for element in dict_filenames if element in [item for item in temp_dir_content]]
    
    if len(common_elements)>0:
        st.error(f"You already have {common_elements} in your knowledge base! Please upload new files!")
        st.stop()


#-----------------------------------------------------------------------------------------------------
def create_embeddings(embedings_placeholder,embeddings_model_name,temp_dir):
    
    default_pd=os.path.join('knowledge_bases','knowledge_base')
    embedings_info=embedings_placeholder
    
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    persist_directory=st.session_state['persist_directory']
    
    # Update knowledge base at 'knowledge_bases/knowledge_base':
    if persist_directory==default_pd:   
        embedings_info.info(f"Preparing a knowledge base at {persist_directory}")
        if os.path.exists(persist_directory):
            # Check for duplicates at 'knowledge_bases/knowledge_base' 
            check_for_duplicates(persist_directory,embeddings_model_name)
            
            # Create a knowledge database            
            db = Chroma(persist_directory=persist_directory, embedding_function=embeddings) 
            texts = process_documents()
            embedings_info.info("Creating embeddings for your data...Please wait...")
            db.add_documents(texts)
            write_embedding_model_name(embeddings_model_name, persist_directory)
        else:
            # Create a knowledge database            
            texts = process_documents()
            db = Chroma.from_documents(
                texts, 
                embeddings, 
                persist_directory=persist_directory)  
            embedings_info.info("Creating embeddings for your data...Please wait...")      
            write_embedding_model_name(embeddings_model_name, persist_directory)    
    
    # Delete existing Chroma and recreate with new data    
    elif chroma_exists(persist_directory) and persist_directory != default_pd:        
        embedings_info.info(f"Deleting existing knowledge base at {persist_directory} and creating a new one...")
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)        
        db.delete_collection()  

        # Create a knowledge database
        texts = process_documents()
        db = Chroma.from_documents(
            texts, 
            embeddings, 
            persist_directory=persist_directory)        
        embedings_info.info("Creating embeddings for your data...Please wait...")
        write_embedding_model_name(embeddings_model_name, persist_directory)

    # Create a new knowledge base
    else:        
        embedings_info.info("Creating a new knowledge database...") 
    
        # Create a knowledge database          
        texts = process_documents()
        db = Chroma.from_documents(
            texts, 
            embeddings, 
            persist_directory=persist_directory)
        embedings_info.info("Creating embeddings for your data...Please wait...")
        write_embedding_model_name(embeddings_model_name, persist_directory)
    
    embedings_info.success("Knowledge base complete!")
    
    # Delete the temporary directory and its contents                    
    shutil.rmtree(temp_dir)  

    st.markdown("")   
  


#--------------------------------------------------------------------------------------------------------
# retriever settings sidebar

def retriever_settings_sidebar():
    retriever_settings=st.sidebar.expander('Retriever_settings', expanded=False)
    with retriever_settings:
        target_source_chunks=st.number_input('Specify number of relevant chunks to return'
                                             ,min_value=2,
                                             max_value=6,
                                             value=4, 
                                             step=1,
                                             help="This parameter controls the number of chunks (pieces of text) from the target source that will be used in the creating the response during the RAG retrieval process.")
        st.session_state['target_source_chunks']=target_source_chunks
        return_source_documents=st.toggle('Return source documents',
                                          value=False,
                                          help="Argument on whether or not to return metadata in the responses"
                                          )
        st.session_state['return_source_documents']=return_source_documents




#--------------------------------------------------------------------------------------------------
# Extracts the longest word in the text as a placeholder for the chat name.
def get_chat_name_placeholder(text):
    
    words = text.split()
    if not words:
        return "Unknown"  # Handle empty text
    longest_word = max(words, key=len)
    return longest_word


#--------------------------------------------------------------------------------------------------
#RAG retriever
def handle_data_chat(prompt, model, model_temp,system_promt,persist_directory,embeddings_model_name,target_source_chunks,return_source_documents):            
       
    # Build prompt    
    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible.  
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
       
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    llm=Ollama(model=model,temperature=model_temp)
    
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)     
    
    qa = RetrievalQA.from_chain_type(
        llm,
        retriever=db.as_retriever(search_kwargs={"k": target_source_chunks}),
        return_source_documents=return_source_documents,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    
    #get the response from LLM
    result=qa({"query": prompt + ' ' + system_promt})
    answer = result["result"]   
    
   
    # stream response    
    for chunk in answer:
        yield chunk#['message']['content']
   

    #response source
    if return_source_documents:    
        #source_metadata = [doc.metadata["source"] for doc in result["source_documents"]]        
        #st.warning(f"Source for the {target_source_chunks} chunks used in creating the respone: {source_metadata}")   
        st.warning("Source for the  chunks used in creating the respone:")
        st.write(result["source_documents"])   
    return 

import time

# typewriter effect for the home page
def typewriter(text: str, speed: int):    

    tokens = text.split()
    container = st.empty()

    for index in range(len(tokens) + 1):
        curr_full_text = " ".join(tokens[:index])
        #markdown_text = f"<span style='color:#38bcf0; font-size: 20px; font-weight: 650;'> {curr_full_text} </span>"  # Inline style within markdown
        markdown_text = curr_full_text
        container.markdown(markdown_text, unsafe_allow_html=True)
        time.sleep(1 / speed)

        