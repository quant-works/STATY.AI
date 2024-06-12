import streamlit as st
import os
import llm_functions as lfc
import re
import shutil

def app():
    
    #----------------------------------------------------------------
    # Chat with your data
    #----------------------------------------------------------------  
    
    #avatar logos
    ai_avatar='default_data/logo.png'
    hs_avatar = 'default_data/hs_logo1.png'

    # settings sidebar
    (sett_theme,output_format)=lfc.settings_sidebar()

    #model selection sidebar
    (menu_option, available_ollama_models,llm_name)=lfc.select_model_sidebar() 
        
    
    #------------------------------------------------------------------------------------------
    # Set environmental variables
    #------------------------------------------------------------------------------------------
    data_chat=False
    
    if 'persist_directory' not in st.session_state:
        st.session_state['persist_directory']=os.path.join('knowledge_bases','knowledge_base' )      
    if 'source_directory' not in st.session_state:    
        st.session_state['source_directory']='temp_data'
    if 'embeddings_model_name'not in st.session_state:
        st.session_state['embeddings_model_name']='all-MiniLM-L6-v2'
    if 'chunk_size' not in st.session_state:
        st.session_state['chunk_size']=500
    if 'chunk_overlap' not in st.session_state:    
        st.session_state['chunk_overlap']=50
    if 'target_source_chunks' not in st.session_state:    
        st.session_state['target_source_chunks']=3
    if 'return_source_documents' not in st.session_state: 
        st.session_state['return_source_documents']=False
    if 'promt' not in st.session_state:
        st.session_state['promt']="Chat with STATY.AI"
    if 'is_separator_regex' not in st.session_state:
        st.session_state['is_separator_regex']=False
    if 'separators' not in st.session_state:
        st.session_state['separators']=["\n\n", "\n", " ", ""]
    if 'splitter_function' not in st.session_state:
        st.session_state['splitter_function']="RecursiveCharacterTextSplitter"    
    if 'length_function' not in st.session_state:
        st.session_state['length_function']=None 
    if 'experimental_splitting_done' not in st.session_state:
        st.session_state['experimental_splitting_done']=False   
    if 'text' not in st.session_state:
        st.session_state['text']=None      
    st.session_state['start_chatting']=False
    
    
    def change_check_splitting_state():
        st.session_state['check_splitting']=False
        st.session_state['experimental_splitting_done']=False
        st.session_state['text'] =None

    #st.write(st.session_state['persist_directory'])
    # model paramater settings sidebar       
    if llm_name:
        
        st.session_state['MODEL']=llm_name
        model = llm_name
        (model_temp,system_promt)=lfc.model_settings(available_ollama_models,llm_name)

        #Sidebar 'Knowledge base'
        with st.sidebar.expander("Knowledge base"):     
            load_data=st.selectbox('Select option',['',"Create knowledge base", "Read knowledge base"], help="To create a new '**knowledge base**' from your files, select 'Create..'. To read existing one, select 'Read..'", key='knowledge_base_selection')
            #read Knowledge base (sidebar)
            if load_data=="Read knowledge base":  
                try:
                    persist_directory= os.path.basename(lfc.file_selector())
                    persist_directory=os.path.join('knowledge_bases',persist_directory )  
                    st.session_state['persist_directory']=persist_directory
                    data_chat=True 
                except Exception as e:
                    st.error("Oops! No knowledge base found!")    
        
        # Display help info only if "Read knowledge base" is not selected
        if load_data !="Read knowledge base":
            lfc.knowledge_base_info()

        # Creating a new knowledge base (main window!)
        if load_data=="Create knowledge base":        

            #Upload your data
            upload_container=st.container(border=True)
            upload_container.markdown("**Upload your data**")
            uploaded_data = upload_container.file_uploader("File upload", 
                                    type=[".docx",".pdf", "txt" , "csv",".html"],accept_multiple_files=True,on_change=change_check_splitting_state)
    
            #delete temp_dir if exists!
            temp_dir = os.path.join(os.getcwd(), "temp_data")  
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir) 

            
            if uploaded_data:  
                # save files temporarily to 'temp_dir' and delete it afterwards                                   
                temp_dir = os.path.join(os.getcwd(), "temp_data")  
                if not os.path.exists(temp_dir):
                    os.makedirs(temp_dir)
                for file_obj in uploaded_data:
                    file_path = os.path.join(temp_dir, file_obj.name)
                    with open(file_path, 'wb') as f:
                        f.write(file_obj.read())
                
                #---------------------------------------------------------------------------------------
                #Knowledge base settings expander
                #---------------------------------------------------------------------------------------
                with st.expander("Knowledge base settings",expanded=False):             
                    knowledge_base_directory=st.toggle("Use default folder ('knowledge_base') for RAG?",
                                                       value=False,
                                                       help="If you select no, then a 'knowledge base' will be located in a new folder having the same name as your file",
                                                       on_change=change_check_splitting_state)
                   
                    kb_container = st.container(border=True)
                    kb_container.markdown('**Separator base settings**')
                    
                    chunk_size=kb_container.number_input("Specify chunk size", 
                                                         min_value=1, 
                                                         max_value=4000,
                                                         step=10,value=1500, 
                                                         help="Larger chunk size processes data faster, but might miss context between sentences/paragraphs. Smaller chunk size preserves context better, but requires more processing power.",
                                                         on_change=change_check_splitting_state)
                    st.session_state['chunk_size']=chunk_size
                    
                    chunk_overlap_perc=kb_container.number_input("Specify chunk overlap in percent of chunk size",
                                                                  min_value=0, 
                                                                  max_value=20,
                                                                  step=1,
                                                                  value=10, 
                                                                  help="Higher overlap prioritizes context, while lower overlap prioritizes speed.",
                                                                  on_change=change_check_splitting_state)
                    chunk_overlap = round(chunk_overlap_perc*chunk_size/100,0)
                    st.session_state['chunk_overlap']=chunk_overlap
                    splitter_functions = {
                        "None": "RecursiveCharacterTextSplitter",
                        "RecursiveCharacterTextSplitter": 'RecursiveCharacterTextSplitter',                         
                        "CharacterTextSplitter": 'char_text_splitter',                                        
                    }
                    
                    splitter_function=kb_container.selectbox("Select the splitter function", 
                                                             options=splitter_functions,
                                                             help="For more info on text splitters please see https://python.langchain.com/docs/modules/data_connection/document_transformers/",
                                                             on_change=change_check_splitting_state)
                    st.session_state['splitter_function']=splitter_functions[splitter_function]

                    if st.session_state['splitter_function']=='char_text_splitter':
                        user_separators=kb_container.text_input("Enter a single text separator (if regex, then select below regex=true)",
                                                                placeholder= 'split_here',
                                                                on_change=change_check_splitting_state)
                    else:    
                        user_separators=kb_container.text_input("Enter comma separated text separators  (if regex, then select below regex=true)",
                                                                value= ["\n\n", "\n", " ", ""],
                                                                help="The default list of separators has the effect of trying to keep all paragraphs (and then sentences, and then words) together as long as possible, as those would generically seem to be the strongest semantically related pieces of text.",
                                                                on_change=change_check_splitting_state)
                    
                    if user_separators:                        
                        st.session_state['separators']=user_separators                             
                                        
                    is_separator_regex = kb_container.toggle("Regex separator?", 
                                                             value=False, 
                                                             help="See https://docs.python.org/3/howto/regex.html for more info on how to use Regex",
                                                             on_change=change_check_splitting_state)
                    st.session_state['is_separator_regex']=is_separator_regex
                    check_splitting=kb_container.toggle("Check the document splits?",
                                                        value=False,
                                                        key='check_splitting')
                   
                     
                    if check_splitting: 
                        #st.write(f"if check splitting {st.session_state['experimental_splitting_done']}")  
                        #st.write(f"txt if check splitting {st.session_state['text']}")                          
                        if st.session_state['experimental_splitting_done']==False:                      
                            with st.spinner("Document splitting..."):  
                                text=lfc.experimental_process_documents()   
                                st.session_state['experimental_splitting_done']=True                         
                                st.session_state['text'] =text   
                        
                        #st.write(f"after if check splitting {st.session_state['experimental_splitting_done']}") 
                        #st.write(f"txt after if check splitting {st.session_state['text']}")       
                        if st.session_state['text'] is not None:
                            text =st.session_state['text']
                           
                            chunks_container = st.container(border=True)
                            chunks_container.markdown('**Document Chunks**')
                            chunks_container.markdown(f"Number of chunks: {len(text)}")
                            chunk_number = chunks_container.selectbox("Select Chunk", list(range(1, len(text) + 1)))

                            # Display the selected chunk's text
                            chunks_container.markdown("Chunk Content:")
                            chunks_container.markdown(text[chunk_number - 1])  

                    eb_container = st.container(border=True)
                    eb_container.markdown('**Embeddings and output settings**')
                    if knowledge_base_directory:# use default knowledge_base
                        default_pd=os.path.join('knowledge_bases','knowledge_base')  
                        model_name_file = os.path.join(default_pd, "embedding_model.txt") 
                        if os.path.exists(model_name_file):
                            embeddings_model_name=lfc.read_embedding_model_name(default_pd)
                            eb_container.markdown(f"You selected embedding model  '{embeddings_model_name}' for the default knowldege base!  \n\n See model details at https://www.sbert.net/docs/pretrained_models.html")
                            st.session_state['embeddings_model_name']= embeddings_model_name
                        else:
                            embeddings_model_name = eb_container.selectbox("Select embedings model",['all-MiniLM-L6-v2','multi-qa-MiniLM-L6-cos-v1','all-MiniLM-L12-v2','distiluse-base-multilingual-cased-v1','paraphrase-multilingual-MiniLM-L12-v2'],help='See https://www.sbert.net/docs/pretrained_models.html for more info')
                            st.session_state['embeddings_model_name']= embeddings_model_name
                    else:
                        embeddings_model_name = eb_container.selectbox("Select embedings model",['all-MiniLM-L6-v2','multi-qa-MiniLM-L6-cos-v1','all-MiniLM-L12-v2','distiluse-base-multilingual-cased-v1','paraphrase-multilingual-MiniLM-L12-v2'],help='See https://www.sbert.net/docs/pretrained_models.html for more info')
                        st.session_state['embeddings_model_name']= embeddings_model_name
                            
                    
                    
                    if knowledge_base_directory==True:
                        st.session_state['persist_directory'] = os.path.join('knowledge_bases','knowledge_base')   
                    else:
                        first_file_info=uploaded_data[0]                                              
                        knowledge_base_name =  os.path.join('knowledge_bases',re.sub(r'\s+', '_', first_file_info.name.split('.')[0]))
                        #set the knowledge_base directory to the file name withouth empty spaces!
                        st.session_state['persist_directory']=knowledge_base_name
                
                #--------------------------------------------------------------------------------------------
                        
                st.markdown("")
                embedings_placeholder=st.empty()
                start_embedings=embedings_placeholder.button("Create embeddings")
                
                if start_embedings: 
                    lfc.create_embeddings(embedings_placeholder,embeddings_model_name,temp_dir)
                    
                    st.markdown("")
                    st.button("Chat with your data", on_click=lfc.chat_after_embedding)
                    
       
        
        #----------------------------------------------------------------------------------------------------
        #Chat with your data
        
        # conversation key  - model & menu_option
        persist_directory = st.session_state['persist_directory']
        conversation_key = f"model_{llm_name}"+menu_option.replace(" ", "")+persist_directory
        
        if data_chat:        
            
             #delete temp_dir if not already deleted!
            temp_dir = os.path.join(os.getcwd(), "temp_data")  
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir) 

            lfc.retriever_settings_sidebar()

            model = st.session_state['MODEL']            
            target_source_chunks = int(st.session_state['target_source_chunks'])  
            persist_directory = st.session_state['persist_directory']
            return_source_documents=st.session_state['return_source_documents']
            embeddings_model_name = lfc.read_embedding_model_name(persist_directory)   
                           
                      
            prompt = st.chat_input(f"Ask '{llm_name}' a question ...")

            
            if conversation_key not in st.session_state.keys():
                st.session_state[conversation_key] = []

            lfc.print_chat_history_timeline(conversation_key)
                
            # run the model
            if prompt:
                st.session_state['promt']=prompt
                st.session_state[conversation_key].append({
                    "content": f"{prompt}", 
                    "role": "user",
                    "options": {
                        "seed": 42,
                        "temperature":model_temp
                    }})
                with st.chat_message("question", avatar=hs_avatar):
                    st.write(prompt)

                messages = [dict(
                    content=message["content"], 
                    role=message["role"], 
                    options=message["options"]) 
                        for message in st.session_state[conversation_key]]
                    
           
                # streaming response
                with st.chat_message("response", avatar=ai_avatar):                          
                
                    response_message = st.write_stream( lfc.handle_data_chat(prompt, model, model_temp,system_promt,persist_directory,embeddings_model_name,target_source_chunks,return_source_documents))
               
                

                st.session_state[conversation_key].append({
                    "content": f"{response_message}", 
                    "role": "assistant",
                    "options": {
                        "seed": 42,
                        "temperature": model_temp
                    }})
                    
            
            if st.session_state[conversation_key]:            
                # Give a prefix to chat-file name
                file_prefix=st.sidebar.text_input("Give a chat some name", value=lfc.get_chat_name_placeholder(st.session_state['promt']),max_chars=20)
                
                # save conversation to file 
                a4,a5=st.sidebar.columns(2)  
                with a4:         
                    save_chat =st.button("Save chat",help="Your conversation will be saved to the folder: 'chat_history'")

                with a5:
                    #Clear chat 
                    clear_chat = st.button("Clear chat")
                    if clear_chat:
                        st.session_state[conversation_key] = []
                        st.rerun()                         
                    
                if save_chat:
                    if file_prefix:
                        try:
                            if output_format==".txt":                    
                                lfc.save_chat_to_txt(file_prefix,llm_name, conversation_key)                    
                            elif output_format==".json":                    
                                lfc.save_chat_to_json(file_prefix,llm_name, conversation_key)
                        except Exception as e:
                            st.sidebar.error(f"An error occurred while saving conversation: {str(e)}")
                    else:
                        st.sidebar.error("Please give a name to your chat")


                        
            
         

            


