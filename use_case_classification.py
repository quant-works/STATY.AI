import streamlit as st
import pandas as pd
import llm_functions as lfc
import os
import shutil
import ollama

def app():
    
  

    #----------------------------------------------------------------
    # Classify with LLMs 
    #----------------------------------------------------------------  
    
    #avatar logos
    ai_avatar='default_data/logo.png'
    hs_avatar = 'default_data/hs_logo1.png'

    if 'promt' not in st.session_state:
        st.session_state['promt']="Chat with STATY.AI"

    # settings sidebar
    (sett_theme,output_format)=lfc.settings_sidebar()

    #model selection sidebar
    (menu_option, available_ollama_models,llm_name)=lfc.select_model_sidebar() 
                    
    # model paramater settings sidebar       
    if llm_name:
        # conversation key  - model & menu_option
        conversation_key = f"model_{llm_name}"+menu_option.replace(" ", "")
        
        (model_temp,system_promt)=lfc.model_settings(available_ollama_models,llm_name)
        if conversation_key not in st.session_state.keys():
            st.session_state[conversation_key] = []
        
        #------------------------------------------------------
        # Settings of the classifier
        #------------------------------------------------------
        basic_messages = []


        role_prompt = st.text_area(
        "Role Prompt (optional)",
        value="e.g., 'You are an academic reviewer...'"
        )
        if len(role_prompt) > 0:
            basic_messages.append({"role": "system", "content": role_prompt})

        task = st.text_area(
        "Task",
        value="e.g., 'Evaluate each document...'"
        )
        if len(task) > 0:
            basic_messages.append({"role": "user", "content": task})
        else:
            st.error("The task can't be empty")
            exit()
        
        examples = st.text_area(
        "Examples (optional)",
        placeholder="e.g., 'Good: clear intro…; Bad: vague intro…'"
        )
        if len(examples) > 0:
            basic_messages.append({"role": "user", "content": examples})


        reasoning = st.text_area(
        "Reasoning Style (optional)",
        placeholder="e.g., 'Provide a brief verdict only.'"
        )
        if len(reasoning) > 0:
            basic_messages.append({"role": "system", "content": reasoning})

              
                
       #------------------------------------------------------
        #Upload your data
        upload_container=st.container(border=True)
        upload_container.markdown("**Upload your data**")
        uploaded_data = upload_container.file_uploader("File upload", 
                                #type=[".pdf"],accept_multiple_files=True,on_change=change_check_splitting_state)
                                type=[".docx",".pdf",".jpg",".jpeg"],accept_multiple_files=True)

        if uploaded_data:   
            #delete temp_dir if exists!
            temp_dir = os.path.join(os.getcwd(), "temp_data")  
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)            
            
            temp_dir = os.path.join(os.getcwd(), "temp_data") 
            if not os.path.exists(temp_dir):                    
                os.makedirs(temp_dir)

                            
            #------------------------------------------
                
                
                
                
            run_button=st.button("Run")
            
            # run the model
            if run_button:
                for file_obj in uploaded_data:
                    file_data = file_obj.read()              

                    # Save to 'temp_data'
                    file_path_temp_dir = os.path.join(temp_dir, file_obj.name)
                    with open(file_path_temp_dir, 'wb') as frsc:
                        frsc.write(file_data)
        
                    #Read the document:
                    filename_lower = file_obj.name.lower()
                    if filename_lower.endswith((".jpg", ".jpeg")):
                        image, encoded_image = lfc.process_image(file_obj)
                        messages = basic_messages.copy()                       
                            
                        messages.append({"role": "user", "content": task, "images": [encoded_image]}) 
                       
                    else:
    
                        content=lfc.load_single_document(file_path_temp_dir)
                        if len(content[0].page_content) > 0:
                            messages = basic_messages.copy()                       
                            
                            messages.append({"role": "user", "content": content[0].page_content})  
        
                  
                    
                                   
                    # streaming response
                    with st.chat_message("response", avatar=ai_avatar):
                        st.markdown( file_obj.name)
                        #st.markdown(os.path.basename(content[0].metadata['source']))
                        #with st.spinner("Generating answer..."):                    
                        response_message = st.write_stream(lfc.llm_stream(llm_name, messages))

                        st.session_state[conversation_key].append({
                            #"content": f"{os.path.basename(content[0].metadata['source'])}\n {response_message}", 
                            "content": f"{ file_obj.name}\n {response_message}", 
                            "role": "assistant",
                            "options": {
                                "seed": 42,
                                "temperature": model_temp
                            }})
                    
                  
                  
            if st.session_state[conversation_key]:            
                # Give a prefix to chat-file name
                file_prefix=st.sidebar.text_input("Give a chat some name", value="Docs classification")
                
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
                
            