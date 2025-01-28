import streamlit as st
import pandas as pd
import llm_functions as lfc
import ollama



def app():
    #avatar logos
    ai_avatar='default_data/logo.png'
    hs_avatar = 'default_data/hs_logo1.png'
    st.session_state['image_key']=0
        

    if 'promt' not in st.session_state:
        st.session_state['promt']="Chat with STATY.AI"


    # settings sidebar
    (sett_theme,output_format)=lfc.settings_sidebar()

    #model selection sidebar
    (menu_option, available_ollama_models,llm_name)=lfc.select_model_sidebar() 

    #st.write("Computer Vision Made Easy: Load, Analyze, and Discover")
    st.markdown('<font color="#38bcf0"><font color="#38bcf0">**Computer Vision Made Easy | Load, Analyze, and Discover** </font>', unsafe_allow_html=True)
    st.markdown("Consider employing vision models such as <a href='https://ollama.com/library/llama3.2-vision' style='color:#38bcf0'>llama3.2-vision</a>.", unsafe_allow_html=True) 
    a4,a5=st.columns(2)
    with a4:
        uploaded_file = st.file_uploader(
                "Choose an image...", type=["jpg", "jpeg", "png", "webp"]
                
            )
        
        if llm_name =="": st.warning("Select vision LLM model from the 'Choose model' menu on the sidebar.")

           
        
    model = llm_name

    if uploaded_file is not None and model:
        image, encoded_image = lfc.process_image(uploaded_file)
        
        if st.checkbox("Show uploaded image",value=False): st.image(image, caption="Uploaded Image", use_column_width=True)

        #new_response = st.button("Generate Response")
        #if new_response:
        #    generate_response(encoded_image, model)


        # conversation key  - model & menu_option
        conversation_key = f"model_{llm_name}"+menu_option.replace(" ", "")+uploaded_file.name.replace(" ", "")
        
        (model_temp,system_promt)=lfc.model_settings(available_ollama_models,llm_name)        
        
        prompt = st.chat_input(f"Ask '{llm_name}' a question ...")

        
        if conversation_key not in st.session_state.keys():
            st.session_state[conversation_key] = []

        lfc.print_chat_history_timeline(conversation_key)
        
        # run the model
        if prompt:
            st.session_state['promt']=prompt
            st.session_state[conversation_key].append({
                "content": f"{prompt}",#+ system_promt, 
                "role": "user",                
                "options": {
                    "seed": 42,
                    "temperature":model_temp
                }}
                )
            with st.chat_message("question", avatar=hs_avatar):
                st.markdown(prompt)

            messages=[
                    {
                        "role": "user",
                        "content": f"{prompt}",
                        "images": [encoded_image],
                    }
                ]

            



            # streaming response
            with st.chat_message("response", avatar=ai_avatar):
                #with st.spinner("Generating answer..."):                    
                response_message = st.write_stream(lfc.llm_stream_image(llm_name, messages))

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
            
            
  

        
