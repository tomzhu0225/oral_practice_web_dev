# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 13:43:24 2023

@author: tomkeen
"""

import streamlit as st
import os
import azure.cognitiveservices.speech as speechsdk
from azure.cognitiveservices.speech.audio import AudioOutputConfig
import base64
import openai
from streamlit_webrtc import WebRtcMode, webrtc_streamer
import pydub
import logging
from pathlib import Path
import queue
import urllib.request
import numpy as np
import time

HERE = Path(__file__).parent

logger = logging.getLogger(__name__)
# inject openai api key
openai.api_key = st.secrets["openaikey"]

def recognize_from_mic(lang):
    audio_file = open("output.mp3", "rb")
    transcript = openai.Audio.transcribe(model="whisper-1",
                                         file=audio_file,language=lang)    
        
    return transcript['text']
def autoplay_audio():
    with open('tts.mp3', "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(
            md,
            unsafe_allow_html=True,
        )
def synthesize_to_speaker(text,lang,azureapi):
	#Find your key and resource region under the 'Keys and Endpoint' tab in your Speech resource in Azure Portal
	#Remember to delete the brackets <> when pasting your key and region!
    lang=lang_convertor(lang)
    speech_config = speechsdk.SpeechConfig(subscription=azureapi, region="francecentral")
    speech_config.speech_synthesis_language = lang
    #In this sample we are using the default speaker 
    #Learn how to customize your speaker using SSML in Azure Cognitive Services Speech documentation
    speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3)  
  
    file_name = "tts.mp3"  
    file_config = speechsdk.audio.AudioOutputConfig(filename=file_name)  
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=file_config)  
    
    result=synthesizer.speak_text_async(text).get()

def respond(conversation):
    response=openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=conversation
    )
    return response.choices[0]['message']['content']

def suggestion(conversation):
    response=openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=conversation
    )
    return response.choices[0]['message']['content']
def concatenate_me(original,new):
    original=original+[{"role": "user", "content": new}]
    return original
def concatenate_you(original,new):
    original=original+[{"role": "assistant", "content": new}]
    return original
def init():
    #initialization of state variable
    if 'count' not in st.session_state:
        st.session_state['count'] = 0
    Me_temp='ME'+str(0)
    if  Me_temp not in st.session_state:
        st.session_state[Me_temp]=''
    if 'conv' not in st.session_state:
        st.session_state['conv'] = []
    You_temp='YOU'+str(0)
    if You_temp not in st.session_state:
        st.session_state[You_temp]=''
    if 'sugg' not in st.session_state:
        st.session_state['sugg'] = ''
def lang_convertor(lang):
    ls=["en-US","zh-CN", "fr-FR",'es-ES','ko-KR',"ja-JP", "it-IT", "pt-PT", "ru-RU"]
    if lang=='en':
        return ls[0]
    if lang=='zh':
        return ls[1]
    if lang=='fr':
        return ls[2]
    if lang=='de':
        return 'de-DE'
    if lang=='es':
        return ls[3]
    if lang=='ko':
        return ls[4]
    if lang=='ja':
        return ls[5]
    if lang=='it':
        return ls[6]
    if lang=='pt':
        return ls[7]
    if lang=='ru':
        return ls[8]
def main():
    global lang_mode
    global text_output
    global Preset
    global respond_mod
    global sugg_mod
    global rtc
    serverlist=[ "google1","test(中国)","google2","google3","google4",  
                'google5',
    'stun.voipbuster.com',  
    'stun.sipgate.net',  
    'stun.ekiga.net',
    'stun.ideasip.com',
    'stun.schlund.de',
    'stun.voiparound.com',
    'stun.voipbuster.com',
    'stun.voipstunt.com',
    'stun.counterpath.com',
    'stun.1und1.de',
    'stun.gmx.net',
    'stun.callwithus.com',
    'stun.counterpath.net',
    'stun.internetcalls.com',
    'numb.viagenie.ca']
    #initialization of state variable
    init()
        
    #stun server

    st.header("Oral practice with AI")


    html_temp = """
                    <div style="background-color:{};padding:1px">
                    
                    </div>
                    """
    left, right = st.columns(2)
    with left: 
        lang_mode = st.selectbox("Choose your language", ["en","zh", "fr",'de', 'es','ko',"ja", "it", "pt", "ru"],key='lang')

     
    with right:
        stun_mode = st.selectbox("Choose the stun server", serverlist,key='stun')
        
        if stun_mode=='test(中国)':
            rtc={"iceServers": [{"urls": ["turn:60.205.206.155:3487"]}]}
        elif stun_mode=="google1":
            rtc={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        elif stun_mode=="google2":
            rtc={"iceServers": [{"urls": ["stun:stun1.l.google.com:19302"]}]}
        elif stun_mode=="google3":
            rtc={"iceServers": [{"urls": ["stun:stun2.l.google.com:19302"]}]}
        elif stun_mode=="google4":
            rtc={"iceServers": [{"urls": ["stun:stun3.l.google.com:19302"]}]}
        elif stun_mode=="google5":
            rtc={"iceServers": [{"urls": ["stun:stun4.l.google.com:19302"]}]}
        for i in range(6, len(serverlist)):
            if stun_mode==serverlist[i]:
                rtc={"iceServers": [{"urls": ["stun:"+serverlist[i]+":3487"]}]}
            
    
    Preset = st.text_input('Preset', placeholder="Enter the scene setting")

    
    
    
    
    
      
    
    with st.sidebar:
        st.markdown("""
        # About 
        This page is providing a new way of practice your oral with openai!
        If you like the app, 
        
        # [Donate me!](https://drawingsword.com/post/donate-me/)
        
        please star source code:[github](https://github.com/tomzhu0225/oral_practice_web_dev)
        
        follow me on bilibili [drawingsword](https://space.bilibili.com/64849811/)
        
        follow me on youtube [linear chu](https://www.youtube.com/channel/UCR2jqmzkrzdB_VUJ2ytospA)
        
        """)
        # st.markdown(html_temp.format("rgba(55, 53, 47, 0.16)"),unsafe_allow_html=True)
        # st.markdown("""
        # # How does it work
        # Simply click on the speak button and enjoy the conversation.
        # """)
        st.markdown(html_temp.format("rgba(55, 53, 47, 0.16)"),unsafe_allow_html=True)
        st.markdown("""
        Made by [@Bowen ZHU](https://www.linkedin.com/in/bowen-zhu-52ba181b9/)
        """,
        unsafe_allow_html=True,
        )
        



            

        
        #app_sst_side()
        if st.button('clear'):
            for key in ['count','conv','sugg']:
                del st.session_state[key]
            init()
        st.write('suggestion:'+st.session_state['sugg'])
        
    for i in range(st.session_state['count']):
            st.markdown("""
    <style>
      .type1 {
        background-color: green;
        padding: 10px;
        border-radius: 10px;
      }
      .type2 {
        background-color: darkblue;
        padding: 10px;
        border-radius: 10px;
      }
    </style>
    """, unsafe_allow_html=True)
            t_y="<div class='type1'> "+st.session_state['ME'+str(i)]+"</div>"
            t_a="<div class='type2'> "+st.session_state['YOU'+str(i)]+"</div>"
            st.write('you said: '+ t_y, unsafe_allow_html=True)
            st.write('AI said: '+ t_a, unsafe_allow_html=True)
    
    app_sst_main()

def app_sst_main():
    webrtc_ctx = webrtc_streamer(
        key="speech-to-text_main",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        rtc_configuration=rtc,
        media_stream_constraints={"video": False, "audio": True},
    )

    status_indicator = st.empty()

    if not webrtc_ctx.state.playing:
        return

    status_indicator.write("Loading...")

    i=0
    sound1 = pydub.AudioSegment.empty()
    sound_eval = pydub.AudioSegment.empty()
    #150 约为3s
    while i<2500 :
        i=i+1
        if webrtc_ctx.audio_receiver:
            

            sound_chunk = pydub.AudioSegment.empty()
            try:
                audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=5)
            except queue.Empty:
                time.sleep(0.1)
                status_indicator.write("No frame arrived.")
                continue

            status_indicator.write("Running. Say something!")

            for audio_frame in audio_frames:
                sound = pydub.AudioSegment(
                    data=audio_frame.to_ndarray().tobytes(),
                    sample_width=audio_frame.format.bytes,
                    frame_rate=audio_frame.sample_rate,
                    channels=len(audio_frame.layout.channels),
                )
                sound_chunk += sound

            if len(sound_chunk) > 0:
                sound_chunk = sound_chunk.set_channels(1).set_frame_rate(16000)
                sound1=sound1+sound_chunk
                sound_eval=sound_eval+sound_chunk
            else:
                break
            if i % 50 ==0 and i>150:
                deci_stop =np.array(sound_eval.get_array_of_samples()) # auto stop
                max_v=np.amax(deci_stop)
                if max_v<700:
                    break
                else:
                    sound_eval = pydub.AudioSegment.empty()

            
        else:
            status_indicator.write("AudioReciver is not set. Abort.")
            break
    sound1.export("output.mp3", format="mp3")
    #buffer =np.array(sound1.get_array_of_samples())
    
    st.write(sound1)
    status_indicator.write("Starting recognition and don't press stop")
    new_me=recognize_from_mic(lang_mode)
    st.session_state['count']=st.session_state['count']+1
    
    if st.session_state['count']==1:     
        
        st.session_state['conv']=st.session_state['conv']+[{"role": "system", "content": Preset}]
        st.session_state['conv'] = concatenate_me(st.session_state['conv'],new_me)
    else:
        st.session_state['conv'] = concatenate_me(st.session_state['conv'],new_me)
    Me_temp='ME'+str(st.session_state['count']-1)
    
    new_you=respond(st.session_state['conv'])
    
    synthesize_to_speaker(new_you,lang_mode,st.secrets["azurekey"])
    autoplay_audio()
    
    You_temp='YOU'+str(st.session_state['count']-1)
    
    st.session_state[You_temp]=new_you
    st.session_state[Me_temp]=new_me
    st.session_state['conv'] = concatenate_you(st.session_state['conv'],new_you)
                                                
    conversation_sugg=st.session_state['conv']+[{"role": "system", "content": 'now you(as AI) will pretent to be the user to give the respond to the assistant'}]
    sugg=suggestion(conversation_sugg)
    st.session_state['sugg']=sugg
    status_indicator.write("Press stop")
        


if __name__ == '__main__':
    
    
    main()
