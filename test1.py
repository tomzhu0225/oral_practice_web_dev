import streamlit as st

#from core import recognize_from_mic,synthesize_to_speaker,respond,concatenate_me,concatenate_you,suggestion
# Initialize the speech config
import io
import azure.cognitiveservices.speech as speechsdk
from azure.cognitiveservices.speech.audio import AudioOutputConfig
import openai
from streamlit_webrtc import WebRtcMode, webrtc_streamer
import pydub
import logging
from pathlib import Path
import queue
import urllib.request
import numpy as np
import time
# def speak():
#     speech_config = speechsdk.SpeechConfig(subscription="5c05507933314a0caa980687fad5e2de", region="francecentral")
#     speech_config.speech_recognition_language = lang_mode
#     speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)    
#     #Asks user for mic input and prints transcription result on screen
    
#     result = speech_recognizer.recognize_once()
    
#     return result.text
HERE = Path(__file__).parent

logger = logging.getLogger(__name__)
def recognize_from_mic(lang,azureapi):
	#Find your key and resource region under the 'Keys and Endpoint' tab in your Speech resource in Azure Portal
	#Remember to delete the brackets <> when pasting your key and region!
    speech_config = speechsdk.SpeechConfig(subscription=azureapi, region="francecentral")
    speech_config.speech_recognition_language = lang
    audio_config = speechsdk.AudioConfig(filename='example.wav')
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config,audio_config=audio_config)    
    #Asks user for mic input and prints transcription result on screen
    
    result = speech_recognizer.recognize_once()
    
    return result.text
def respond(conversation,mod,key):
    openai.api_key = key
    response = openai.Completion.create(
    model=mod,
    #model="text-curie-001",
    prompt=conversation,
    temperature=1,
    max_tokens=150,
    top_p=1,
    frequency_penalty=1,
    presence_penalty=0.1,
    stop=["ME:", "YOU:"])
    return response.choices[0].text

def suggestion(conversation,mod,key):
    openai.api_key = key
    response = openai.Completion.create(
    model=mod,
    prompt=conversation,
    temperature=1,
    max_tokens=150,
    top_p=1,
    frequency_penalty=1,
    presence_penalty=0.1,
    stop=["ME:", "YOU:"])
    return response.choices[0].text
def concatenate_me(original,new):
    return original+'ME:\n'+new+"YOU:\n"
def concatenate_you(original,new):
    return original+new
# This code is based on https://github.com/streamlit/demo-self-driving/blob/230245391f2dda0cb464008195a470751c01770b/streamlit_app.py#L48  # noqa: E501
def download_file(url, download_to: Path, expected_size=None):
    # Don't download the file twice.
    # (If possible, verify the download using the file length.)
    if download_to.exists():
        if expected_size:
            if download_to.stat().st_size == expected_size:
                return
        else:
            st.info(f"{url} is already downloaded.")
            if not st.button("Download again?"):
                return

    download_to.parent.mkdir(parents=True, exist_ok=True)

    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading %s..." % url)
        progress_bar = st.progress(0)
        with open(download_to, "wb") as output_file:
            with urllib.request.urlopen(url) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    weights_warning.warning(
                        "Downloading %s... (%6.2f/%6.2f MB)"
                        % (url, counter / MEGABYTES, length / MEGABYTES)
                    )
                    progress_bar.progress(min(counter / length, 1.0))
    # Finally, we remove these visual elements by calling .empty().
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()
    
 

def main():
    global lang_mode
    global text_output
    global Preset
    global respond_mod
    global sugg_mod
    
    #model


    
    #init
    
    
    if 'count' not in st.session_state:
        st.session_state['count'] = 0
    Me_temp='ME'+str(st.session_state['count'])
    if  Me_temp not in st.session_state:
        st.session_state[Me_temp]=''
    if 'conv' not in st.session_state:
        st.session_state['conv'] = ''
    You_temp='YOU'+str(st.session_state['count'])
    if You_temp not in st.session_state:
        st.session_state[You_temp]=''
    #stun server

    st.header("Oral practice with AI(网页版开发中...)")


    html_temp = """
                    <div style="background-color:{};padding:1px">
                    
                    </div>
                    """
    left, right = st.columns(2)
    with left: 
        lang_mode = st.selectbox("Choose your language", ["zh-CN", "en-US", "fr-FR", "ja-JP"],key='lang')
    with right:
        int_mode = st.selectbox('Choose the model',["high Intelligence", "medium Intelligence", "low Intelligence"],key='intel')
    if int_mode=='high Intelligence':
        respond_mod="text-davinci-003"
        sugg_mod="text-davinci-003"
    elif int_mode=='medium Intelligence':
        respond_mod="text-davinci-003"
        sugg_mod="text-curie-001"
    else:
        respond_mod="text-curie-001"
        sugg_mod="text-curie-001"
     
    Preset = st.text_input('Preset', placeholder="Enter the scene setting")
    #st.write('The current movie title is', title)
    
    
    
    
    
      
    
    with st.sidebar:
        st.markdown("""
        # About 
        This page is providing a new way of practice your oral with openai! 
        """)
        st.markdown(html_temp.format("rgba(55, 53, 47, 0.16)"),unsafe_allow_html=True)
        st.markdown("""
        # How does it work
        Simply click on the speak button and enjoy the conversation.
        """)
        st.markdown(html_temp.format("rgba(55, 53, 47, 0.16)"),unsafe_allow_html=True)
        st.markdown("""
        Made by [@Bowen ZHU](https://github.com/tomzhu0225/oral_practice_with_openai)
        """,
        unsafe_allow_html=True,
        )
        
        app_sst()
          

        
        
    for i in range(st.session_state['count']+1):
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
    
def app_sst():
    """A sample to use WebRTC in sendonly mode to transfer audio frames
    from the browser to the server and visualize them with matplotlib
    and `st.pyplot`."""
    webrtc_ctx = webrtc_streamer(
        key="sendonly-audio",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.xten.com:3478"]}]},
        media_stream_constraints={"audio": True},
    )


    sound_window_len = 15000  # 15s
    sound_window_buffer = None
    while True:
        if webrtc_ctx.audio_receiver:
            try:
                audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
            except queue.Empty:
                logger.warning("Queue is empty. Abort.")
                break

            sound_chunk = pydub.AudioSegment.empty()
            for audio_frame in audio_frames:
                sound = pydub.AudioSegment(
                    data=audio_frame.to_ndarray().tobytes(),
                    sample_width=audio_frame.format.bytes,
                    frame_rate=audio_frame.sample_rate,
                    channels=len(audio_frame.layout.channels),
                )
                sound_chunk += sound

            if len(sound_chunk) > 0:
                if sound_window_buffer is None:
                    sound_window_buffer = pydub.AudioSegment.silent(
                        duration=sound_window_len
                    )

                sound_window_buffer += sound_chunk
                if len(sound_window_buffer) > sound_window_len:
                    sound_window_buffer = sound_window_buffer[-sound_window_len:]

    st.write(sound_window_buffer)
    
    st.write(1)
    new_me=recognize_from_mic(lang_mode,azurekey)
    st.write(2)
    st.session_state['count']=st.session_state['count']+1
    
    if st.session_state['count']==1:     
        st.session_state['conv'] = concatenate_me(Preset,new_me)
        st.session_state['conv'] = concatenate_me(st.session_state['conv'],new_me)
    else:
        st.session_state['conv'] = concatenate_me(st.session_state['conv'],new_me)
    Me_temp='ME'+str(st.session_state['count']-1)
    new_you=respond(st.session_state['conv'],respond_mod,openaikey)
    You_temp='YOU'+str(st.session_state['count']-1)
    
    st.session_state[You_temp]=new_you
    st.session_state[Me_temp]=new_me
    st.session_state['conv'] = concatenate_you(st.session_state['conv'],new_you)

    conversation_sugg=st.session_state['conv']+'\nME:'
    sugg=suggestion(conversation_sugg,sugg_mod,openaikey)
    

        


if __name__ == '__main__':
    
    
    main()
