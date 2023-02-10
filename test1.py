import streamlit as st

#from core import recognize_from_mic,synthesize_to_speaker,respond,concatenate_me,concatenate_you,suggestion
# Initialize the speech config
import base64
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
    audio_config = speechsdk.audio.AudioConfig(filename="output.wav")
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config,audio_config=audio_config)    
    #Asks user for mic input and prints transcription result on screen
    
    result = speech_recognizer.recognize_once()
    
    return result.text
def autoplay_audio(data):
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
    speech_config = speechsdk.SpeechConfig(subscription=azureapi, region="francecentral")
    speech_config.speech_synthesis_language = lang
    #In this sample we are using the default speaker 
    #Learn how to customize your speaker using SSML in Azure Cognitive Services Speech documentation
    audio_config = AudioOutputConfig(use_default_speaker=True)
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)    
    speech_synthesis_result=synthesizer.speak_text_async(text).get()
    return speech_synthesis_result
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
    # MODEL_URL = "https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.pbmm"  # noqa
    # LANG_MODEL_URL = "https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer"  # noqa
    # MODEL_LOCAL_PATH = HERE / "models/deepspeech-0.9.3-models.pbmm"
    # LANG_MODEL_LOCAL_PATH = HERE / "models/deepspeech-0.9.3-models.scorer"

    # download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=188915987)
    # download_file(LANG_MODEL_URL, LANG_MODEL_LOCAL_PATH, expected_size=953363776)

    # lm_alpha = 0.931289039105002
    # lm_beta = 1.1834137581510284
    # beam = 100

    
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
    
def app_sst():
    global buffer
    webrtc_ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": False, "audio": True},
    )

    status_indicator = st.empty()

    if not webrtc_ctx.state.playing:
        return

    status_indicator.write("Loading...")
    text_output = st.empty()
    stream = None
    i=0
    while i<100 :
        i=i+1
        if webrtc_ctx.audio_receiver:
            

            sound_chunk = pydub.AudioSegment.empty()
            try:
                audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
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
                buffer =sound_chunk.get_array_of_samples()
            else:
                break

        else:
            status_indicator.write("AudioReciver is not set. Abort.")
            break
    sound_chunk.export("output.wav", format="wav")
    st.write(buffer)
    status_indicator.write("Starting recognition...")
    
    # st.write(sound_window_buffer)
    # st.audio(sound_window_buffer)
    # st.write(1)
    # new_me=recognize_from_mic(lang_mode,azurekey)
    # st.write(2)


    new_me=recognize_from_mic(lang_mode,st.secrets["azurekey"])
    st.write(new_me)
    st.session_state['count']=st.session_state['count']+1
    
    if st.session_state['count']==1:     
        st.session_state['conv'] = concatenate_me(Preset,new_me)
        st.session_state['conv'] = concatenate_me(st.session_state['conv'],new_me)
    else:
        st.session_state['conv'] = concatenate_me(st.session_state['conv'],new_me)
    Me_temp='ME'+str(st.session_state['count']-1)
    new_you=respond(st.session_state['conv'],respond_mod,st.secrets["openaikey"])
    
    # data=synthesize_to_speaker(new_you,lang_mode,st.secrets["azurekey"])
    # autoplay_audio(data)
    
    You_temp='YOU'+str(st.session_state['count']-1)
    
    st.session_state[You_temp]=new_you
    st.session_state[Me_temp]=new_me
    st.session_state['conv'] = concatenate_you(st.session_state['conv'],new_you)

    conversation_sugg=st.session_state['conv']+'\nME:'
    sugg=suggestion(conversation_sugg,sugg_mod,st.secrets["openaikey"])
    

        


if __name__ == '__main__':
    
    
    main()
