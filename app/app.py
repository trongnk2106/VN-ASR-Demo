import io
import librosa
import numpy as np
import streamlit as st
import audiomentations
from matplotlib import pyplot as plt
# import librosa.display
from scipy.io import wavfile
import pydub
import sys
import pyaudio
import os
import wave
from model import S2T_Model

plt.rcParams["figure.figsize"] = (10, 7)


def record_audio(filename, duration=10):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100

    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

    st.sidebar.info(f"Recording... Speak into the microphone (max {duration} seconds).")
    frames = []
    for i in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)

    st.sidebar.success("Recording complete. Click 'Recognize' to process.")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


def create_pipeline(transformations: list):
    pipeline = []
    for index, transformation in enumerate(transformations):
        if transformation:
            pipeline.append(index_to_transformation(index))

    return pipeline


def create_audio_player(audio_data, sample_rate):
    virtualfile = io.BytesIO()
    wavfile.write(virtualfile, rate=sample_rate, data=audio_data)

    return virtualfile


@st.cache_data
def handle_uploaded_audio_file(uploaded_file):
    a = pydub.AudioSegment.from_file(
        file=uploaded_file, format=uploaded_file.name.split(".")[-1]
    )

    channel_sounds = a.split_to_mono()
    samples = [s.get_array_of_samples() for s in channel_sounds]

    fp_arr = np.array(samples).T.astype(np.float32)
    fp_arr /= np.iinfo(samples[0].typecode).max

    return fp_arr[:, 0], a.frame_rate


def plot_wave(y, sr):
    print(y, sr)
    print("bug here")
    fig, ax = plt.subplots()
    img = librosa.display.waveshow(y, sr=sr, ax =ax, color="blue")

    return plt.gcf()


def plot_transformation(y, sr, transformation_name):
    D = librosa.stft(y)  # STFT of y
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(S_db, x_axis='time', y_axis='linear', ax=ax)
    ax.set(title=transformation_name)
    fig.colorbar(img, ax=ax, format="%+2.f dB")

    return plt.gcf()


def spacing():
    st.markdown("<br></br>", unsafe_allow_html=True)


def plot_audio_transformations(y, sr, pipeline: audiomentations.Compose):
    cols = [1, 1, 1]

    col1, col2, col3 = st.columns(cols)
    with col1:
        st.markdown(
            f"<h4 style='text-align: center; color: black;'>Original</h5>",
            unsafe_allow_html=True,
        )
        st.pyplot(plot_transformation(y, sr, "Original"))
    with col2:
        st.markdown(
            f"<h4 style='text-align: center; color: black;'>Wave plot </h5>",
            unsafe_allow_html=True,
        )
        st.pyplot(plot_wave(y, sr))
    with col3:
        st.markdown(
            f"<h4 style='text-align: center; color: black;'>Audio</h5>",
            unsafe_allow_html=True,
        )
        spacing()
        st.audio(create_audio_player(y, sr))
    st.markdown("---")

    y = y
    sr = sr
    for col_index, individual_transformation in enumerate(pipeline.transforms):
        transformation_name = (
            str(type(individual_transformation)).split("'")[1].split(".")[-1]
        )
        modified = individual_transformation(y, sr)
        fig = plot_transformation(modified, sr, transformation_name=transformation_name)
        y = modified

        col1, col2, col3 = st.columns(cols)

        with col1:
            st.markdown(
                f"<h4 style='text-align: center; color: black;'>{transformation_name}</h5>",
                unsafe_allow_html=True,
            )
            st.pyplot(fig)
        with col2:
            st.markdown(
                f"<h4 style='text-align: center; color: black;'>Wave plot </h5>",
                unsafe_allow_html=True,
            )
            st.pyplot(plot_wave(modified, sr))
            spacing()

        with col3:
            st.markdown(
                f"<h4 style='text-align: center; color: black;'>Audio</h5>",
                unsafe_allow_html=True,
            )
            spacing()
            st.audio(create_audio_player(modified, sr))
        st.markdown("---")
        plt.close("all")


def load_audio_sample(file):
    y, sr = librosa.load(file, sr=22050)

    return y, sr


def index_to_transformation(index: int):
    if index == 0:
        return audiomentations.AddGaussianNoise(p=1.0)
    elif index == 1:
        return audiomentations.FrequencyMask(p=1.0)
    elif index == 2:
        return audiomentations.TimeMask(p=0.5)
    elif index == 3:
        return audiomentations.Padding(p=1.0)





def recognize(file_path):
    text_out = S2T_Model.predict(file_path)
    return text_out



def action(file_uploader, transformations):
    #save up file
    if not os.path.exists("./temp"):
        os.makedirs("./temp")
    path_save = os.path.join('./temp' ,file_uploader.name)
    with open(path_save ,"wb") as f:
        f.write(file_uploader.getbuffer())
        
    
    if file_uploader is not None:
        y, sr = handle_uploaded_audio_file(file_uploader)
    else:
        y, sr = None, None

    pipeline = audiomentations.Compose(create_pipeline(transformations))
    try:
        plot_audio_transformations(y, sr, pipeline)
    except Exception as e:
        print(e)
        print("No files selected!!!")
    
    
    text = recognize(path_save)
    st.success(text)
    st.balloons()
    #remove file
    os.remove(path_save)

def main():
    placeholder = st.empty()
    placeholder2 = st.empty()
    st.markdown(
        "# Vietnamese Speech to Text App\n"
        "Once you have chosen augmentation techniques, select or upload an audio file\n. "
        'Then click "Apply" to start! \n\n'
    )

    if True:
        col1, col2, col3 = st.columns([1,9,1])

        with col1:
            st.write("")

        with col2:
            st.image("./app/assets/demoo.gif")

        with col3:
            st.write("")

        st.subheader("Team members:")
        members = ''' 
            Nguyen Thanh Trong\n
            Nguyen Dang Quang Tuan\n'''
        st.markdown(members)

        st.success("Waiting")
    # st.sidebar.image("demo/assets/demoo.gif")
    st.sidebar.markdown("Choose the transformations here:")
    gaussian_noise = st.sidebar.checkbox("GaussianNoise")
    frequency_mask = st.sidebar.checkbox("FrequencyMask")
    time_mask = st.sidebar.checkbox("TimeMask")
    padding = st.sidebar.checkbox("Padding")

    st.sidebar.markdown("---")
    st.sidebar.markdown("Upload an audio file here:")
    file_uploader = st.sidebar.file_uploader(
        label="", type=[".wav", ".wave", ".flac", ".mp3", ".ogg"]
    )
    st.sidebar.markdown("---")
    record_button = st.sidebar.button("Voice Here")

    if record_button:
        
        if not os.path.exists("./temp"):
            os.makedirs("./temp")
            
        file_path = os.path.join("./temp", "recorded_audio.wav")
        record_audio(file_path)

    st.sidebar.markdown("---")
    if st.sidebar.button("Recognize"):
        placeholder.empty()
        placeholder2.empty()
        transformations = [
            gaussian_noise,
            frequency_mask,
            time_mask,
            padding,
        ]

        action(
            file_uploader=file_uploader,
            transformations=transformations,
        )




if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="Audio augmentation visualization")
    main()