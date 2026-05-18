import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, ifft
import soundfile as sf
import tempfile

fs = 44100

st.title("FFT Müzik Filtreleme Uygulaması")

cutoff = st.slider("Cutoff Frekansı", 100, 3000, 800)

notes = [440, 523, 659, 523, 440]
note_duration = 1

signal = np.array([])

for f in notes:
    t_note = np.linspace(0, note_duration, int(fs * note_duration), endpoint=False)

    note = (
        np.sin(2*np.pi*f*t_note) +
        0.4*np.sin(2*np.pi*2*f*t_note) +
        0.2*np.sin(2*np.pi*3*f*t_note)
    )

    signal = np.concatenate((signal, note))

signal = signal / np.max(np.abs(signal))

N = len(signal)

Y = fft(signal)
freq = fftfreq(N, 1/fs)

Y_filtered = Y.copy()
Y_filtered[np.abs(freq) > cutoff] = 0

signal_filtered = np.real(ifft(Y_filtered))

fig, ax = plt.subplots(2,2, figsize=(12,6))

ax[0,0].plot(signal[:3000])
ax[0,0].set_title("Orijinal")

ax[0,1].plot(signal_filtered[:3000])
ax[0,1].set_title("Filtrelenmiş")

ax[1,0].plot(freq[:N//2], np.abs(Y[:N//2]))
ax[1,0].set_xlim(0,2000)

ax[1,1].plot(freq[:N//2], np.abs(Y_filtered[:N//2]))
ax[1,1].set_xlim(0,2000)

st.pyplot(fig)

# Audio export
temp1 = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
sf.write(temp1.name, signal, fs)

temp2 = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
sf.write(temp2.name, signal_filtered, fs)

st.subheader("Orijinal Ses")
st.audio(temp1.name)

st.subheader("Filtrelenmiş Ses")
st.audio(temp2.name)
