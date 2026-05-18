import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, ifft
import soundfile as sf
import tempfile

# -------------------
# Ayarlar
# -------------------
fs = 44100

st.set_page_config(page_title="FFT Audio Filter", layout="centered")
st.title("🎵 FFT Ses Filtreleme Uygulaması")

# -------------------
# UI
# -------------------
cutoff = st.slider("Cutoff Frekansı (Hz)", 100, 3000, 800)

notes = [440, 523, 659, 523, 440]
note_duration = 1

# -------------------
# Sinyal üretimi
# -------------------
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

# -------------------
# FFT
# -------------------
N = len(signal)
Y = fft(signal)
freq = fftfreq(N, 1/fs)

# Low-pass
Y_filtered = Y.copy()
Y_filtered[np.abs(freq) > cutoff] = 0
signal_lowpass = np.real(ifft(Y_filtered))

# High-pass
signal_highpass = signal - signal_lowpass

# -------------------
# Grafik (küçültülmüş)
# -------------------
fig, ax = plt.subplots(2, 2, figsize=(7, 3.5))

ax[0, 0].plot(signal[:3000])
ax[0, 0].set_title("Orijinal")

ax[0, 1].plot(signal_lowpass[:3000])
ax[0, 1].set_title("Low-pass")

ax[1, 0].plot(freq[:N//2], np.abs(Y[:N//2]))
ax[1, 0].set_xlim(0, 2000)
ax[1, 0].set_title("FFT Orijinal")

ax[1, 1].plot(freq[:N//2], np.abs(Y_filtered[:N//2]))
ax[1, 1].set_xlim(0, 2000)
ax[1, 1].set_title("FFT Filtreli")

plt.tight_layout()

st.pyplot(fig)

# -------------------
# Audio helper
# -------------------
def save_audio(data):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(tmp.name, data, fs)
    return tmp.name

# -------------------
# SESLER
# -------------------
st.markdown("## 🔊 Ses Çıktıları")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Orijinal")
    st.audio(save_audio(signal))

with col2:
    st.subheader("Low-pass")
    st.audio(save_audio(signal_lowpass))

with col3:
    st.subheader("High-pass")
    st.audio(save_audio(signal_highpass))
