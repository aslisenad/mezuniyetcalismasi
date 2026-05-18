import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, ifft
import tempfile
import soundfile as sf

# -------------------
# AYARLAR
# -------------------
fs = 44100
note_duration = 0.4
cutoff = 1000

st.set_page_config(page_title="FFT Melodi Analizi", layout="centered")
st.title("🎵 FFT Melodi Karşılaştırma Uygulaması")

# -------------------
# MELODİ ÜRETİMİ
# -------------------
notes = [440, 494, 523, 587, 659, 587, 523, 494, 440]

soft = np.array([])
hard = np.array([])

for f in notes:
    t = np.linspace(0, note_duration, int(fs * note_duration), endpoint=False)

    s = np.sin(2 * np.pi * f * t)

    h = (
        np.sin(2 * np.pi * f * t) +
        0.6 * np.sin(2 * np.pi * 2 * f * t) +
        0.35 * np.sin(2 * np.pi * 3 * f * t) +
        0.2 * np.random.randn(len(t))
    )

    soft = np.concatenate((soft, s))
    hard = np.concatenate((hard, h))

soft = soft / np.max(np.abs(soft))
hard = hard / np.max(np.abs(hard))

# -------------------
# FFT
# -------------------
N = len(hard)
freq = fftfreq(N, 1 / fs)

Y_hard = fft(hard)

Y_filtered = Y_hard.copy()
Y_filtered[np.abs(freq) > cutoff] = 0

hard_filtered = np.real(ifft(Y_filtered))
hard_filtered = hard_filtered / np.max(np.abs(hard_filtered))

# -------------------
# FFT helper
# -------------------
def get_fft(signal):
    Y = fft(signal)
    return freq[:N // 2], np.abs(Y[:N // 2]) / N

f_soft, m_soft = get_fft(soft)
f_hard, m_hard = get_fft(hard)
f_filt, m_filt = get_fft(hard_filtered)

# -------------------
# GRAFİK
# -------------------
st.subheader("📊 FFT Karşılaştırma Grafiği")

fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(f_soft, m_soft, label="Yumuşak Melodi", color="blue")
ax.plot(f_hard, m_hard, label="Sert Melodi", color="red", alpha=0.7)
ax.plot(f_filt, m_filt, label="Filtrelenmiş Sert Melodi", color="green")

ax.axvline(cutoff, color="black", linestyle="--", label="Cutoff = 1000 Hz")

ax.set_xlim(0, 2500)
ax.set_xlabel("Frekans (Hz)")
ax.set_ylabel("Genlik")
ax.set_title("FFT Karşılaştırması")
ax.legend()
ax.grid(True)

st.pyplot(fig)

# -------------------
# AUDIO EXPORT
# -------------------
def save_audio(data):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(tmp.name, data, fs)
    return tmp.name

st.subheader("🔊 Sesler")

col1, col2, col3 = st.columns(3)

with col1:
    st.write("Yumuşak")
    st.audio(save_audio(soft))

with col2:
    st.write("Sert")
    st.audio(save_audio(hard))

with col3:
    st.write("Filtrelenmiş")
    st.audio(save_audio(hard_filtered))

# -------------------
# ANİMASYON (STREAMLIT VERSION)
# -------------------
st.subheader("🎞️ FFT Değişim Görselleştirme")

fig2, ax2 = plt.subplots(figsize=(8, 4))

data = [
    (f_soft, m_soft, "Yumuşak", "blue"),
    (f_hard, m_hard, "Sert", "red"),
    (f_filt, m_filt, "Filtrelenmiş", "green"),
]

choice = st.selectbox("Görselleştir:", ["Yumuşak", "Sert", "Filtrelenmiş"])

if choice == "Yumuşak":
    x, y = f_soft, m_soft
elif choice == "Sert":
    x, y = f_hard, m_hard
else:
    x, y = f_filt, m_filt

ax2.plot(x, y, color="purple")
ax2.set_xlim(0, 2500)
ax2.set_title(f"{choice} FFT")
ax2.grid(True)

st.pyplot(fig2)
