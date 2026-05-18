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

def get_fft(signal):
    Y = fft(signal)
    return freq[:N // 2], np.abs(Y[:N // 2]) / N

f_soft, m_soft = get_fft(soft)
f_hard, m_hard = get_fft(hard)
f_filt, m_filt = get_fft(hard_filtered)

# -------------------
# AUDIO
# -------------------
def save_audio(data):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(tmp.name, data, fs)
    return tmp.name

# =========================================================
# 🔥 BAŞTAKİ BÜYÜK KARŞILAŞTIRMA GRAFİĞİ
# =========================================================
st.markdown("## 📊 Genel FFT Karşılaştırma Grafiği")

fig0, ax0 = plt.subplots(figsize=(8, 4))

ax0.plot(f_soft, m_soft, label="Yumuşak", color="blue")
ax0.plot(f_hard, m_hard, label="Sert", color="red", alpha=0.7)
ax0.plot(f_filt, m_filt, label="Filtreli", color="green")

ax0.axvline(cutoff, color="black", linestyle="--", label="Cutoff = 1000 Hz")

ax0.set_xlim(0, 2500)
ax0.set_xlabel("Frekans (Hz)")
ax0.set_ylabel("Genlik")
ax0.set_title("Tüm Melodilerin FFT Karşılaştırması")
ax0.legend()
ax0.grid(True)

st.pyplot(fig0)

# =========================================================
# 🎵 1. BLOK - YUMUŞAK
# =========================================================
st.markdown("## 🎵 Yumuşak Melodi")

fig1, ax1 = plt.subplots(figsize=(8, 3))
ax1.plot(f_soft, m_soft, color="blue")
ax1.set_title("Yumuşak FFT")
ax1.set_xlim(0, 2500)
ax1.grid(True)

st.pyplot(fig1)
st.audio(save_audio(soft))

# =========================================================
# 🔊 2. BLOK - SERT
# =========================================================
st.markdown("## 🔊 Sert Melodi")

fig2, ax2 = plt.subplots(figsize=(8, 3))
ax2.plot(f_hard, m_hard, color="red")
ax2.set_title("Sert FFT")
ax2.set_xlim(0, 2500)
ax2.grid(True)

st.pyplot(fig2)
st.audio(save_audio(hard))

# =========================================================
# 🎧 3. BLOK - FİLTRELİ
# =========================================================
st.markdown("## 🎧 Filtrelenmiş Melodi")

fig3, ax3 = plt.subplots(figsize=(8, 3))
ax3.plot(f_filt, m_filt, color="green")
ax3.set_title("Filtrelenmiş FFT")
ax3.set_xlim(0, 2500)
ax3.grid(True)

st.pyplot(fig3)
st.audio(save_audio(hard_filtered))
