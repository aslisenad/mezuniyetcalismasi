import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, ifft, fft2, fftshift, ifft2, ifftshift
from skimage import io, color, img_as_float
import tempfile
import soundfile as sf

# -------------------
# AYARLAR
# -------------------
fs = 44100
note_duration = 0.4
cutoff = 1000

st.set_page_config(page_title="FFT Proje", layout="centered")
st.title("🎵 FFT + Ses + Görüntü Analizi")

# =========================================================
# 🎵 SES ÜRETİMİ
# =========================================================
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

soft /= np.max(np.abs(soft))
hard /= np.max(np.abs(hard))

N = len(hard)
freq = fftfreq(N, 1 / fs)

Y_hard = fft(hard)

Y_filtered = Y_hard.copy()
Y_filtered[np.abs(freq) > cutoff] = 0

hard_filtered = np.real(ifft(Y_filtered))
hard_filtered /= np.max(np.abs(hard_filtered))

def save_audio(data):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(tmp.name, data, fs)
    return tmp.name

def get_fft(signal):
    Y = fft(signal)
    return freq[:N // 2], np.abs(Y[:N // 2]) / N

f_soft, m_soft = get_fft(soft)
f_hard, m_hard = get_fft(hard)
f_filt, m_filt = get_fft(hard_filtered)

# =========================================================
# 📊 GENEL GRAFİK
# =========================================================
st.markdown("## 📊 Genel FFT Karşılaştırma")

fig0, ax0 = plt.subplots(figsize=(7, 3.5))

ax0.plot(f_soft, m_soft, label="Yumuşak", color="blue")
ax0.plot(f_hard, m_hard, label="Sert", color="red", alpha=0.7)
ax0.plot(f_filt, m_filt, label="Filtreli", color="green")

ax0.axvline(cutoff, color="black", linestyle="--", label="Cutoff")

ax0.set_xlim(0, 2500)
ax0.set_title("Tüm Sesler FFT")
ax0.legend()
ax0.grid(True)

st.pyplot(fig0)

# =========================================================
# 🎵 SES BLOKLARI
# =========================================================
st.markdown("## 🎵 Ses Analizi")

st.markdown("### 🎵 Yumuşak")
fig1, ax1 = plt.subplots(figsize=(7, 3))
ax1.plot(f_soft, m_soft, color="blue")
ax1.set_xlim(0, 2500)
ax1.grid(True)
st.pyplot(fig1)
st.audio(save_audio(soft))

st.markdown("### 🔊 Sert")
fig2, ax2 = plt.subplots(figsize=(7, 3))
ax2.plot(f_hard, m_hard, color="red")
ax2.set_xlim(0, 2500)
ax2.grid(True)
st.pyplot(fig2)
st.audio(save_audio(hard))

st.markdown("### 🎧 Filtrelenmiş")
fig3, ax3 = plt.subplots(figsize=(7, 3))
ax3.plot(f_filt, m_filt, color="green")
ax3.set_xlim(0, 2500)
ax3.grid(True)
st.pyplot(fig3)
st.audio(save_audio(hard_filtered))

# =========================================================
# 🖼️ GÖRSEL FFT (KÜÇÜLTÜLDÜ)
# =========================================================
st.markdown("## 🖼️ Görüntü FFT Analizi")

uploaded = st.file_uploader("Görsel yükle", type=["jpg", "png"])
radius = st.slider("Radius", 5, 150, 45)

if uploaded is not None:
    image_rgb = io.imread(uploaded)
else:
    st.info("Varsayılan görsel kullanılıyor")
    image_rgb = io.imread("ataturk.jpg")

if image_rgb.ndim == 3:
    image_gray = color.rgb2gray(image_rgb)
else:
    image_gray = image_rgb

image = img_as_float(image_gray)

F = fft2(image)
F_shifted = fftshift(F)

rows, cols = image.shape
crow, ccol = rows // 2, cols // 2

Y, X = np.ogrid[:rows, :cols]
distance = np.sqrt((X - ccol)**2 + (Y - crow)**2)

islem = st.selectbox(
    "İşlem",
    ["Orijinal", "FFT", "Low-Pass", "High-Pass", "Low-Pass Sonuç", "High-Pass Sonuç"]
)

if islem == "Orijinal":
    result = image

elif islem == "FFT":
    result = np.log(1 + np.abs(F_shifted))

elif islem == "Low-Pass":
    mask = distance <= radius
    result = np.log(1 + np.abs(F_shifted * mask))

elif islem == "High-Pass":
    mask = distance > radius
    result = np.log(1 + np.abs(F_shifted * mask))

elif islem == "Low-Pass Sonuç":
    mask = distance <= radius
    img = np.real(ifft2(ifftshift(F_shifted * mask)))
    result = (img - img.min()) / (img.max() - img.min())

elif islem == "High-Pass Sonuç":
    mask = distance > radius
    img = np.real(ifft2(ifftshift(F_shifted * mask)))
    result = (img - img.min()) / (img.max() - img.min())

# 🔧 KÜÇÜLTÜLMÜŞ GÖRSEL
fig, ax = plt.subplots(figsize=(2.8, 2.8), dpi=120)
ax.imshow(result, cmap="gray")
ax.axis("off")

st.pyplot(fig, use_container_width=False)
