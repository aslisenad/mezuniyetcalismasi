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

st.set_page_config(page_title="FFT + Image + Audio", layout="centered")
st.title("🎵 FFT + 🖼️ Görüntü + 🔊 Ses Analizi")

# =========================================================
# 🎵 SES KISMI
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

# =========================================================
# 🖼️ GÖRSEL FFT KISMI
# =========================================================
st.markdown("## 🖼️ Görüntü FFT Analizi")

uploaded = st.file_uploader("Bir görsel yükle (jpg/png)", type=["jpg", "png"])

radius = st.slider("Low/High Pass Radius", 5, 150, 45)

if uploaded is not None:
    image_rgb = io.imread(uploaded)
else:
    st.info("Varsayılan görsel kullanılıyor (ataturk.jpg)")
    image_rgb = io.imread("ataturk.jpg")

# gri
if image_rgb.ndim == 3:
    image_gray = color.rgb2gray(image_rgb)
else:
    image_gray = image_rgb

image = img_as_float(image_gray)

# FFT 2D
F = fft2(image)
F_shifted = fftshift(F)

rows, cols = image.shape
crow, ccol = rows // 2, cols // 2

Y, X = np.ogrid[:rows, :cols]
distance = np.sqrt((X - ccol)**2 + (Y - crow)**2)

# dropdown
islem = st.selectbox(
    "İşlem seç",
    [
        "Orijinal",
        "FFT Spektrumu",
        "Low-Pass Spektrumu",
        "Low-Pass Sonuç",
        "High-Pass Spektrumu",
        "High-Pass Sonuç"
    ]
)

if islem == "Orijinal":
    result = image

elif islem == "FFT Spektrumu":
    result = np.log(1 + np.abs(F_shifted))

elif islem == "Low-Pass Spektrumu":
    mask = distance <= radius
    result = np.log(1 + np.abs(F_shifted * mask))

elif islem == "Low-Pass Sonuç":
    mask = distance <= radius
    img = np.real(ifft2(ifftshift(F_shifted * mask)))
    result = (img - img.min()) / (img.max() - img.min())

elif islem == "High-Pass Spektrumu":
    mask = distance > radius
    result = np.log(1 + np.abs(F_shifted * mask))

elif islem == "High-Pass Sonuç":
    mask = distance > radius
    img = np.real(ifft2(ifftshift(F_shifted * mask)))
    result = (img - img.min()) / (img.max() - img.min())

fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(result, cmap="gray")
ax.set_title(islem)
ax.axis("off")

st.pyplot(fig)

# =========================================================
# 🎵 SES BLOKLARI (3 SES)
# =========================================================
st.markdown("## 🔊 Ses Analizi")

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
