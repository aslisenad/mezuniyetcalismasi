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
# 🖼️ GÖRSEL FFT (SUNUM İÇİN DÜZELTİLDİ)
# =========================================================
st.markdown("## 🖼️ Görüntü FFT Analizi")

# 1. Önce yükleme butonunu gösterelim
uploaded = st.file_uploader("Analiz için bir görsel yükleyin", type=["jpg", "png"])

# 2. Eğer henüz bir görsel yüklenmemişse, buraya bir uyarı yazalım ve durduralım.
if uploaded is None:
    st.warning("👉 Lütfen analiz yapabilmek için yukarıdaki butonu kullanarak bilgisayarınızdan bir görsel (.jpg veya .png) yükleyin.")
    # st.stop() komutu, kodun geri kalanını çalıştırmaz, böylece site çökmez,
    # sadece yükleme uyarısını göstererek bekler.
    st.stop()

# 3. Kod buraya geldiyse, kesinlikle bir dosya yüklenmiştir.
radius = st.slider("Filtre Yarıçapı (Radius)", 5, 150, 45)

# Yüklenen dosyayı okuyalım (Hata riski yok, çünkü 'if uploaded is None' kontrolü yaptık)
image_rgb = io.imread(uploaded)

if image_rgb.ndim == 3:
    image_gray = color.rgb2gray(image_rgb)
else:
    image_gray = image_rgb

image = img_as_float(image_gray)

# --- Buradan sonrası sizin orijinal kodunuzun devamıdır ---
F = fft2(image)
F_shifted = fftshift(F)

rows, cols = image.shape
crow, ccol = rows // 2, cols // 2

Y, X = np.ogrid[:rows, :cols]
distance = np.sqrt((X - ccol)**2 + (Y - crow)**2)

islem = st.selectbox(
    "Gerçekleştirilecek İşlem",
    ["Orijinal Görsel", "FFT Spektrumu", "Low-Pass Filtre Maskesi", "High-Pass Filtre Maskesi", "Low-Pass Uygulanmış Sonuç", "High-Pass Uygulanmış Sonuç"]
)

if islem == "Orijinal Görsel":
    result = image

elif islem == "FFT Spektrumu":
    result = np.log(1 + np.abs(F_shifted))

elif islem == "Low-Pass Filtre Maskesi":
    mask = distance <= radius
    result = np.log(1 + np.abs(F_shifted * mask))

elif islem == "High-Pass Filtre Maskesi":
    mask = distance > radius
    result = np.log(1 + np.abs(F_shifted * mask))

elif islem == "Low-Pass Uygulanmış Sonuç":
    mask = distance <= radius
    img = np.real(ifft2(ifftshift(F_shifted * mask)))
    # Sonucu normalize et
    result = (img - img.min()) / (img.max() - img.min()) if img.max() > img.min() else img

elif islem == "High-Pass Uygulanmış Sonuç":
    mask = distance > radius
    img = np.real(ifft2(ifftshift(F_shifted * mask)))
    # Sonucu normalize et
    result = (img - img.min()) / (img.max() - img.min()) if img.max() > img.min() else img

# 🔧 GÖRSELİ GÖSTER
fig, ax = plt.subplots(figsize=(4, 4), dpi=100) # Biraz daha büyütüldü
ax.imshow(result, cmap="gray")
ax.axis("off")
st.pyplot(fig, use_container_width=False)
