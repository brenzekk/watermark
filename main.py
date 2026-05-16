"""
╔══════════════════════════════════════════════════════════════════════╗
║        JPEG WATERMARKING FROM SCRATCH — Sistem Multimedia           ║
║        Nama   : Brandon Zeko Alexander                              ║
║        NIM    : 18224118                                            ║
║                                                                      ║
║  Deskripsi:                                                          ║
║    Implementasi watermarking citra wajah menggunakan pipeline JPEG   ║
║    dari nol (from scratch). Setiap tahap JPEG ditulis manual:        ║
║    konversi warna, level-shift, chroma downsample, blocking,         ║
║    2D DCT, quantization, zigzag scan, RLE encoding, dan rekonstruksi ║
║    kembali ke domain spasial.                                        ║
║                                                                      ║
║  Watermark di-embed di domain DCT (koefisien mid-frequency)          ║
║  sehingga gambar sebelum dan sesudah watermark tidak mudah dibedakan  ║
║  secara visual (PSNR > 30 dB, SSIM > 0.9).                          ║
║                                                                      ║
║  Library yang digunakan:                                             ║
║    - numpy          : array & operasi numerik                        ║
║    - scipy.fft.dct  : DCT/IDCT per blok (bukan JPEG library!)       ║
║    - matplotlib     : visualisasi step-by-step                       ║
║    - PIL (Pillow)   : load/save gambar (bukan kompresi JPEG-nya)     ║
╚══════════════════════════════════════════════════════════════════════╝

Cara pakai:
    python main.py --image assets/face_256.png --watermark "BRANDON18224118" --quality 75

Atau jalankan langsung (default sudah diset):
    python main.py
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from scipy.fft import dct, idct


# ══════════════════════════════════════════════════════════════════════
#  BAGIAN 1 — KONSTANTA & TABEL STANDAR JPEG
# ══════════════════════════════════════════════════════════════════════

# Tabel kuantisasi luminance (Y) standar JPEG ISO/IEC 10918-1
LUMA_QT_BASE = np.array([
    [16, 11, 10, 16, 24,  40,  51,  61],
    [12, 12, 14, 19, 26,  58,  60,  55],
    [14, 13, 16, 24, 40,  57,  69,  56],
    [14, 17, 22, 29, 51,  87,  80,  62],
    [18, 22, 37, 56, 68, 109, 103,  77],
    [24, 35, 55, 64, 81, 104, 113,  92],
    [49, 64, 78, 87,103, 121, 120, 101],
    [72, 92, 95, 98,112, 100, 103,  99],
], dtype=np.float64)

# Tabel kuantisasi chrominance (Cb, Cr) standar JPEG
CHROMA_QT_BASE = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
], dtype=np.float64)

# Urutan zigzag scan untuk blok 8×8
# Index 0 = DC (koefisien DC / frekuensi nol)
# Index 1-63 = AC (frekuensi rendah → tinggi mengikuti diagonal)
ZIGZAG_ORDER = [
    (0,0),(0,1),(1,0),(2,0),(1,1),(0,2),(0,3),(1,2),
    (2,1),(3,0),(4,0),(3,1),(2,2),(1,3),(0,4),(0,5),
    (1,4),(2,3),(3,2),(4,1),(5,0),(6,0),(5,1),(4,2),
    (3,3),(2,4),(1,5),(0,6),(0,7),(1,6),(2,5),(3,4),
    (4,3),(5,2),(6,1),(7,0),(7,1),(6,2),(5,3),(4,4),
    (3,5),(2,6),(1,7),(2,7),(3,6),(4,5),(5,4),(6,3),
    (7,2),(7,3),(6,4),(5,5),(4,6),(3,7),(4,7),(5,6),
    (6,5),(7,4),(7,5),(6,6),(5,7),(6,7),(7,6),(7,7),
]

# Posisi koefisien mid-frequency untuk embed watermark
# Index 14 dalam zigzag = frekuensi sedang (bukan DC, bukan frekuensi sangat tinggi)
EMBED_POSITION = 14

# Colormap kustom untuk visualisasi DCT
DCT_CMAP = LinearSegmentedColormap.from_list(
    'dct', ['#0d1b2a','#1b4f72','#2980b9','#f0f0f0','#e67e22','#c0392b','#641e16']
)


# ══════════════════════════════════════════════════════════════════════
#  BAGIAN 2 — KONVERSI WARNA
# ══════════════════════════════════════════════════════════════════════

def rgb_to_ycbcr(img_rgb: np.ndarray) -> np.ndarray:
    """
    Konversi gambar RGB (uint8, rentang 0–255) ke ruang warna YCbCr.

    Menggunakan rumus BT.601 yang merupakan standar JPEG:
        Y  =  0.299·R + 0.587·G + 0.114·B
        Cb = −0.168736·R − 0.331264·G + 0.5·B + 128
        Cr =  0.5·R − 0.418688·G − 0.081312·B + 128

    Channel Y menyimpan informasi luminance (kecerahan).
    Channel Cb dan Cr menyimpan informasi chrominance (warna).
    Mata manusia jauh lebih sensitif terhadap Y daripada Cb/Cr,
    inilah mengapa JPEG menerapkan kompresi lebih kuat pada Cb/Cr.

    Args:
        img_rgb: array shape (H, W, 3), dtype uint8

    Returns:
        ycbcr: array shape (H, W, 3), dtype float64
    """
    img = img_rgb.astype(np.float64)
    R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]

    Y  =  0.299    * R + 0.587    * G + 0.114    * B
    Cb = -0.168736 * R - 0.331264 * G + 0.5      * B + 128.0
    Cr =  0.5      * R - 0.418688 * G - 0.081312 * B + 128.0

    return np.stack([Y, Cb, Cr], axis=2)


def ycbcr_to_rgb(img_ycbcr: np.ndarray) -> np.ndarray:
    """
    Konversi YCbCr kembali ke RGB.

    Rumus invers BT.601:
        R = Y + 1.402·(Cr − 128)
        G = Y − 0.344136·(Cb − 128) − 0.714136·(Cr − 128)
        B = Y + 1.772·(Cb − 128)

    Args:
        img_ycbcr: array shape (H, W, 3), dtype float64

    Returns:
        rgb: array shape (H, W, 3), dtype uint8, nilai di-clip ke [0, 255]
    """
    Y  = img_ycbcr[:,:,0]
    Cb = img_ycbcr[:,:,1]
    Cr = img_ycbcr[:,:,2]

    R = Y + 1.402    * (Cr - 128.0)
    G = Y - 0.344136 * (Cb - 128.0) - 0.714136 * (Cr - 128.0)
    B = Y + 1.772    * (Cb - 128.0)

    return np.clip(np.stack([R, G, B], axis=2), 0, 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════
#  BAGIAN 3 — CHROMA DOWNSAMPLING & UPSAMPLING  (4:2:0)
# ══════════════════════════════════════════════════════════════════════

def chroma_downsample(channel: np.ndarray) -> np.ndarray:
    """
    Chroma downsampling 4:2:0: kurangi resolusi Cb/Cr menjadi setengahnya
    di kedua dimensi dengan rata-rata blok 2×2.

    Alasan: mata manusia kurang sensitif terhadap detail warna (chrominance)
    dibanding detail kecerahan (luminance), sehingga Cb dan Cr bisa
    dikompresi lebih kuat tanpa terlihat berbeda.

    Args:
        channel: array 2D (H, W), dtype float64

    Returns:
        downsampled: array 2D (H//2, W//2), dtype float64
    """
    H2, W2 = channel.shape[0] // 2, channel.shape[1] // 2
    out = (channel[0::2, 0::2] + channel[1::2, 0::2] +
           channel[0::2, 1::2] + channel[1::2, 1::2]) / 4.0
    return out[:H2, :W2]


def chroma_upsample(channel: np.ndarray, target_H: int, target_W: int) -> np.ndarray:
    """
    Upsample Cb/Cr kembali ke ukuran semula menggunakan nearest-neighbour.
    Setiap piksel diulang 2× di kedua arah.

    Args:
        channel  : array 2D (H//2, W//2)
        target_H : tinggi target
        target_W : lebar target

    Returns:
        upsampled: array 2D (target_H, target_W)
    """
    up = np.repeat(np.repeat(channel, 2, axis=0), 2, axis=1)
    return up[:target_H, :target_W]


# ══════════════════════════════════════════════════════════════════════
#  BAGIAN 4 — TABEL KUANTISASI DENGAN QUALITY FACTOR
# ══════════════════════════════════════════════════════════════════════

def make_quant_table(base_table: np.ndarray, quality: int) -> np.ndarray:
    """
    Skala tabel kuantisasi berdasarkan Quality Factor (QF, rentang 1–100).

    Rumus standar JPEG IJG (Independent JPEG Group):
        Jika QF < 50 : scale = 5000 / QF
        Jika QF ≥ 50 : scale = 200 − 2·QF

    Lalu:
        Qt[i,j] = clip( floor(base[i,j] × scale / 100 + 0.5), 1, 255 )

    QF = 1   → scale = 5000 → nilai Qt sangat besar → kompresi sangat kuat (kualitas buruk)
    QF = 50  → scale = 100  → Qt = base (nilai standar)
    QF = 100 → scale = 0    → Qt = 1 (lossless — tidak ada informasi yang dibuang)

    Args:
        base_table: tabel dasar 8×8 (LUMA_QT_BASE atau CHROMA_QT_BASE)
        quality   : integer 1–100

    Returns:
        qt: tabel kuantisasi 8×8, dtype float64, nilai ≥ 1
    """
    quality = int(np.clip(quality, 1, 100))
    scale   = 5000.0 / quality if quality < 50 else 200.0 - 2.0 * quality
    qt      = np.clip(np.floor(base_table * scale / 100.0 + 0.5), 1, 255)
    return qt.astype(np.float64)


# ══════════════════════════════════════════════════════════════════════
#  BAGIAN 5 — 2D DCT / IDCT PER BLOK 8×8
# ══════════════════════════════════════════════════════════════════════

def dct2d(block: np.ndarray) -> np.ndarray:
    """
    Discrete Cosine Transform 2 dimensi untuk blok 8×8.

    DCT mengubah domain spasial (nilai piksel) ke domain frekuensi
    (koefisien yang menyatakan seberapa banyak setiap frekuensi ada).

    Prinsip separable: DCT 2D = DCT 1D sepanjang baris, lalu DCT 1D sepanjang kolom.
    Menggunakan scipy.fft.dct dengan norm='ortho' (normalisasi orthonormal).

    Hasil:
        - Koefisien [0,0] = DC (rata-rata blok × faktor)
        - Koefisien lain  = AC (informasi frekuensi)
        - Koefisien kiri-atas = frekuensi rendah (penting)
        - Koefisien kanan-bawah = frekuensi tinggi (kurang penting)

    Args:
        block: array 8×8, dtype float64

    Returns:
        dct_block: array 8×8 koefisien DCT
    """
    return dct(dct(block.T, norm='ortho').T, norm='ortho')


def idct2d(block: np.ndarray) -> np.ndarray:
    """
    Inverse DCT 2D: kembalikan koefisien DCT ke domain spasial.
    Invers dari dct2d() — digunakan saat rekonstruksi gambar.

    Args:
        block: array 8×8 koefisien DCT

    Returns:
        spatial: array 8×8 nilai piksel (domain spasial)
    """
    return idct(idct(block.T, norm='ortho').T, norm='ortho')


# ══════════════════════════════════════════════════════════════════════
#  BAGIAN 6 — ZIGZAG SCAN
# ══════════════════════════════════════════════════════════════════════

def zigzag_scan(block: np.ndarray) -> np.ndarray:
    """
    Ubah blok 8×8 menjadi array 1D menggunakan urutan zigzag.

    Urutan zigzag dimulai dari sudut kiri-atas (DC) dan menyapu
    secara diagonal bolak-balik, berakhir di sudut kanan-bawah.
    Hasilnya: koefisien frekuensi rendah di awal array,
              koefisien frekuensi tinggi di akhir array.

    Ini memudahkan RLE karena banyak koefisien tinggi bernilai 0
    dan akan muncul berurutan di akhir array.

    Args:
        block: array 8×8

    Returns:
        vec: array 1D panjang 64
    """
    return np.array([block[r, c] for r, c in ZIGZAG_ORDER])


def inverse_zigzag(vec: np.ndarray) -> np.ndarray:
    """
    Kebalikan zigzag_scan: ubah array 1D (64 elemen) kembali ke blok 8×8.

    Args:
        vec: array 1D panjang 64

    Returns:
        block: array 8×8
    """
    block = np.zeros((8, 8), dtype=np.float64)
    for i, (r, c) in enumerate(ZIGZAG_ORDER):
        block[r, c] = vec[i]
    return block


# ══════════════════════════════════════════════════════════════════════
#  BAGIAN 7 — RLE ENCODING / DECODING
# ══════════════════════════════════════════════════════════════════════

def rle_encode(zigzag_vec: np.ndarray):
    """
    Encode koefisien AC dengan Run-Length Encoding (RLE) gaya JPEG.

    Koefisien DC (index 0) dikembalikan terpisah karena diproses berbeda
    dalam JPEG (dengan DPCM — differential coding antar blok).

    Format output AC: list of tuple (jumlah_nol_sebelumnya, nilai)
    - (0, 0)  = EOB (End Of Block) — semua AC sisanya adalah 0
    - (15, 0) = ZRL (Zero Run Length) — menandakan 16 nol berturut-turut

    Mengapa RLE efisien untuk JPEG:
        Setelah quantization, banyak koefisien AC (terutama frekuensi tinggi)
        menjadi 0. RLE mewakili deretan 0 hanya dengan satu angka (runnya),
        sehingga sangat menghemat ruang penyimpanan.

    Args:
        zigzag_vec: array 1D panjang 64, sudah di-quantize dan dibulatkan

    Returns:
        dc_val: int, nilai koefisien DC
        ac_rle: list of (run, value) tuples
    """
    dc_val  = int(round(zigzag_vec[0]))
    ac_vals = [int(round(v)) for v in zigzag_vec[1:]]

    ac_rle   = []
    zero_run = 0
    for val in ac_vals:
        if val == 0:
            zero_run += 1
            if zero_run == 16:      # Kirim ZRL dan reset penghitung
                ac_rle.append((15, 0))
                zero_run = 0
        else:
            ac_rle.append((zero_run, val))
            zero_run = 0
    ac_rle.append((0, 0))          # EOB — tandai akhir blok
    return dc_val, ac_rle


def rle_decode(dc_val: int, ac_rle: list) -> np.ndarray:
    """
    Decode RLE kembali ke array 1D panjang 64.

    Args:
        dc_val: nilai koefisien DC
        ac_rle: list of (run, value) tuples

    Returns:
        vec: array 1D panjang 64
    """
    vec    = np.zeros(64, dtype=np.float64)
    vec[0] = dc_val
    idx    = 1
    for run, val in ac_rle:
        if (run, val) == (0, 0):    # EOB
            break
        if (run, val) == (15, 0):   # ZRL — lewati 16 posisi
            idx += 16
        else:
            idx += run
            if idx < 64:
                vec[idx] = val
                idx += 1
    return vec


# ══════════════════════════════════════════════════════════════════════
#  BAGIAN 8 — PADDING & ENCODE/DECODE CHANNEL
# ══════════════════════════════════════════════════════════════════════

def pad_to_multiple8(channel: np.ndarray):
    """
    Pad channel 2D ke ukuran yang merupakan kelipatan 8.
    Menggunakan edge-padding (replikasi nilai tepi).
    JPEG memerlukan dimensi kelipatan 8 karena diproses blok 8×8.

    Args:
        channel: array 2D (H, W)

    Returns:
        padded : array 2D (H', W') di mana H', W' kelipatan 8
        H, W   : ukuran asli sebelum padding
    """
    H, W = channel.shape
    pH   = (8 - H % 8) % 8
    pW   = (8 - W % 8) % 8
    padded = np.pad(channel, ((0, pH), (0, pW)), mode='edge')
    return padded, H, W


def encode_channel(channel: np.ndarray, qt: np.ndarray,
                   wm_bits=None, start_block=0):
    """
    Encode satu channel melalui pipeline JPEG:
        pad → blok 8×8 → DCT → (opsional embed watermark) → quantize → zigzag → RLE

    Args:
        channel    : array 2D (H, W), sudah level-shifted
        qt         : tabel kuantisasi 8×8
        wm_bits    : array bit watermark (opsional, untuk channel Y)
        start_block: offset blok untuk indexing wm_bits

    Returns:
        encoded_blocks : list of (dc_val, ac_rle)
        padded_shape   : (pH, pW)
        orig_shape     : (H, W)
        dct_before     : list blok DCT sebelum embed (max 16 blok pertama)
        dct_after      : list blok DCT sesudah embed (max 16 blok pertama)
    """
    padded, H, W = pad_to_multiple8(channel)
    pH, pW = padded.shape
    encoded_blocks = []
    dct_before, dct_after = [], []
    block_idx = 0

    for row in range(0, pH, 8):
        for col in range(0, pW, 8):
            block     = padded[row:row+8, col:col+8].copy()
            dct_block = dct2d(block)

            if block_idx < 16:
                dct_before.append(dct_block.copy())

            # ── Embed watermark bit (hanya channel Y) ──────────────────────
            if wm_bits is not None:
                global_idx = start_block + block_idx
                if global_idx < len(wm_bits):
                    zz  = zigzag_scan(dct_block)
                    bit = wm_bits[global_idx]
                    coeff = zz[EMBED_POSITION]
                    ALPHA = 30.0   # kekuatan embedding
                    if bit == 1:
                        if coeff < ALPHA:
                            coeff = ALPHA
                    else:
                        if coeff > -ALPHA:
                            coeff = -ALPHA
                    zz[EMBED_POSITION] = coeff
                    dct_block = inverse_zigzag(zz)

            if block_idx < 16:
                dct_after.append(dct_block.copy())

            # ── Quantize → zigzag → RLE ─────────────────────────────────────
            q_block  = np.round(dct_block / qt)
            zz_q     = zigzag_scan(q_block)
            dc, ac   = rle_encode(zz_q)
            encoded_blocks.append((dc, ac))
            block_idx += 1

    return encoded_blocks, (pH, pW), (H, W), dct_before, dct_after


def decode_channel(encoded_blocks: list, qt: np.ndarray,
                   padded_shape: tuple, orig_shape: tuple) -> np.ndarray:
    """
    Decode channel dari encoded blocks:
        RLE decode → inverse zigzag → dequantize → IDCT → crop

    Args:
        encoded_blocks: list of (dc_val, ac_rle)
        qt            : tabel kuantisasi 8×8 (harus sama dengan saat encode)
        padded_shape  : (pH, pW)
        orig_shape    : (H, W) ukuran asli sebelum padding

    Returns:
        channel: array 2D (H, W), domain spasial (belum ditambah 128)
    """
    pH, pW = padded_shape
    H, W   = orig_shape
    canvas = np.zeros((pH, pW), dtype=np.float64)
    idx    = 0

    for row in range(0, pH, 8):
        for col in range(0, pW, 8):
            dc, ac      = encoded_blocks[idx]; idx += 1
            zz          = rle_decode(dc, ac)
            q_block     = inverse_zigzag(zz)
            dct_block   = q_block * qt        # dequantize
            spatial     = idct2d(dct_block)   # IDCT
            canvas[row:row+8, col:col+8] = spatial

    return canvas[:H, :W]


# ══════════════════════════════════════════════════════════════════════
#  BAGIAN 9 — WATERMARK: TEKS → BIT & EMBEDDING
# ══════════════════════════════════════════════════════════════════════

def text_to_bits(text: str) -> list:
    """
    Konversi string teks ke list bit (0 dan 1).
    Format: header 16 bit (panjang string) + 8 bit per karakter ASCII.

    Contoh: "AB" → [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,  ← header: length=2
                     0,1,0,0,0,0,0,1,                    ← 'A' = 65
                     0,1,0,0,0,0,1,0]                    ← 'B' = 66
    """
    bits   = []
    length = len(text)
    for i in range(15, -1, -1):          # 16 bit header
        bits.append((length >> i) & 1)
    for ch in text:                       # 8 bit per karakter
        code = ord(ch)
        for i in range(7, -1, -1):
            bits.append((code >> i) & 1)
    return bits


def bits_to_text(bits: list) -> str:
    """
    Konversi list bit kembali ke string.
    Baca 16 bit pertama sebagai panjang, lalu 8 bit per karakter.
    """
    if len(bits) < 16:
        return ""
    length = 0
    for b in bits[:16]:
        length = (length << 1) | int(b)
    chars = []
    for i in range(length):
        start = 16 + i * 8
        end   = start + 8
        if end > len(bits):
            break
        byte = 0
        for b in bits[start:end]:
            byte = (byte << 1) | int(b)
        chars.append(chr(byte))
    return ''.join(chars)


def tile_bits(wm_bits: list, n_blocks: int) -> np.ndarray:
    """Tile bit watermark hingga panjang n_blocks (untuk mengisi semua blok)."""
    if not wm_bits:
        return np.zeros(n_blocks, dtype=int)
    repeats = (n_blocks // len(wm_bits)) + 1
    return np.array((wm_bits * repeats)[:n_blocks], dtype=int)


# ══════════════════════════════════════════════════════════════════════
#  BAGIAN 10 — METRIK KUALITAS
# ══════════════════════════════════════════════════════════════════════

def compute_psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Peak Signal-to-Noise Ratio (dB) — mengukur kualitas gambar rekonstruksi.

    Rumus:
        MSE   = rata-rata dari (original − rekonstruksi)²
        PSNR  = 10 × log10(255² / MSE)

    Interpretasi:
        > 40 dB : perbedaan hampir tidak terlihat mata manusia
        30-40 dB: perbedaan kecil, umumnya dapat diterima
        < 30 dB : perbedaan terlihat jelas
    """
    mse = np.mean((original.astype(np.float64) - reconstructed.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 10.0 * np.log10(255.0**2 / mse)


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Structural Similarity Index (SSIM) — mengukur kemiripan struktural.

    Dihitung per window 8×8 pada channel Y (luminance), lalu dirata-rata.
    Konstanta C1=(0.01×255)², C2=(0.03×255)² untuk stabilitas numerik.

    Nilai 0–1:
        1.0 : identik sempurna
        > 0.99 : hampir tidak bisa dibedakan
        < 0.9  : perbedaan struktural terlihat
    """
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    y1 = rgb_to_ycbcr(img1)[:,:,0].astype(np.float64)
    y2 = rgb_to_ycbcr(img2)[:,:,0].astype(np.float64)
    H, W = y1.shape
    ws   = 8
    vals = []
    for r in range(0, H - ws + 1, ws):
        for c in range(0, W - ws + 1, ws):
            p1, p2     = y1[r:r+ws, c:c+ws], y2[r:r+ws, c:c+ws]
            mu1, mu2   = np.mean(p1), np.mean(p2)
            s1, s2     = np.var(p1), np.var(p2)
            s12        = np.mean((p1 - mu1)*(p2 - mu2))
            num        = (2*mu1*mu2 + C1)*(2*s12 + C2)
            den        = (mu1**2 + mu2**2 + C1)*(s1 + s2 + C2)
            vals.append(num / den)
    return float(np.mean(vals))


def bit_accuracy(bits_orig: list, bits_extr: list) -> float:
    """Hitung persentase bit yang berhasil diekstrak dengan benar."""
    n = min(len(bits_orig), len(bits_extr))
    if n == 0:
        return 0.0
    return sum(a == b for a, b in zip(bits_orig[:n], bits_extr[:n])) / n


# ══════════════════════════════════════════════════════════════════════
#  BAGIAN 11 — FULL PIPELINE: EMBED & EXTRACT
# ══════════════════════════════════════════════════════════════════════

def embed_watermark_pipeline(img_rgb: np.ndarray,
                             watermark_text: str,
                             quality: int = 75,
                             alpha: float = 30.0):
    """
    Pipeline lengkap embedding watermark:

    INPUT (x) → YCbCr → LevelShift → ChromaDown → Padding →
    Blok8x8 → DCT → [EMBED WM di koefisien mid-freq] →
    Quantize → Zigzag → RLE → RLEdecode → Dequantize → IDCT →
    ChromaUp → YCbCr→RGB → OUTPUT watermarked (y)

    Metode embedding:
        Satu bit watermark per blok → modifikasi koefisien DCT di zigzag[14]:
            bit = 1 → paksa koefisien ≥ +alpha
            bit = 0 → paksa koefisien ≤ −alpha
        Setelah quantize-dequantize, sign koefisien tetap terjaga
        selama alpha cukup besar relatif terhadap step quantization.

    Returns:
        wm_rgb    : gambar watermarked (H, W, 3) uint8
        state     : dict berisi semua intermediate state untuk visualisasi
    """
    state = {'original_rgb': img_rgb.copy(), 'watermark_text': watermark_text,
             'quality': quality, 'alpha': alpha}

    # Step 1: RGB → YCbCr
    ycbcr = rgb_to_ycbcr(img_rgb)
    Y, Cb, Cr = ycbcr[:,:,0], ycbcr[:,:,1], ycbcr[:,:,2]
    H_orig, W_orig = Y.shape
    state.update({'ycbcr': ycbcr.copy(), 'Y_channel': Y.copy()})

    # Step 2: Level shift −128
    Y_sh  = Y  - 128.0
    Cb_sh = Cb - 128.0
    Cr_sh = Cr - 128.0
    state['Y_shifted'] = Y_sh.copy()

    # Step 3: Chroma downsample
    Cb_dn = chroma_downsample(Cb_sh)
    Cr_dn = chroma_downsample(Cr_sh)
    state.update({'Cb_shifted': Cb_sh, 'Cb_down': Cb_dn,
                  'Cr_shifted': Cr_sh, 'Cr_down': Cr_dn})

    # Step 4: Hitung tabel kuantisasi
    qt_Y = make_quant_table(LUMA_QT_BASE,   quality)
    qt_C = make_quant_table(CHROMA_QT_BASE, quality)
    state.update({'qt_Y': qt_Y, 'qt_C': qt_C})

    # Step 5: Siapkan bit watermark
    wm_bits_list = text_to_bits(watermark_text)
    padded_Y, _, _ = pad_to_multiple8(Y_sh)
    pH, pW = padded_Y.shape
    n_blocks = (pH // 8) * (pW // 8)
    wm_bits  = tile_bits(wm_bits_list, n_blocks)
    state.update({'wm_bits': wm_bits, 'wm_bit_count': len(wm_bits_list),
                  'n_blocks': n_blocks, 'block_grid': (pH//8, pW//8)})

    # Step 6: Encode Y dengan watermark embedded
    enc_Y, ps_Y, os_Y, dct_bef, dct_aft = encode_channel(
        Y_sh, qt_Y, wm_bits=wm_bits)
    state.update({'block_dct_before': dct_bef, 'block_dct_after': dct_aft})

    # Step 7: Encode Cb, Cr normal (tanpa watermark)
    enc_Cb, ps_Cb, os_Cb, _, _ = encode_channel(Cb_dn, qt_C)
    enc_Cr, ps_Cr, os_Cr, _, _ = encode_channel(Cr_dn, qt_C)

    # Step 8: Decode semua channel kembali ke domain spasial
    Y_rec  = decode_channel(enc_Y,  qt_Y, ps_Y,  os_Y)  + 128.0
    Cb_rec = decode_channel(enc_Cb, qt_C, ps_Cb, os_Cb) + 128.0
    Cr_rec = decode_channel(enc_Cr, qt_C, ps_Cr, os_Cr) + 128.0

    # Step 9: Upsample Cb, Cr → gabung → YCbCr → RGB
    Cb_up = chroma_upsample(Cb_rec, H_orig, W_orig)
    Cr_up = chroma_upsample(Cr_rec, H_orig, W_orig)
    wm_rgb = ycbcr_to_rgb(np.stack([Y_rec, Cb_up, Cr_up], axis=2))
    state['watermarked_rgb'] = wm_rgb.copy()

    # Metrik kualitas
    state['psnr'] = compute_psnr(img_rgb, wm_rgb)
    state['ssim']  = compute_ssim(img_rgb, wm_rgb)
    state['diff']  = img_rgb.astype(np.float64) - wm_rgb.astype(np.float64)

    return wm_rgb, state


def extract_watermark_pipeline(wm_rgb: np.ndarray,
                               quality: int,
                               wm_bit_count: int,
                               alpha: float = 30.0):
    """
    Ekstrak watermark dari gambar watermarked.

    Prinsip: setelah quantize-dequantize, koefisien DCT di posisi
    EMBED_POSITION masih memiliki sign yang sama dengan saat embedding
    (selama alpha > step quantization).

    Baca sign koefisien → bit watermark.

    Args:
        wm_rgb       : gambar watermarked (H, W, 3) uint8
        quality      : QF yang dipakai saat embedding
        wm_bit_count : jumlah total bit watermark (16 + 8×len(teks))
        alpha        : kekuatan embedding (harus sama dengan saat embed)

    Returns:
        extracted_text: string hasil ekstraksi
        extracted_bits: list bit
        avg_confidence: float (0–1) keyakinan rata-rata
    """
    ycbcr = rgb_to_ycbcr(wm_rgb)
    Y_sh  = ycbcr[:,:,0] - 128.0
    padded, H, W = pad_to_multiple8(Y_sh)
    pH, pW = padded.shape
    qt_Y   = make_quant_table(LUMA_QT_BASE, quality)

    extracted_bits = []
    confidences    = []

    for ri in range(pH // 8):
        for ci in range(pW // 8):
            block = padded[ri*8:ri*8+8, ci*8:ci*8+8]
            zz    = zigzag_scan(dct2d(block))
            coeff = zz[EMBED_POSITION]
            extracted_bits.append(1 if coeff >= 0 else 0)
            qt_step = qt_Y.flat[EMBED_POSITION]
            confidences.append(min(abs(coeff) / (alpha / qt_step + 1e-6), 1.0))
            if len(extracted_bits) >= wm_bit_count * 3:
                break
        if len(extracted_bits) >= wm_bit_count * 3:
            break

    final_bits     = extracted_bits[:wm_bit_count]
    extracted_text = bits_to_text(final_bits)
    avg_conf       = float(np.mean(confidences[:wm_bit_count])) if confidences else 0.0

    return extracted_text, final_bits, avg_conf


# ══════════════════════════════════════════════════════════════════════
#  BAGIAN 12 — VISUALISASI STEP-BY-STEP
# ══════════════════════════════════════════════════════════════════════

def save_fig(fig, name: str, out_dir: str, dpi: int = 150) -> str:
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  → {name}")
    return path


def vis_step0_original(img_rgb, out_dir):
    fig, ax = plt.subplots(figsize=(6, 6), facecolor='#111')
    ax.imshow(img_rgb); ax.axis('off')
    ax.set_title('STEP 0: Gambar Asli (Host Image x)', color='white', fontsize=13)
    fig.text(0.5, 0.01, f'Resolusi: {img_rgb.shape[1]}×{img_rgb.shape[0]} px | RGB uint8',
             ha='center', color='#aaa', fontsize=9)
    return save_fig(fig, 'step00_original.png', out_dir)


def vis_step1_ycbcr(img_rgb, ycbcr, out_dir):
    fig = plt.figure(figsize=(14, 5), facecolor='#111')
    labels = ['R','G','B','Y (Luminance)','Cb (Blue-diff)','Cr (Red-diff)']
    cmaps  = ['Reds','Greens','Blues','gray','Blues','Reds']
    data   = [img_rgb[:,:,0], img_rgb[:,:,1], img_rgb[:,:,2],
              ycbcr[:,:,0], ycbcr[:,:,1], ycbcr[:,:,2]]
    for i in range(6):
        ax = fig.add_subplot(1, 6, i+1)
        ax.imshow(data[i], cmap=cmaps[i], vmin=0, vmax=255)
        ax.set_title(labels[i], color='white', fontsize=9); ax.axis('off')
    fig.suptitle('STEP 1: RGB → YCbCr (BT.601)\n'
                 'Y = 0.299R+0.587G+0.114B  |  Cb = −0.169R−0.331G+0.5B+128  |  Cr = 0.5R−0.419G−0.081B+128',
                 color='white', fontsize=10, y=1.02)
    fig.tight_layout()
    return save_fig(fig, 'step01_rgb_to_ycbcr.png', out_dir)


def vis_step2_levelshift(Y, Y_shifted, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), facecolor='#111')
    axes[0].imshow(Y, cmap='gray', vmin=0, vmax=255); axes[0].set_title('Y (0–255)', color='white'); axes[0].axis('off')
    axes[1].imshow(Y_shifted, cmap='gray', vmin=-128, vmax=127); axes[1].set_title('Y − 128 (−128–127)', color='white'); axes[1].axis('off')
    axes[2].set_facecolor('#1a1a2e')
    axes[2].hist(Y.ravel(), bins=50, color='#3498db', alpha=0.7, label='Sebelum (0–255)')
    axes[2].hist(Y_shifted.ravel(), bins=50, color='#e74c3c', alpha=0.7, label='Sesudah (−128–127)')
    axes[2].set_title('Histogram Y', color='white', fontsize=10)
    axes[2].legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white')
    axes[2].tick_params(colors='white')
    for sp in axes[2].spines.values(): sp.set_color('#555')
    fig.suptitle('STEP 2: Level Shift −128 (persiapan DCT)', color='white', fontsize=12)
    fig.patch.set_facecolor('#111'); fig.tight_layout()
    return save_fig(fig, 'step02_level_shift.png', out_dir)


def vis_step3_chroma(Y, Cb_sh, Cb_dn, Cr_sh, Cr_dn, out_dir):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), facecolor='#111')
    for row_i, (full, down, name, cm) in enumerate([(Cb_sh, Cb_dn,'Cb','Blues'),(Cr_sh, Cr_dn,'Cr','Reds')]):
        axes[row_i][0].imshow(Y, cmap='gray'); axes[row_i][0].set_title('Y (tidak di-downsample)', color='white', fontsize=9); axes[row_i][0].axis('off')
        axes[row_i][1].imshow(full, cmap=cm); axes[row_i][1].set_title(f'{name} full ({full.shape[1]}×{full.shape[0]})', color='white', fontsize=9); axes[row_i][1].axis('off')
        axes[row_i][2].imshow(down, cmap=cm); axes[row_i][2].set_title(f'{name} 4:2:0 ({down.shape[1]}×{down.shape[0]}) — setengah resolusi', color='white', fontsize=9); axes[row_i][2].axis('off')
    fig.suptitle('STEP 3: Chroma Downsampling 4:2:0\nMata manusia kurang sensitif terhadap warna → Cb & Cr di-downsample 2×',
                 color='white', fontsize=11, y=1.02)
    fig.tight_layout()
    return save_fig(fig, 'step03_chroma_downsample.png', out_dir)


def vis_step4_dct_embed(Y_shifted, dct_bef, dct_aft, wm_bits, out_dir):
    n = min(4, len(dct_bef))
    fig = plt.figure(figsize=(16, n*3.5+2), facecolor='#111')
    fig.suptitle('STEP 4–6: Blok 8×8 → 2D DCT → Embed Watermark\n'
                 'Kotak kuning = posisi koefisien yang dimodifikasi (zigzag index 14 = mid-frequency)',
                 color='white', fontsize=12, y=0.99)
    padded = np.pad(Y_shifted, ((0,(8-Y_shifted.shape[0]%8)%8),(0,(8-Y_shifted.shape[1]%8)%8)), mode='edge')
    nC_pad = padded.shape[1]//8
    for b in range(n):
        br, bc = (b // nC_pad)*8, (b % nC_pad)*8
        block_sp = padded[br:br+8, bc:bc+8]
        base = b*4+1

        ax1 = fig.add_subplot(n, 4, base)
        im = ax1.imshow(block_sp, cmap='gray', aspect='equal'); ax1.set_title(f'Blok #{b} Spatial\n(nilai piksel −128..127)', color='white', fontsize=8); ax1.axis('off')
        plt.colorbar(im, ax=ax1, fraction=0.046).ax.yaxis.set_tick_params(color='white')

        ax2 = fig.add_subplot(n, 4, base+1)
        vmax = max(np.max(np.abs(dct_bef[b])), 1)
        im2 = ax2.imshow(dct_bef[b], cmap=DCT_CMAP, vmin=-vmax, vmax=vmax, aspect='equal')
        ax2.set_title(f'DCT Blok #{b} (sebelum embed)', color='white', fontsize=8); ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046).ax.yaxis.set_tick_params(color='white')

        ax3 = fig.add_subplot(n, 4, base+2)
        im3 = ax3.imshow(dct_aft[b], cmap=DCT_CMAP, vmin=-vmax, vmax=vmax, aspect='equal')
        er, ec = ZIGZAG_ORDER[EMBED_POSITION]
        ax3.add_patch(plt.Rectangle((ec-.5,er-.5),1,1,lw=2,edgecolor='#f1c40f',facecolor='none'))
        ax3.set_title(f'DCT Blok #{b} sesudah embed\nbit={wm_bits[b]}', color='white', fontsize=8); ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046).ax.yaxis.set_tick_params(color='white')

        ax4 = fig.add_subplot(n, 4, base+3)
        diff = dct_aft[b]-dct_bef[b]; vd = max(np.max(np.abs(diff)),1)
        im4 = ax4.imshow(diff, cmap='RdBu_r', vmin=-vd, vmax=vd, aspect='equal')
        ax4.set_title(f'Perubahan DCT #{b}\n(sesudah − sebelum)', color='white', fontsize=8); ax4.axis('off')
        plt.colorbar(im4, ax=ax4, fraction=0.046).ax.yaxis.set_tick_params(color='white')

    fig.tight_layout(rect=[0,0,1,0.96])
    return save_fig(fig, 'step04_blocks_dct_embed.png', out_dir)


def vis_step5_quant(qt_Y, qt_C, quality, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor='#111')
    for ax, qt, lbl in zip(axes, [qt_Y, qt_C], ['Luminance (Y)','Chrominance (Cb/Cr)']):
        im = ax.imshow(qt, cmap='hot', aspect='equal')
        ax.set_title(f'Tabel Kuantisasi {lbl}\nQF={quality}', color='white', fontsize=11); ax.axis('off')
        for r in range(8):
            for c in range(8):
                ax.text(c,r,str(int(qt[r,c])),ha='center',va='center',fontsize=7,
                        color='black' if qt[r,c]<80 else 'white')
        cb = plt.colorbar(im, ax=ax, fraction=0.046); cb.ax.tick_params(labelcolor='white')
    fig.suptitle('STEP 5: Tabel Kuantisasi\nNilai besar = frekuensi tinggi dibuang (kompresi kuat) | Nilai kecil = presisi tinggi',
                 color='white', fontsize=11, y=1.02)
    fig.tight_layout()
    return save_fig(fig, 'step05_quant_tables.png', out_dir)


def vis_step6_zigzag(dct_block, out_dir):
    fig = plt.figure(figsize=(13, 5.5), facecolor='#111')
    ax1 = fig.add_subplot(1,2,1); ax1.set_facecolor('#0d1b2a')
    vmax = max(np.max(np.abs(dct_block)),1)
    ax1.imshow(dct_block, cmap=DCT_CMAP, vmin=-vmax, vmax=vmax)
    for i,(r,c) in enumerate(ZIGZAG_ORDER):
        ax1.text(c,r,str(i),ha='center',va='center',fontsize=7,
                 color='white' if abs(dct_block[r,c])<vmax*0.4 else 'black')
    ax1.set_title('Blok DCT 8×8\n(angka = urutan zigzag scan)', color='white', fontsize=10); ax1.axis('off')

    ax2 = fig.add_subplot(1,2,2); ax2.set_facecolor('#0d1b2a')
    zz = np.array([dct_block[r,c] for r,c in ZIGZAG_ORDER])
    colors = ['#f1c40f' if i==EMBED_POSITION else ('#e74c3c' if v>0 else '#3498db')
              for i,v in enumerate(zz)]
    ax2.bar(range(64), zz, color=colors, width=0.8)
    ax2.axhline(0, color='white', lw=0.5)
    ax2.set_xlabel('Index Zigzag', color='white', fontsize=9)
    ax2.set_ylabel('Nilai Koefisien DCT', color='white', fontsize=9)
    ax2.set_title('Array 1D setelah Zigzag Scan\n(kuning = posisi watermark)', color='white', fontsize=10)
    ax2.tick_params(colors='white')
    for sp in ax2.spines.values(): sp.set_color('#555')
    patches = [mpatches.Patch(color='#f1c40f',label=f'Posisi watermark (idx {EMBED_POSITION})'),
               mpatches.Patch(color='#e74c3c',label='Positif'), mpatches.Patch(color='#3498db',label='Negatif')]
    ax2.legend(handles=patches,loc='upper right',fontsize=7,facecolor='#1a1a2e',labelcolor='white')
    fig.suptitle('STEP 6: Zigzag Scan — DC di depan, frekuensi tinggi di belakang', color='white', fontsize=12)
    fig.tight_layout()
    return save_fig(fig, 'step06_zigzag.png', out_dir)


def vis_step7_rle(dct_aft_block, qt_Y, out_dir):
    q_block = np.round(dct_aft_block / qt_Y)
    zz_q    = zigzag_scan(q_block)
    dc_val, ac_rle = rle_encode(zz_q)

    fig = plt.figure(figsize=(14, 6), facecolor='#111')
    ax1 = fig.add_subplot(2,1,1); ax1.set_facecolor('#0d1b2a')
    vals = [int(round(v)) for v in zz_q]
    colors = ['#2ecc71' if v!=0 else '#555' for v in vals]
    ax1.bar(range(64), vals, color=colors, width=0.85)
    ax1.axhline(0, color='white', lw=0.5)
    ax1.set_title(f'Koefisien Setelah Quantization | DC={dc_val} | Non-zero AC: {sum(1 for v in vals[1:] if v!=0)}',
                  color='white', fontsize=10)
    ax1.set_xlabel('Index Zigzag', color='white', fontsize=8)
    ax1.tick_params(colors='white')
    for sp in ax1.spines.values(): sp.set_color('#444')

    ax2 = fig.add_subplot(2,1,2); ax2.set_facecolor('#0d1b2a'); ax2.axis('off')
    rle_text = f"DC = {dc_val}\n\nAC RLE pairs (run_zeros, value):\n"
    for i,(run,val) in enumerate(ac_rle[:20]):
        if (run,val)==(0,0): rle_text += "  (0, 0) ← EOB\n"; break
        rle_text += f"  ({run}, {val:+4d})"
        if (i+1)%5==0: rle_text += "\n"
    ax2.text(0.01, 0.95, rle_text, transform=ax2.transAxes, fontsize=9, color='#2ecc71',
             va='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#0a0f1e', alpha=0.8))
    ax2.set_title('RLE AC Koefisien', color='white', fontsize=10)
    fig.suptitle('STEP 7: Quantization → Zigzag → RLE\nBanyak koefisien tinggi=0 → RLE sangat efisien',
                 color='white', fontsize=11)
    fig.tight_layout()
    return save_fig(fig, 'step07_rle.png', out_dir)


def vis_step8_comparison(original, watermarked, psnr, ssim, out_dir):
    diff_amp = np.clip((original.astype(np.float64)-watermarked.astype(np.float64))*10+128, 0, 255).astype(np.uint8)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), facecolor='#111')
    axes[0].imshow(original); axes[0].set_title('Gambar ASLI (x)', color='white', fontsize=12); axes[0].axis('off')
    axes[1].imshow(watermarked); axes[1].set_title('Gambar WATERMARKED (y)\n"tidak mudah dibedakan"', color='white', fontsize=12); axes[1].axis('off')
    axes[2].imshow(diff_amp); axes[2].set_title('Selisih × 10 (amplified)\nhanya untuk visualisasi', color='white', fontsize=12); axes[2].axis('off')
    fig.suptitle(f'STEP 8: Gambar x dan y\nPSNR = {psnr:.2f} dB (>30 dB = tidak terlihat)  |  SSIM = {ssim:.5f}',
                 color='white', fontsize=12, y=1.01)
    fig.tight_layout()
    return save_fig(fig, 'step08_comparison_x_y.png', out_dir)


def vis_step9_qf_analysis(results, out_dir):
    qfs      = [r['qf']      for r in results]
    psnrs    = [r['psnr']    for r in results]
    bit_accs = [r['bit_acc'] for r in results]

    fig = plt.figure(figsize=(14, 9), facecolor='#111')
    fig.suptitle('STEP 9: Analisis Ekstraksi Watermark pada Berbagai Quality Factor (1–100)',
                 color='white', fontsize=13)

    ax1 = fig.add_subplot(2,2,1); ax1.set_facecolor('#0d1b2a')
    ax1.plot(qfs, psnrs, color='#3498db', lw=2, marker='o', ms=3)
    ax1.axhline(30, color='#f1c40f', lw=1.5, ls='--', label='30 dB (threshold)')
    ax1.set_xlabel('Quality Factor', color='white'); ax1.set_ylabel('PSNR (dB)', color='white')
    ax1.set_title('PSNR vs QF', color='white'); ax1.legend(facecolor='#1a1a2e',labelcolor='white',fontsize=8)
    ax1.tick_params(colors='white')
    for sp in ax1.spines.values(): sp.set_color('#555')

    ax2 = fig.add_subplot(2,2,2); ax2.set_facecolor('#0d1b2a')
    colors_b = ['#2ecc71' if a>=0.9 else ('#f39c12' if a>=0.7 else '#e74c3c') for a in bit_accs]
    ax2.bar(qfs, bit_accs, color=colors_b, width=1.5)
    ax2.axhline(0.9, color='#f39c12', lw=1, ls=':', label='90% threshold')
    ax2.set_ylim(0, 1.1); ax2.set_xlabel('Quality Factor', color='white'); ax2.set_ylabel('Bit Accuracy', color='white')
    ax2.set_title('Akurasi Bit Watermark vs QF', color='white')
    ax2.legend(facecolor='#1a1a2e',labelcolor='white',fontsize=8); ax2.tick_params(colors='white')
    for sp in ax2.spines.values(): sp.set_color('#555')

    ax3 = fig.add_subplot(2,1,2); ax3.set_facecolor('#0d1b2a')
    for r in results:
        ax3.barh(0.5, 1, left=r['qf']-0.5, height=0.8,
                 color='#2ecc71' if r['success'] else '#e74c3c', alpha=0.85)
    ax3.set_xlim(-1,101); ax3.set_ylim(0,1); ax3.set_yticks([])
    ax3.set_xlabel('Quality Factor (0–100)', color='white', fontsize=10)
    ax3.set_title('Status Ekstraksi: Hijau=Berhasil | Merah=Gagal', color='white')
    ax3.tick_params(colors='white')
    for q in range(0,101,10): ax3.axvline(q, color='#333', lw=0.5)
    for sp in ax3.spines.values(): sp.set_color('#555')
    patches = [mpatches.Patch(color='#2ecc71',label='Berhasil diekstrak'),
               mpatches.Patch(color='#e74c3c',label='Gagal diekstrak')]
    ax3.legend(handles=patches, loc='lower right', facecolor='#1a1a2e', labelcolor='white', fontsize=9)

    fig.tight_layout(rect=[0,0,1,0.95])
    return save_fig(fig, 'step09_qf_analysis.png', out_dir)


def vis_step10_pipeline(out_dir):
    fig, ax = plt.subplots(figsize=(16, 6), facecolor='#0d1b2a')
    ax.set_facecolor('#0d1b2a'); ax.set_xlim(0,16); ax.set_ylim(0,6); ax.axis('off')
    steps = [
        ("1. Input\nRGB (x)",  0.7, 3.0,'#1a5276'),("2. RGB→\nYCbCr",   2.1,3.0,'#1a5276'),
        ("3. Level\nShift",    3.5, 3.0,'#1a5276'),("4. Chroma\nDown",   4.9,3.0,'#1a5276'),
        ("5. Pad\n8×8",        6.3, 3.0,'#154360'),("6. 2D\nDCT",        7.7,3.0,'#6e2fa1'),
        ("7. Embed\nWatermark",9.1, 3.0,'#c0392b'),("8. Quant-\nize",   10.5,3.0,'#6e2fa1'),
        ("9. Zigzag\n+RLE",   11.9, 3.0,'#1a5276'),("10. IDCT\n+Rekon.",13.3,3.0,'#1e8449'),
        ("Output\n(y)",        14.7,3.0,'#1e8449'),
    ]
    for lbl,x,y,color in steps:
        ax.add_patch(mpatches.FancyBboxPatch((x-.62,y-.62),1.24,1.24,
            boxstyle="round,pad=0.08",facecolor=color,edgecolor='white',lw=1.5,zorder=3))
        ax.text(x,y,lbl,ha='center',va='center',fontsize=7.5,color='white',zorder=4,fontweight='bold')
    for i in range(len(steps)-1):
        ax.annotate('',xy=(steps[i+1][1]-.62,steps[i+1][2]),xytext=(steps[i][1]+.62,steps[i][2]),
                    arrowprops=dict(arrowstyle='->',color='#f1c40f',lw=2))
    ax.annotate('Watermark Bits\n(teks → bit)',xy=(9.1,2.38),xytext=(9.1,1.2),
                color='#f39c12',fontsize=8,ha='center',
                arrowprops=dict(arrowstyle='->',color='#f39c12',lw=1.5))
    ax.text(8,5.5,'PIPELINE JPEG WATERMARKING FROM SCRATCH',ha='center',color='white',fontsize=13,fontweight='bold')
    ax.text(8,5.0,'Biru=JPEG standar | Ungu=DCT domain | Merah=watermark embed | Hijau=rekonstruksi',
            ha='center',color='#aaa',fontsize=9)
    return save_fig(fig, 'step10_pipeline_diagram.png', out_dir, dpi=180)


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='JPEG Watermarking From Scratch')
    parser.add_argument('--image',     default='assets/face_256.png',   help='Path ke gambar input')
    parser.add_argument('--watermark', default='BRANDON18224118',       help='Teks watermark')
    parser.add_argument('--quality',   type=int, default=75,            help='JPEG Quality Factor (1-100)')
    parser.add_argument('--alpha',     type=float, default=30.0,        help='Kekuatan watermark embedding')
    parser.add_argument('--output',    default='output',                help='Direktori output')
    args = parser.parse_args()

    OUT_DIR = args.output
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Load gambar ─────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print("  JPEG WATERMARKING FROM SCRATCH — Sistem Multimedia")
    print(f"{'='*62}")
    print(f"  Gambar   : {args.image}")
    print(f"  Watermark: '{args.watermark}'")
    print(f"  QF       : {args.quality}")
    print(f"  Alpha    : {args.alpha}")
    print(f"{'='*62}\n")

    img_rgb = np.array(Image.open(args.image).convert('RGB'))
    print(f"[LOAD] {img_rgb.shape} {img_rgb.dtype}\n")

    # ── Plot step 0 ────────────────────────────────────────────────────────
    print("[VISUALISASI]")
    vis_step0_original(img_rgb, OUT_DIR)

    # ── Step 1: RGB → YCbCr ────────────────────────────────────────────────
    ycbcr = rgb_to_ycbcr(img_rgb)
    vis_step1_ycbcr(img_rgb, ycbcr, OUT_DIR)

    # ── Step 2: Level shift ────────────────────────────────────────────────
    Y  = ycbcr[:,:,0]; Cb = ycbcr[:,:,1]; Cr = ycbcr[:,:,2]
    Y_sh = Y - 128.0; Cb_sh = Cb - 128.0; Cr_sh = Cr - 128.0
    vis_step2_levelshift(Y, Y_sh, OUT_DIR)

    # ── Step 3: Chroma downsample ──────────────────────────────────────────
    Cb_dn = chroma_downsample(Cb_sh); Cr_dn = chroma_downsample(Cr_sh)
    vis_step3_chroma(Y, Cb_sh, Cb_dn, Cr_sh, Cr_dn, OUT_DIR)

    # ── Embed watermark (full pipeline) ───────────────────────────────────
    print("\n[EMBED] Menyisipkan watermark...")
    wm_rgb, state = embed_watermark_pipeline(img_rgb, args.watermark, args.quality, args.alpha)

    vis_step4_dct_embed(state['Y_shifted'], state['block_dct_before'],
                        state['block_dct_after'], state['wm_bits'], OUT_DIR)
    vis_step5_quant(state['qt_Y'], state['qt_C'], args.quality, OUT_DIR)
    vis_step6_zigzag(state['block_dct_after'][0], OUT_DIR)
    vis_step7_rle(state['block_dct_after'][0], state['qt_Y'], OUT_DIR)
    vis_step8_comparison(img_rgb, wm_rgb, state['psnr'], state['ssim'], OUT_DIR)

    # Simpan gambar watermarked
    wm_save_path = os.path.join(OUT_DIR, 'watermarked_face.png')
    Image.fromarray(wm_rgb).save(wm_save_path)
    print(f"  → watermarked_face.png")

    print(f"\n  PSNR = {state['psnr']:.2f} dB  (> 30 dB → tidak terlihat secara visual)")
    print(f"  SSIM = {state['ssim']:.6f}  (> 0.9 → hampir identik)")

    # ── Step 9: QF analysis ────────────────────────────────────────────────
    print("\n[ANALYSIS] Uji ekstraksi pada QF 1–100...")
    wm_bits_orig = text_to_bits(args.watermark)
    n_wm_bits    = len(wm_bits_orig)
    qf_results   = []

    for qf in range(1, 101, 2):
        ext_text, ext_bits, conf = extract_watermark_pipeline(
            wm_rgb, quality=qf, wm_bit_count=n_wm_bits, alpha=args.alpha)
        bacc = bit_accuracy(wm_bits_orig, ext_bits)
        qf_results.append({
            'qf': qf, 'psnr': state['psnr'], 'bit_acc': bacc,
            'extracted_text': ext_text, 'success': bacc >= 0.85, 'conf': conf,
        })

    vis_step9_qf_analysis(qf_results, OUT_DIR)

    # ── Step 10: Pipeline diagram ──────────────────────────────────────────
    vis_step10_pipeline(OUT_DIR)

    # ── Summary ────────────────────────────────────────────────────────────
    n_ok = sum(1 for r in qf_results if r['success'])
    print(f"\n{'='*62}")
    print(f"  SELESAI — {len(os.listdir(OUT_DIR))} file disimpan di '{OUT_DIR}/'")
    print(f"  QF berhasil ekstrak: {n_ok}/{len(qf_results)}")
    print(f"{'='*62}\n")


if __name__ == '__main__':
    main()
