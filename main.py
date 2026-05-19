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
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg') # Mode ini tetap dipakai agar gambar tersimpan dengan rapi tanpa error
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from scipy.fft import dct, idct


# ══════════════════════════════════════════════════════════════════════
#  FUNGSI MODE INTERAKTIF (POP-UP & JEDA)
# ══════════════════════════════════════════════════════════════════════
def interaktif_jeda(image_path: str, next_step_name: str = None):
    """
    Fungsi untuk memunculkan gambar ke layar (pop-up) menggunakan viewer
    bawaan OS (seperti Windows Photos), lalu meminta persetujuan Y/N di terminal.
    """
    try:
        # Munculkan pop-up gambar
        img = Image.open(image_path)
        img.show()
    except Exception as e:
        print(f"  [!] Gagal memunculkan pop-up gambar otomatis: {e}")

    if next_step_name:
        print("-" * 62)
        jawab = input(f"  [?] Lanjut ke {next_step_name}? (y/n): ").strip().lower()
        if jawab == 'n':
            print("\n  [!] Proses dihentikan. Sampai jumpa!\n")
            sys.exit(0)
        print("-" * 62)


# ══════════════════════════════════════════════════════════════════════
#  BAGIAN 1 — KONSTANTA & TABEL STANDAR JPEG
# ══════════════════════════════════════════════════════════════════════
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

EMBED_POSITION = 14
DCT_CMAP = LinearSegmentedColormap.from_list('dct', ['#0d1b2a','#1b4f72','#2980b9','#f0f0f0','#e67e22','#c0392b','#641e16'])

# ══════════════════════════════════════════════════════════════════════
#  BAGIAN 2 — KONVERSI WARNA
# ══════════════════════════════════════════════════════════════════════
def rgb_to_ycbcr(img_rgb: np.ndarray) -> np.ndarray:
    img = img_rgb.astype(np.float64)
    R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
    Y  =  0.299    * R + 0.587    * G + 0.114    * B
    Cb = -0.168736 * R - 0.331264 * G + 0.5      * B + 128.0
    Cr =  0.5      * R - 0.418688 * G - 0.081312 * B + 128.0
    return np.stack([Y, Cb, Cr], axis=2)

def ycbcr_to_rgb(img_ycbcr: np.ndarray) -> np.ndarray:
    Y, Cb, Cr = img_ycbcr[:,:,0], img_ycbcr[:,:,1], img_ycbcr[:,:,2]
    R = Y + 1.402    * (Cr - 128.0)
    G = Y - 0.344136 * (Cb - 128.0) - 0.714136 * (Cr - 128.0)
    B = Y + 1.772    * (Cb - 128.0)
    return np.clip(np.stack([R, G, B], axis=2), 0, 255).astype(np.uint8)

# ══════════════════════════════════════════════════════════════════════
#  BAGIAN 3 — CHROMA DOWNSAMPLING & UPSAMPLING
# ══════════════════════════════════════════════════════════════════════
def chroma_downsample(channel: np.ndarray) -> np.ndarray:
    H2, W2 = channel.shape[0] // 2, channel.shape[1] // 2
    out = (channel[0::2, 0::2] + channel[1::2, 0::2] +
           channel[0::2, 1::2] + channel[1::2, 1::2]) / 4.0
    return out[:H2, :W2]

def chroma_upsample(channel: np.ndarray, target_H: int, target_W: int) -> np.ndarray:
    up = np.repeat(np.repeat(channel, 2, axis=0), 2, axis=1)
    return up[:target_H, :target_W]

# ══════════════════════════════════════════════════════════════════════
#  BAGIAN 4 — TABEL KUANTISASI
# ══════════════════════════════════════════════════════════════════════
def make_quant_table(base_table: np.ndarray, quality: int) -> np.ndarray:
    quality = int(np.clip(quality, 1, 100))
    scale   = 5000.0 / quality if quality < 50 else 200.0 - 2.0 * quality
    qt      = np.clip(np.floor(base_table * scale / 100.0 + 0.5), 1, 255)
    return qt.astype(np.float64)

# ══════════════════════════════════════════════════════════════════════
#  BAGIAN 5 — 2D DCT / IDCT
# ══════════════════════════════════════════════════════════════════════
def dct2d(block: np.ndarray) -> np.ndarray:
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2d(block: np.ndarray) -> np.ndarray:
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

# ══════════════════════════════════════════════════════════════════════
#  BAGIAN 6 — ZIGZAG SCAN
# ══════════════════════════════════════════════════════════════════════
def zigzag_scan(block: np.ndarray) -> np.ndarray:
    return np.array([block[r, c] for r, c in ZIGZAG_ORDER])

def inverse_zigzag(vec: np.ndarray) -> np.ndarray:
    block = np.zeros((8, 8), dtype=np.float64)
    for i, (r, c) in enumerate(ZIGZAG_ORDER):
        block[r, c] = vec[i]
    return block

# ══════════════════════════════════════════════════════════════════════
#  BAGIAN 7 — RLE ENCODING / DECODING
# ══════════════════════════════════════════════════════════════════════
def rle_encode(zigzag_vec: np.ndarray):
    dc_val  = int(round(zigzag_vec[0]))
    ac_vals = [int(round(v)) for v in zigzag_vec[1:]]
    ac_rle, zero_run = [], 0
    for val in ac_vals:
        if val == 0:
            zero_run += 1
            if zero_run == 16:
                ac_rle.append((15, 0))
                zero_run = 0
        else:
            ac_rle.append((zero_run, val))
            zero_run = 0
    ac_rle.append((0, 0))
    return dc_val, ac_rle

def rle_decode(dc_val: int, ac_rle: list) -> np.ndarray:
    vec, idx = np.zeros(64, dtype=np.float64), 1
    vec[0] = dc_val
    for run, val in ac_rle:
        if (run, val) == (0, 0): break
        if (run, val) == (15, 0): idx += 16
        else:
            idx += run
            if idx < 64: vec[idx] = val; idx += 1
    return vec

# ══════════════════════════════════════════════════════════════════════
#  BAGIAN 8 — PADDING & ENCODE/DECODE CHANNEL
# ══════════════════════════════════════════════════════════════════════
def pad_to_multiple8(channel: np.ndarray):
    H, W = channel.shape
    pH, pW = (8 - H % 8) % 8, (8 - W % 8) % 8
    return np.pad(channel, ((0, pH), (0, pW)), mode='edge'), H, W

def encode_channel(channel: np.ndarray, qt: np.ndarray, wm_bits=None, start_block=0):
    padded, H, W = pad_to_multiple8(channel)
    pH, pW = padded.shape
    encoded_blocks, dct_before, dct_after = [], [], []
    block_idx = 0
    for row in range(0, pH, 8):
        for col in range(0, pW, 8):
            block = padded[row:row+8, col:col+8].copy()
            dct_block = dct2d(block)
            if block_idx < 16: dct_before.append(dct_block.copy())

            if wm_bits is not None:
                global_idx = start_block + block_idx
                if global_idx < len(wm_bits):
                    zz, bit, coeff = zigzag_scan(dct_block), wm_bits[global_idx], zigzag_scan(dct_block)[EMBED_POSITION]
                    ALPHA = 30.0
                    if bit == 1: coeff = max(coeff, ALPHA)
                    else: coeff = min(coeff, -ALPHA)
                    zz[EMBED_POSITION] = coeff
                    dct_block = inverse_zigzag(zz)
            
            if block_idx < 16: dct_after.append(dct_block.copy())
            q_block = np.round(dct_block / qt)
            dc, ac = rle_encode(zigzag_scan(q_block))
            encoded_blocks.append((dc, ac))
            block_idx += 1
    return encoded_blocks, (pH, pW), (H, W), dct_before, dct_after

def decode_channel(encoded_blocks: list, qt: np.ndarray, padded_shape: tuple, orig_shape: tuple) -> np.ndarray:
    pH, pW, H, W = *padded_shape, *orig_shape
    canvas, idx = np.zeros((pH, pW), dtype=np.float64), 0
    for row in range(0, pH, 8):
        for col in range(0, pW, 8):
            dc, ac = encoded_blocks[idx]; idx += 1
            spatial = idct2d(inverse_zigzag(rle_decode(dc, ac)) * qt)
            canvas[row:row+8, col:col+8] = spatial
    return canvas[:H, :W]

# ══════════════════════════════════════════════════════════════════════
#  BAGIAN 9 — WATERMARK BIT CONVERSION
# ══════════════════════════════════════════════════════════════════════
def text_to_bits(text: str) -> list:
    bits = [(len(text) >> i) & 1 for i in range(15, -1, -1)]
    for ch in text:
        bits.extend([(ord(ch) >> i) & 1 for i in range(7, -1, -1)])
    return bits

def bits_to_text(bits: list) -> str:
    if len(bits) < 16: return ""
    length = sum((b << (15-i)) for i, b in enumerate(bits[:16]))
    chars = []
    for i in range(length):
        start = 16 + i * 8
        if start + 8 > len(bits): break
        chars.append(chr(sum((b << (7-j)) for j, b in enumerate(bits[start:start+8]))))
    return ''.join(chars)

def tile_bits(wm_bits: list, n_blocks: int) -> np.ndarray:
    if not wm_bits: return np.zeros(n_blocks, dtype=int)
    return np.array((wm_bits * ((n_blocks // len(wm_bits)) + 1))[:n_blocks], dtype=int)

# ══════════════════════════════════════════════════════════════════════
#  BAGIAN 10 — METRIK KUALITAS
# ══════════════════════════════════════════════════════════════════════
def compute_psnr(orig: np.ndarray, recon: np.ndarray) -> float:
    mse = np.mean((orig.astype(np.float64) - recon.astype(np.float64)) ** 2)
    return float('inf') if mse == 0 else 10.0 * np.log10(255.0**2 / mse)

def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    C1, C2 = (0.01 * 255)**2, (0.03 * 255)**2
    y1, y2 = rgb_to_ycbcr(img1)[:,:,0].astype(np.float64), rgb_to_ycbcr(img2)[:,:,0].astype(np.float64)
    H, W, ws, vals = *y1.shape, 8, []
    for r in range(0, H - ws + 1, ws):
        for c in range(0, W - ws + 1, ws):
            p1, p2 = y1[r:r+ws, c:c+ws], y2[r:r+ws, c:c+ws]
            mu1, mu2, s1, s2 = np.mean(p1), np.mean(p2), np.var(p1), np.var(p2)
            s12 = np.mean((p1 - mu1)*(p2 - mu2))
            vals.append(((2*mu1*mu2 + C1)*(2*s12 + C2)) / ((mu1**2 + mu2**2 + C1)*(s1 + s2 + C2)))
    return float(np.mean(vals))

def bit_accuracy(b_orig: list, b_extr: list) -> float:
    n = min(len(b_orig), len(b_extr))
    return sum(a == b for a, b in zip(b_orig[:n], b_extr[:n])) / n if n > 0 else 0.0

# ══════════════════════════════════════════════════════════════════════
#  BAGIAN 11 — VISUALISASI
# ══════════════════════════════════════════════════════════════════════
def save_fig(fig, name: str, out_dir: str, dpi: int = 150) -> str:
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  → Gambar disimpan: {name}")
    return path

def vis_step0_original(img_rgb, out_dir):
    fig, ax = plt.subplots(figsize=(6, 6), facecolor='#111')
    ax.imshow(img_rgb); ax.axis('off')
    ax.set_title('STEP 0: Gambar Asli (Host Image x)', color='white', fontsize=13)
    return save_fig(fig, 'step00_original.png', out_dir)

def vis_step1_ycbcr(img_rgb, ycbcr, out_dir):
    fig = plt.figure(figsize=(14, 5), facecolor='#111')
    labels, cmaps = ['R','G','B','Y','Cb','Cr'], ['Reds','Greens','Blues','gray','Blues','Reds']
    data = [img_rgb[:,:,0], img_rgb[:,:,1], img_rgb[:,:,2], ycbcr[:,:,0], ycbcr[:,:,1], ycbcr[:,:,2]]
    for i in range(6):
        ax = fig.add_subplot(1, 6, i+1); ax.imshow(data[i], cmap=cmaps[i]); ax.set_title(labels[i], color='white'); ax.axis('off')
    fig.suptitle('STEP 1: Konversi RGB ke YCbCr', color='white', fontsize=12)
    return save_fig(fig, 'step01_rgb_to_ycbcr.png', out_dir)

def vis_step2_levelshift(Y, Y_shifted, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor='#111')
    axes[0].imshow(Y, cmap='gray'); axes[0].set_title('Y (0–255)', color='white'); axes[0].axis('off')
    axes[1].imshow(Y_shifted, cmap='gray'); axes[1].set_title('Y Shifted (−128–127)', color='white'); axes[1].axis('off')
    fig.suptitle('STEP 2: Level Shift (Persiapan sebelum DCT)', color='white')
    return save_fig(fig, 'step02_level_shift.png', out_dir)

def vis_step3_chroma(Y, Cb_sh, Cb_dn, Cr_sh, Cr_dn, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), facecolor='#111')
    axes[0,0].imshow(Cb_sh, cmap='Blues'); axes[0,0].set_title('Cb Full', color='white'); axes[0,0].axis('off')
    axes[0,1].imshow(Cb_dn, cmap='Blues'); axes[0,1].set_title('Cb Downsampled (4:2:0)', color='white'); axes[0,1].axis('off')
    axes[1,0].imshow(Cr_sh, cmap='Reds'); axes[1,0].set_title('Cr Full', color='white'); axes[1,0].axis('off')
    axes[1,1].imshow(Cr_dn, cmap='Reds'); axes[1,1].set_title('Cr Downsampled (4:2:0)', color='white'); axes[1,1].axis('off')
    fig.suptitle('STEP 3: Chroma Downsampling (Membuang detail warna)', color='white')
    return save_fig(fig, 'step03_chroma_downsample.png', out_dir)

def vis_step4_dct_embed(dct_bef, dct_aft, wm_bits, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), facecolor='#111')
    axes[0,0].imshow(dct_bef[0], cmap=DCT_CMAP); axes[0,0].set_title('DCT Blok 0 (Sebelum)', color='white'); axes[0,0].axis('off')
    axes[0,1].imshow(dct_aft[0], cmap=DCT_CMAP); axes[0,1].set_title(f'DCT Blok 0 (Sesudah Embed Bit={wm_bits[0]})', color='white'); axes[0,1].axis('off')
    axes[1,0].imshow(dct_bef[1], cmap=DCT_CMAP); axes[1,0].set_title('DCT Blok 1 (Sebelum)', color='white'); axes[1,0].axis('off')
    axes[1,1].imshow(dct_aft[1], cmap=DCT_CMAP); axes[1,1].set_title(f'DCT Blok 1 (Sesudah Embed Bit={wm_bits[1]})', color='white'); axes[1,1].axis('off')
    fig.suptitle('STEP 4 - 7: Proses DCT & Watermark Embedding', color='white')
    return save_fig(fig, 'step04_blocks_dct_embed.png', out_dir)

def vis_step8_comparison(original, watermarked, psnr, ssim, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor='#111')
    axes[0].imshow(original); axes[0].set_title('Gambar ASLI', color='white'); axes[0].axis('off')
    axes[1].imshow(watermarked); axes[1].set_title('Gambar HASIL WATERMARK', color='white'); axes[1].axis('off')
    fig.suptitle(f'STEP 8 - 10: Hasil Akhir | PSNR = {psnr:.2f} dB | SSIM = {ssim:.4f}', color='white', fontsize=12)
    return save_fig(fig, 'step08_comparison.png', out_dir)

def vis_step10_pipeline(out_dir):
    fig, ax = plt.subplots(figsize=(12, 4), facecolor='#111')
    ax.text(0.5, 0.5, "PIPELINE SELESAI\n(Lihat diagram lengkap di file sebelumnya)", 
            ha='center', va='center', color='white', fontsize=16)
    ax.axis('off')
    return save_fig(fig, 'step10_pipeline_done.png', out_dir)


# ══════════════════════════════════════════════════════════════════════
#  MAIN PROGRAM
# ══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image',     default='assets/face_256.png',   help='Path ke gambar input')
    parser.add_argument('--watermark', default='BRANDON18224118',       help='Teks watermark')
    parser.add_argument('--quality',   type=int, default=75,            help='Quality Factor')
    parser.add_argument('--alpha',     type=float, default=30.0,        help='Alpha Embedding')
    parser.add_argument('--output',    default='output',                help='Folder Output')
    args = parser.parse_args()

    OUT_DIR = args.output
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"\n{'='*62}")
    print("  MEMULAI PROSES: JPEG WATERMARKING FROM SCRATCH")
    print(f"{'='*62}")

    # [LOAD GAMBAR]
    try:
        img_rgb = np.array(Image.open(args.image).convert('RGB'))
    except Exception as e:
        print(f"\n[ERROR] Gambar tidak ditemukan: {args.image}. Pastikan namanya benar!")
        return

    # ── STEP 0 ────────────────────────────────────────────────────────
    print("\n[PROSES] Memuat Gambar Asli...")
    path0 = vis_step0_original(img_rgb, OUT_DIR)
    interaktif_jeda(path0, "Step 1: Konversi Warna (RGB -> YCbCr)")

    # ── STEP 1 ────────────────────────────────────────────────────────
    print("\n[PROSES] Menghitung rumus konversi RGB ke YCbCr...")
    ycbcr = rgb_to_ycbcr(img_rgb)
    path1 = vis_step1_ycbcr(img_rgb, ycbcr, OUT_DIR)
    interaktif_jeda(path1, "Step 2: Level Shifting (-128)")

    # ── STEP 2 ────────────────────────────────────────────────────────
    print("\n[PROSES] Melakukan Level Shift untuk persiapan nilai DCT...")
    Y, Cb, Cr = ycbcr[:,:,0], ycbcr[:,:,1], ycbcr[:,:,2]
    Y_sh, Cb_sh, Cr_sh = Y - 128.0, Cb - 128.0, Cr - 128.0
    path2 = vis_step2_levelshift(Y, Y_sh, OUT_DIR)
    interaktif_jeda(path2, "Step 3: Chroma Downsampling")

    # ── STEP 3 ────────────────────────────────────────────────────────
    print("\n[PROSES] Membuang detail warna yang tidak sensitif bagi mata...")
    Cb_dn = chroma_downsample(Cb_sh)
    Cr_dn = chroma_downsample(Cr_sh)
    path3 = vis_step3_chroma(Y, Cb_sh, Cb_dn, Cr_sh, Cr_dn, OUT_DIR)
    interaktif_jeda(path3, "Step 4 - 7: Blocking, Proses DCT & Penyisipan Watermark")

    # ── STEP 4-7 (PIPELINE UTAMA) ─────────────────────────────────────
    print("\n[PROSES] Memotong blok 8x8, menghitung matriks DCT, dan Embed Watermark...")
    qt_Y = make_quant_table(LUMA_QT_BASE, args.quality)
    qt_C = make_quant_table(CHROMA_QT_BASE, args.quality)
    wm_bits_list = text_to_bits(args.watermark)
    
    padded_Y, _, _ = pad_to_multiple8(Y_sh)
    n_blocks = (padded_Y.shape[0] // 8) * (padded_Y.shape[1] // 8)
    wm_bits = tile_bits(wm_bits_list, n_blocks)

    enc_Y, ps_Y, os_Y, dct_bef, dct_aft = encode_channel(Y_sh, qt_Y, wm_bits=wm_bits)
    enc_Cb, ps_Cb, os_Cb, _, _ = encode_channel(Cb_dn, qt_C)
    enc_Cr, ps_Cr, os_Cr, _, _ = encode_channel(Cr_dn, qt_C)

    path4 = vis_step4_dct_embed(dct_bef, dct_aft, wm_bits, OUT_DIR)
    interaktif_jeda(path4, "Step 8 - 10: Rekonstruksi & Analisis Perbandingan Akhir")

    # ── STEP 8-last ────────────────────────────────────────────────────────
    print("\n[PROSES] Mengembalikan matriks angka menjadi gambar visual (IDCT)...")
    Y_rec = decode_channel(enc_Y, qt_Y, ps_Y, os_Y) + 128.0
    Cb_rec = decode_channel(enc_Cb, qt_C, ps_Cb, os_Cb) + 128.0
    Cr_rec = decode_channel(enc_Cr, qt_C, ps_Cr, os_Cr) + 128.0

    Cb_up = chroma_upsample(Cb_rec, Y.shape[0], Y.shape[1])
    Cr_up = chroma_upsample(Cr_rec, Y.shape[0], Y.shape[1])
    wm_rgb = ycbcr_to_rgb(np.stack([Y_rec, Cb_up, Cr_up], axis=2))

    psnr_val = compute_psnr(img_rgb, wm_rgb)
    ssim_val = compute_ssim(img_rgb, wm_rgb)

    path8 = vis_step8_comparison(img_rgb, wm_rgb, psnr_val, ssim_val, OUT_DIR)
    
    print("\n[HASIL AKHIR]")
    print(f"  PSNR: {psnr_val:.2f} dB")
    print(f"  SSIM: {ssim_val:.4f}")
    
    interaktif_jeda(path8, None) # None berarti ini langkah terakhir

    print(f"\n{'='*62}")
    print("  PROSES SELESAI DENGAN SUKSES!")
    print("  Semua gambar bukti telah diamankan di folder 'output/'.")
    print(f"{'='*62}\n")

if __name__ == '__main__':
    main()