# JPEG Watermarking From Scratch — Sistem Multimedia

**Nama:** Brandon Zeko Alexander  
**NIM:** 18224118

---

## Deskripsi

Implementasi **watermarking citra** menggunakan pipeline JPEG yang dibangun dari nol (*from scratch*), tanpa menggunakan library kompresi JPEG yang sudah jadi. Watermark disisipkan di domain DCT (koefisien mid-frequency) sehingga gambar sebelum (`x`) dan sesudah watermark (`y`) **tidak mudah dibedakan** secara visual.

---

## Arsitektur Pipeline

```
INPUT (x) → RGB→YCbCr → Level Shift −128 → Chroma Downsample 4:2:0
           → Padding ke kelipatan 8 → Blok 8×8 → 2D DCT
           → [EMBED WATERMARK di koefisien mid-frequency]
           → Quantization → Zigzag Scan → RLE Encoding
           → RLE Decode → Dequantize → IDCT
           → Chroma Upsample → YCbCr→RGB → OUTPUT watermarked (y)
```

---

## Tahapan JPEG From Scratch

### Step 1 — Konversi RGB → YCbCr (BT.601)
```
Y  =  0.299·R + 0.587·G + 0.114·B
Cb = −0.169·R − 0.331·G + 0.5·B   + 128
Cr =  0.5·R   − 0.419·G − 0.081·B + 128
```
Mata manusia jauh lebih sensitif terhadap luminance (Y) daripada chrominance (Cb/Cr).

### Step 2 — Level Shift (−128)
Setiap nilai pixel dikurangi 128 agar rentang berubah dari [0, 255] ke [−128, 127]. DCT bekerja lebih efisien pada data yang berpusat di nol.

### Step 3 — Chroma Downsampling 4:2:0
Cb dan Cr di-downsample 2× di kedua dimensi (rata-rata 2×2 block). Karena mata kurang sensitif terhadap warna, ini menghemat ~50% data chrominance tanpa penurunan kualitas yang signifikan.

### Step 4 — Padding ke Kelipatan 8
Gambar di-pad ke ukuran yang merupakan kelipatan 8 (edge-padding), karena JPEG memproses data dalam blok 8×8.

### Step 5 — 2D DCT per Blok 8×8
DCT (Discrete Cosine Transform) mengubah domain spasial (pixel) ke domain frekuensi. Menggunakan `scipy.fft.dct` dengan pendekatan separable (baris lalu kolom). Koefisien kiri-atas = frekuensi rendah (penting), kanan-bawah = frekuensi tinggi (kurang penting).

### Step 6 — Watermark Embedding (DCT Domain)
Satu bit watermark di-embed per blok 8×8 pada **koefisien mid-frequency** (zigzag index ke-14):
- `bit = 1` → paksa koefisien ≥ +α  
- `bit = 0` → paksa koefisien ≤ −α

Alpha (α = 30) adalah kekuatan embedding. Setelah quantize-dequantize, *sign* koefisien tetap terjaga sehingga watermark bisa diekstrak kembali.

### Step 7 — Quantization
Setiap koefisien DCT dibagi dengan nilai dari tabel kuantisasi, lalu dibulatkan ke integer. Tabel kuantisasi diskalakan berdasarkan **Quality Factor (QF)**:
```
Jika QF < 50 : scale = 5000 / QF
Jika QF ≥ 50 : scale = 200 − 2·QF
Qt[i,j] = clip(floor(base[i,j] × scale / 100 + 0.5), 1, 255)
```

### Step 8 — Zigzag Scan
Blok 8×8 disusun ulang menjadi array 1D mengikuti pola zigzag. Hasilnya: koefisien DC di depan, frekuensi rendah di awal, frekuensi tinggi di akhir — memudahkan RLE karena nol-nol berkumpul di akhir.

### Step 9 — RLE Encoding
AC koefisien di-encode dengan Run-Length Encoding:
- `(run, value)` = berapa nol sebelumnya, lalu nilai non-nol
- `(0, 0)` = EOB (End Of Block)
- `(15, 0)` = ZRL (16 nol berturut-turut)

### Step 10 — Rekonstruksi (Decode)
RLE decode → inverse zigzag → dequantize (×Qt) → IDCT → +128 → chroma upsample → YCbCr→RGB.

---

## Cara Pakai

### Install Dependensi
```bash
pip install numpy scipy matplotlib pillow
```

### Jalankan Pipeline Lengkap
```bash
python main.py
```

### Dengan Argumen Custom
```bash
python main.py --image assets/face_256.png \
               --watermark "BRANDON18224118" \
               --quality 75 \
               --alpha 30.0 \
               --output output
```

---

## Output

| File | Deskripsi |
|------|-----------|
| `step00_original.png` | Gambar asli (host image x) |
| `step01_rgb_to_ycbcr.png` | Konversi RGB → YCbCr channel |
| `step02_level_shift.png` | Level shift −128 & histogram |
| `step03_chroma_downsample.png` | Chroma downsampling 4:2:0 |
| `step04_blocks_dct_embed.png` | Blok 8×8, DCT, embed watermark |
| `step05_quant_tables.png` | Tabel kuantisasi (heatmap) |
| `step06_zigzag.png` | Ilustrasi zigzag scan |
| `step07_rle.png` | RLE encoding koefisien |
| `step08_comparison_x_y.png` | Gambar x vs y (original vs watermarked) |
| `step09_qf_analysis.png` | Analisis ekstraksi pada QF 1–100 |
| `step10_pipeline_diagram.png` | Diagram pipeline lengkap |
| `watermarked_face.png` | Gambar hasil watermarking (y) |

---

## Metrik Kualitas

| Metrik | Nilai | Keterangan |
|--------|-------|------------|
| **PSNR** | ~31.7 dB | > 30 dB = perbedaan tidak terlihat |
| **SSIM** | ~0.926 | > 0.9 = hampir identik |
| **Bit Accuracy** | 100% | pada semua QF 1–100 |

---

## Struktur File

```
watermark_project/
├── main.py              ← Script utama (self-contained, semua step di sini)
├── assets/
│   ├── face.jpeg        ← Foto wajah asli
│   └── face_256.png     ← Foto yang dipakai (256×256)
├── output/              ← Semua visualisasi dihasilkan di sini
│   ├── step00_original.png
│   ├── ...
│   └── watermarked_face.png
└── README.md
```

---

## Library yang Digunakan

| Library | Fungsi | Keterangan |
|---------|--------|------------|
| `numpy` | Operasi array & numerik | — |
| `scipy.fft.dct` | DCT/IDCT per blok | **Bukan library JPEG!** Hanya fungsi matematika DCT |
| `matplotlib` | Visualisasi step-by-step | — |
| `PIL (Pillow)` | Load/save gambar | Hanya untuk I/O, bukan kompresi JPEG |

> **Catatan:** Seluruh logika JPEG (quantization, zigzag, RLE, chroma sampling, rekonstruksi) ditulis manual. `scipy.fft.dct` hanya digunakan sebagai fungsi matematika DCT, bukan sebagai library kompresi JPEG.
