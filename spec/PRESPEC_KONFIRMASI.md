# Spesifikasi pra-uji (dibekukan SEBELUM data konfirmasi dibuka) — dibuat 25 Nov 2026

Data konfirmasi: CIC-IDS-2017 (OpenML 45277, kolom Timestamp) dan Elliptic (Kaggle, 46.564 node berlabel, 49 time step).
Tidak ada data dari dua set ini yang pernah dipakai untuk memilih aturan, parameter, atau metrik.

## Aturan yang dibekukan (tanpa perubahan setelah melihat data)
- Aturan ambang: naive, oracle, cq (cq_lib.py md5 9d2447967285034590407d556bbecb4e), cap25, gcap25, conf, npu (delta=0,05), cl, trust (5% validasi berlabel bersih), ideal (rujukan tak terjangkau).
- Pembelajar: LR, RF, XGB. Strategi: BASE, CW, RUS, SMOTE. Tingkat poison 0, 0,2, 0,4. Noise: acak dan terlambat (delay).
- Anggaran FPR: 1% dan 0,1%. Pembagian waktu: rolling-origin dengan tepi 0,5, 0,625, 0,75, 0,875, 1,0 (Elliptic: urut time step).
- Kode: cq2_run.py (md5 7a1d43d68c832c4647de1ff6594fef02), pemuat data baru ditulis tanpa mengubah aturan.

## Metrik utama (ditetapkan di muka)
1. Galat kalibrasi mutlak |log2(FPR/target)| (lantai FPR 1e-4), rata-rata per dataset.
2. Pangsa dalam 1,5x anggaran (ok15). 3. Recall berpenalti anggaran (penR).
4. Akurasi pi_hat vs pi sebenarnya.
## Uji
- Wilcoxon berpasangan per dataset/split, koreksi Holm, alpha 0,05; laporkan ukuran efek dan CI bootstrap 95%.
- Hipotesis konfirmasi: H1 (naive runtuh), H2 (recall dan FPR monoton antar aturan), H3 (CQ lebih baik dari naive), H4 (bias pi_hat bergantung AUC), H5 (resampling berinteraksi dengan kalibrasi).
## Aturan pelaporan
- Semua hasil dilaporkan, termasuk negatif. Tidak ada dataset, strategi, atau metrik yang dibuang setelah melihat hasil.
- Penyimpangan dari dokumen ini dilaporkan sebagai penyimpangan di naskah.

## Catatan pelaksanaan (26 Nov 2026, sebelum hasil konfirmasi dilihat)
- cq2_run.py md5 7a1d43d6... diperluas hanya dengan dua pemuat data (load_cic, load_elliptic); aturan ambang tidak diubah. md5 baru: e96332bbe36729cb4a0d4c2d622ffb92.
- CIC-IDS-2017 (OpenML 45277, salinan parquet 389.007.233 byte): urut Timestamp (format d/m/Y campuran, dayfirst); 288.602 baris tanpa Timestamp dibuang; 2.830.743 baris, 19,7% positif (bukan Benign). Kolom Flow_ID, Source_IP, Destination_IP, Timestamp, Benign, Sub_Cat, Cat tidak dipakai sebagai fitur. Kolom Device dikodekan sebagai kategori.
- Elliptic: 46.564 node berlabel, 165 fitur, 4.545 ilisit (9,8%), diurut menurut time step; empat jendela rolling-origin.
- Dijalankan sebagai job Slurm 2062-2065 (Elliptic dan CIC, noise acak dan delay), STRATS=BASE,CW,RUS,SMOTE, level 0/0,2/0,4, anggaran 1% dan 0,1%.
