# Spesifikasi pra-uji metode TPC (dibekukan sebelum data konfirmasi untuk TPC dijalankan) — 26 Nov 2026
- Kode: cq3_run.py (md5 27e47ea3c9862138d79bf3085f4016da), dibuat dari cq2_run.py (md5 8d322581...) oleh mk3.py; aturan lama tidak berubah.
- Noise baru `delay2`: positif berumur a∈[0,1] (0=terbaru) belum berlabel dengan peluang exp(−a/λ); λ dicari sehingga rata-rata porsi belum berlabel = tingkat poison (0,2; 0,4).
- Aturan baru: `tpc` dan `tpcf` = kuantil terkoreksi (cq_lib.cq_threshold) dengan π dari estimasi temporal; π = (p − r_val)/(1 − r_val) terpotong [0; 0,9]; K=10 potongan; penjaga 2×; fallback ke π CQ.
- Pengembangan: creditcard, IEEE-CIS, UNSW (LR, RF, XGB; BASE, CW; poison 0, 0,2, 0,4; anggaran 1% dan 0,1%). Tidak ada penyetelan parameter setelah melihat hasil konfirmasi.
- Konfirmasi: Elliptic, CIC-IDS2017 dengan konfigurasi sama (parameter beku: K=10, penjaga 2×).
- Hipotesis: HT1 tpcf/tpc mengurangi galat kalibrasi mutlak dibanding naive pada delay2 dengan Holm p<0,05 pada dataset pengembangan; HT2 sama pada ≥1 dataset konfirmasi; HT3 akurasi π: |π_tpcf − π_true| < |π_cq − π_true|. Kegagalan dilaporkan sebagai kegagalan.
- Metrik dan uji sama dengan PRESPEC_KONFIRMASI.md (analyze_conf.py).
