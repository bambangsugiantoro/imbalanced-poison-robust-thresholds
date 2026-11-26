# Sumber, tautan, dan pencatatan eksperimen (untuk naskah dan revisi)

Dibuat 25 Nov 2026. Tujuan: satu tempat untuk mengecek asal data, cara memperolehnya, versi, dan jalur reproduksi tiap eksperimen supaya revisi tidak membingungkan. Tautan yang ditandai [terverifikasi] sudah dibuka pada tanggal ini; yang ditandai [verifikasi dulu] ditulis dari pengetahuan umum dan harus dibuka ulang sebelum dikutip di naskah.

## A. Dataset benchmark 49 (acak, CV 5x2)
- Pemuat: paket Python `imbalanced-databases` (repo https://github.com/gykovacs/imbalanced_databases) [terverifikasi: repo ada, README menyatakan tidak dipelihara lagi dan menunjuk pengganti https://github.com/gykovacs/common_datasets]. Dipasang dengan `pip install --no-deps imbalanced-databases` (lihat `code/hpc_setup.sh`). Daftar nama, ukuran, IR: `docs/DATASET_REGISTRY.csv`. Nama 49 dataset: `hasil/datasets.txt`.
- Koleksi KEEL (42 dataset): repositori KEEL imbalanced, https://sci2s.ugr.es/keel/imbalanced.php [verifikasi dulu]. Sitasi: Alcala-Fdez J. et al. (2011), KEEL data-mining software tool and data set repository, J Mult-Valued Logic Soft Comput 17 [verifikasi detail volume/halaman lewat Crossref sebelum dikutip].
- Koleksi KRNN (7 dataset: spectf, pc1, kc1, hiva, ada, satimage, sylva): sitasi di registri "Zhang, Tari, Cheriet 2017"; judul lengkap dan DOI belum diverifikasi, cari lewat Crossref/Google Scholar.
- Asal asli dataset: UCI Machine Learning Repository, https://archive.ics.uci.edu [verifikasi dulu]; NASA MDP/PROMISE untuk pc1 dan kc1; WCCI 2006 Agnostic Learning vs Prior Knowledge untuk ada dan sylva; NIPS 2003 challenge untuk hiva. Sitasi UCI per dataset diisi dari kolom "Asal data" di tabel berikut.
- Catatan versi: paket tidak dipelihara; sebut versi paket dan tanggal akses di naskah (jalankan `pip show imbalanced-databases` di HPC untuk mencatat versi).

| No | Dataset (nama paket) | n | d | IR | n minor | Asal data | Koleksi |
|---|---|---|---|---|---|---|---|
| 1 | iris0 | 150 | 4 | 2.0 | 50 | UCI Iris | KEEL |
| 2 | glass0 | 214 | 9 | 2.06 | 70 | UCI Glass Identification | KEEL |
| 3 | glass1 | 214 | 9 | 1.82 | 76 | UCI Glass Identification | KEEL |
| 4 | glass_0_1_2_3_vs_4_5_6 | 214 | 9 | 3.2 | 51 | UCI Glass Identification | KEEL |
| 5 | glass6 | 214 | 9 | 6.38 | 29 | UCI Glass Identification | KEEL |
| 6 | new_thyroid1 | 215 | 5 | 5.14 | 35 | UCI Thyroid Disease (new-thyroid) | KEEL |
| 7 | ecoli_0_vs_1 | 220 | 7 | 1.86 | 77 | UCI Ecoli (Horton & Nakai) | KEEL |
| 8 | ecoli_0_3_4_7_vs_5_6 | 257 | 7 | 9.28 | 25 | UCI Ecoli (Horton & Nakai) | KEEL |
| 9 | spectf | 267 | 44 | 3.85 | 55 | UCI SPECTF Heart | KRNN |
| 10 | habarman | 306 | 3 | 2.78 | 81 | UCI Haberman Survival | KEEL |
| 11 | ecoli_0_1_4_7_vs_5_6 | 332 | 6 | 12.28 | 25 | UCI Ecoli (Horton & Nakai) | KEEL |
| 12 | ecoli1 | 336 | 7 | 3.36 | 77 | UCI Ecoli (Horton & Nakai) | KEEL |
| 13 | ecoli3 | 336 | 7 | 8.6 | 35 | UCI Ecoli (Horton & Nakai) | KEEL |
| 14 | ecoli_0_1_4_7_vs_2_3_5_6 | 336 | 7 | 10.59 | 29 | UCI Ecoli (Horton & Nakai) | KEEL |
| 15 | ecoli2 | 336 | 7 | 5.46 | 52 | UCI Ecoli (Horton & Nakai) | KEEL |
| 16 | led7digit_0_2_4_5_6_7_8_9_vs_1 | 443 | 7 | 10.97 | 37 | UCI LED Display Domain | KEEL |
| 17 | yeast_1_vs_7 | 459 | 7 | 14.3 | 30 | UCI Yeast | KEEL |
| 18 | page_blocks_1_3_vs_4 | 472 | 10 | 15.86 | 28 | UCI Page Blocks Classification | KEEL |
| 19 | yeast_2_vs_4 | 514 | 8 | 9.08 | 51 | UCI Yeast | KEEL |
| 20 | wisconsin | 683 | 9 | 1.86 | 239 | UCI Breast Cancer Wisconsin (Original) | KEEL |
| 21 | abalone9_18 | 731 | 8 | 16.4 | 42 | UCI Abalone | KEEL |
| 22 | pima | 768 | 8 | 1.87 | 268 | UCI/NIDDK Pima Indians Diabetes | KEEL |
| 23 | vehicle2 | 846 | 18 | 2.88 | 218 | UCI Statlog Vehicle Silhouettes | KEEL |
| 24 | vehicle0 | 846 | 18 | 3.25 | 199 | UCI Statlog Vehicle Silhouettes | KEEL |
| 25 | vehicle1 | 846 | 18 | 2.9 | 217 | UCI Statlog Vehicle Silhouettes | KEEL |
| 26 | vehicle3 | 846 | 18 | 2.99 | 212 | UCI Statlog Vehicle Silhouettes | KEEL |
| 27 | vowel0 | 988 | 13 | 9.98 | 90 | UCI Vowel Recognition | KEEL |
| 28 | flaref | 1066 | 11 | 23.79 | 43 | UCI Solar Flare | KEEL |
| 29 | pc1 | 1109 | 21 | 13.4 | 77 | NASA MDP / PROMISE software defect (PC1) | KRNN |
| 30 | kr_vs_k_zero_vs_eight | 1460 | 6 | 53.07 | 27 | UCI Chess King-Rook vs King | KEEL |
| 31 | winequality_white_3_9_vs_5 | 1482 | 11 | 58.28 | 25 | UCI Wine Quality (Cortez et al. 2009) | KEEL |
| 32 | winequality_red_4 | 1599 | 11 | 29.17 | 53 | UCI Wine Quality (Cortez et al. 2009) | KEEL |
| 33 | abalone_19_vs_10_11_12_13 | 1622 | 8 | 49.69 | 32 | UCI Abalone | KEEL |
| 34 | car_good | 1728 | 6 | 24.04 | 69 | UCI Car Evaluation | KEEL |
| 35 | car_vgood | 1728 | 6 | 25.58 | 65 | UCI Car Evaluation | KEEL |
| 36 | shuttle_c0_vs_c4 | 1829 | 9 | 13.87 | 123 | UCI Statlog Shuttle | KEEL |
| 37 | abalone_20_vs_8_9_10 | 1916 | 8 | 72.69 | 26 | UCI Abalone | KEEL |
| 38 | kc1 | 2109 | 21 | 5.47 | 326 | NASA MDP / PROMISE software defect (KC1) | KRNN |
| 39 | kr_vs_k_zero_vs_fifteen | 2193 | 6 | 80.22 | 27 | UCI Chess King-Rook vs King | KEEL |
| 40 | kr_vs_k_one_vs_fifteen | 2244 | 6 | 27.77 | 78 | UCI Chess King-Rook vs King | KEEL |
| 41 | abalone_17_vs_7_8_9_10 | 2338 | 8 | 39.31 | 58 | UCI Abalone | KEEL |
| 42 | kr_vs_k_zero_one_vs_draw | 2901 | 6 | 26.63 | 105 | UCI Chess King-Rook vs King | KEEL |
| 43 | kr_vs_k_three_vs_eleven | 2935 | 6 | 35.23 | 81 | UCI Chess King-Rook vs King | KEEL |
| 44 | shuttle_2_vs_5 | 3316 | 9 | 66.67 | 49 | UCI Statlog Shuttle | KEEL |
| 45 | hiva | 3845 | 1617 | 27.48 | 135 | NIPS 2003/WCCI 2006 ADA-HIVA challenge (IBM drug discovery) | KRNN |
| 46 | ada | 4147 | 47 | 3.03 | 1029 | WCCI 2006 Agnostic Learning vs Prior Knowledge (ADA, census) | KRNN |
| 47 | page_blocks0 | 5472 | 10 | 8.79 | 559 | UCI Page Blocks Classification | KEEL |
| 48 | satimage | 6435 | 36 | 9.28 | 626 | UCI Statlog Landsat Satellite | KRNN |
| 49 | sylva | 13086 | 212 | 15.26 | 805 | WCCI 2006 ALvsPK (SYLVA, forest cover) | KRNN |

## B. Dataset keamanan temporal (creditcard dan IEEE-CIS)
- Creditcard: OpenML id 42397, nama CreditCardFraudDetection v2, 284.807 instans, target `Class`, kolom waktu `Time`. Halaman: https://www.openml.org/d/42397 . Arff: https://openml.org/data/v1/download/21829702/CreditCardFraudDetection.arff . Asal: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud [terverifikasi lewat API OpenML]. Sitasi yang diminta deskripsinya: Dal Pozzolo A., Caelen O., Johnson R.A., Bontempi G. (2015), Calibrating Probability with Undersampling for Unbalanced Classification, IEEE Symposium on Computational Intelligence and Data Mining (CIDM). Metadata lisensi OpenML tidak jelas ("English"); periksa syarat penggunaan di Kaggle sebelum redistribusi (kita tidak redistribusi).
- IEEE-CIS: OpenML id 46858, nama IEEE-CIS_Fraud_Detection v2, target `isFraud`, kolom waktu `TransactionDT`. Halaman: https://www.openml.org/d/46858 . Arff: https://openml.org/data/v1/download/22124421/IEEE-CIS_Fraud_Detection.arff . Asal: https://www.kaggle.com/competitions/ieee-fraud-detection/data (kompetisi IEEE CIS Fraud Detection 2019, data Vesta Corporation) [terverifikasi lewat API OpenML]. Lisensi: "Kaggle and IEEE CIS"; baca aturan kompetisi sebelum klaim penggunaan; kita tidak mendistribusikan ulang.
- Pemuatan: `code_sec/sec_bench.py` (`load_openml`), pembatas mayoritas CAP=60000 (creditcard) dan 30000 (IEEE-CIS).

## C. UNSW-NB15 (dataset ketiga, temporal)
- Halaman resmi: https://research.unsw.edu.au/projects/unsw-nb15-dataset [terverifikasi]. Folder berkas sumber (SharePoint): https://unsw-my.sharepoint.com/:f:/g/personal/z5025758_ad_unsw_edu_au/EnuQZZn3XuNBjgfcUu4DIVMBLCHyoLHqOswirpOQifr1ag?e=gKWkLS [terverifikasi 25 Nov 2026]. Jalur di folder: `UNSW-NB15 dataset / CSV Files /`.
- Berkas yang dipakai (diunduh 25 Nov 2026 ke HPC `~/imb_bench/data_unsw/`): UNSW-NB15_1.csv (168.979.718 B), _2.csv (165.221.021 B), _3.csv (154.588.103 B), _4.csv (97.588.754 B), NUSW-NB15_features.csv (4.044 B). 49 kolom, tanpa header; nama kolom dari berkas fitur. Waktu: `Stime`; label: `Label` (1 = serangan).
- Sitasi (verifikasi dulu lewat halaman resmi dan Crossref): Moustafa N., Slay J. (2015), UNSW-NB15: a comprehensive data set for network intrusion detection systems, MilCIS 2015; Moustafa N., Slay J. (2016), The evaluation of Network Anomaly Detection Systems, Information Security Journal 25(1-3).
- Versi lain (Zenodo, HuggingFace, CIC UNSW-NB15 Augmented) tidak dipakai karena tidak berurutan waktu atau berbeda proses; jangan dicampur.
- Pemuat: `code_sec/sec_unsw_cap.py` (fungsi `load_unsw`).

## D. Peta eksperimen (job HPC -> skrip -> keluaran -> bagian PANDUAN)
| Job | Skrip | Keluaran (di ~/imb_bench) | Isi | Bagian |
|---|---|---|---|---|
| 1977 | code/run_bench.py (array) | results/ | benchmark 49 dataset, 9 strategi x 4 learner | 1-10 |
| (lain) | code/analyze.py | results_paired_vs_tuned_baseline.csv | uji Wilcoxon+Holm | 1-10 |
| (lain) | HGB tanpa early stopping | lihat PANDUAN | hipotesis HGB ditolak | 11 |
| 2034 | code_sec/sec_poison.py | results_poison/ ; sec_poison_out.txt | poisoning temporal creditcard, IEEE-CIS | 12 |
| (lain) | code/meta_selector.py | meta_f1_out.txt, meta_mcc_out.txt | pemilih strategi (hasil negatif) | 13 |
| 2037/2038 | code_sec/sec_poison_thr.py | results_poison_thr/ ; thr_out.txt | dekomposisi ambang, trim25/50 | 14 |
| 2039/2040 | code_sec/sec_poison_thr2.py | results_poison_thr2/ ; cap_out.txt | cap/gated (AUC>=0.95) | 15 |
| 2041 | code_sec/bench49_poison.py, analyze_b49.py | results_b49/ ; b49_out.txt, b49_perdataset.csv, b49_tests.csv | 49 dataset, FPR 5% dan 1% | 16 |
| 2042 | code_sec/sec_unsw_cap.py | results_unsw/unsw.csv ; logs/unsw_cap.out | konfirmasi UNSW-NB15 | 17 (menyusul) |

Seed dan parameter tetap: CV acak random_state=20261124 (benchmark) dan 20261125 (b49); permutasi flip 7000+jendela (temporal) dan 9000+fold (b49); seed model 1000+indeks. Jendela temporal EDGES 0.5, 0.625, 0.75, 0.875, 1.0. Validasi = 25% terakhir (temporal) atau pemisahan stratifikasi 75/25 (acak).

## E. Panduan bila ada revisi eksperimen
1. Jangan menimpa folder hasil lama. Jalankan versi baru ke folder baru (misal results_b49_v2) dan catat di tabel D.
2. Aturan (cap median/q25, gerbang AUC 0,95) dikembangkan di creditcard dan IEEE-CIS; UNSW-NB15 dan 49 dataset adalah konfirmasi. Jangan mengubah aturan setelah melihat UNSW tanpa mencatatnya sebagai pengembangan baru.
3. Analisis yang masih terbuka: recall pada FPR aktual yang disetarakan (bagian 16 butir 2), dan ambang FPR dari validasi kecil yang meleset pada data bersih.
4. Sebelum submit: verifikasi semua sitasi lewat Crossref/DOI, periksa lisensi tiap dataset, perbarui Data Availability dengan tautan repositori GitHub dan DOI Zenodo (setelah rilis final).
5. Repositori publik: https://github.com/bambangsugiantoro/imbalanced-poison-robust-thresholds (belum memuat sec_unsw_cap.py, LICENSE, dan hasil lengkap).

## F. Dataset temporal held-out (Elliptic, CIC-IDS2017, BAF)
- CIC-IDS2017: OpenML id 45277 (versi yang dipakai). Catatan kualitas label: Lanvin et al. (2023) dan Liu et al. (2022) mendokumentasikan kesalahan label pada korpus ini (lihat daftar pustaka naskah).
- Elliptic Bitcoin dataset: Weber et al. (2019), arXiv:1908.02591. Sumber asli: Kaggle (Elliptic). Tidak didistribusikan ulang.
- Bank Account Fraud (BAF): Jesus et al. (2022), arXiv:2211.13358. Sub-sampel 1-dari-5 baris setelah diurutkan menurut bulan. Lisensi distribusi belum diperiksa; periksa halaman distribusi sebelum digunakan ulang.
- Pemuat dan skrip: lihat `code_sec/`. Spesifikasi yang dibekukan sebelum run: `PRESPEC_*.md` (TPC, AUDIT, TILT, AST, AST2, KONFIRMASI). Tabel akhir: `results/hasil_final/`.
- Tidak ada data mentah yang disimpan di repositori ini.
