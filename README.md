# Leaf-Classifier

## Alur:
#### Training
cek model, ada atau gak: 
-Kalau ada load model dan selesai
-Kalau ga ada, training masing-masing daun sesuai direktorinya. Kalau udah ditraining, model disimpan

nb: models isinya 5 model (sesuai dengan jumlah daun). parameter = (c=0.01, gamma=.00000001, tipe kernel="RBF"). Cara training bisa dilihat di prediksi

#### Prediksi
-input gambar (dari web)
-simpan gambar dulu
-crop gambar
-buang background putih (gak halus buangnya, tapi aman)
-prediksi dengan svm. Dari hasil itu daun ada probabilitynya masuk mana, misal excocaria=0.6, havea=0.3, chinese_tallow=0.9, yang dipilih yang paling tinggi, yaitu chinese_tallow

