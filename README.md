# Örgü ve Tekstil Kumaşlarında İlmek ve Doku Kusurlarının CNN ile Tespiti

Bu depo, tekstil ve triko üretiminde örgü kumaş yüzeyindeki ilmek kaçıkları, delikler, iplik yığılmaları ve yağ lekelerini tespit etmek üzere geliştirilmiş derin öğrenme (Evrişimli Sinir Ağları - CNN) projesini içermektedir.

---

## 📌 Proje Özeti
* **Problem Türü:** İkili Sınıflandırma (Hatasız Örgü vs. Kusurlu Örgü)
* **Model Mimarisi:** 4 Bloklu Özel CNN (Conv2D + MaxPooling + Flatten + Dense + Dropout)
* **Veri Seti:** 1.000 adet 128x128 gri tonlamalı makro örgü görüntüsü (%80 Hatasız, %20 Kusurlu - Gerçekçi Sınıf Dengesizliği)
* **Temel Metrikler (Test Kümesi - 150 Örnek):**
  * **Doğruluk (Accuracy):** %100.00
  * **Kesinlik (Precision):** %100.00
  * **Duyarlılık (Recall):** %100.00 *(Kritik hata: Sıfır kusur kaçırma)*
  * **F1-Skoru:** 1.0000

---

## 📁 Dizin Yapısı
```text
├── figures/                          # EDA ve Değerlendirme Grafikleri
│   ├── eda_1_sample_images.png       # Örnek kumaş dokuları (Hatasız vs Kusurlu)
│   ├── eda_2_class_distribution.png  # Sınıf dağılımı (Dengesizlik analizi)
│   ├── eda_3_pixel_intensity_hist.png# Piksel yoğunluk histogramı
│   ├── eval_1_learning_curves.png    # Eğitim & Doğrulama Kayıp/Doğruluk eğrileri
│   ├── eval_2_confusion_matrix.png   # Karışıklık Matrisi (Confusion Matrix)
│   └── eval_3_error_analysis.png     # Tahmin ve Hata İnceleme Görseli
├── knitwear_defect_colab_notebook.ipynb # Google Colab / Jupyter Notebook
├── train_and_evaluate.py             # Uçtan uca veri üretimi, model eğitimi ve test kodu
├── evaluation_metrics.txt            # Test kümesi sayısal metrik özetleri
├── PROJE_RAPORU.md                   # 3-5 Sayfalık Resmi Proje Teslim Raporu
└── README.md                         # Proje tanıtım ve dokümantasyon dosyası
```

---

## 🚀 Hızlı Başlangıç

### 1. Google Colab Üzerinde Çalıştırma
`knitwear_defect_colab_notebook.ipynb` dosyasını [Google Colab](https://colab.research.google.com)'e yükleyerek tek tıkla (`Runtime -> Run all`) modeli eğitebilir ve grafikleri görüntüleyebilirsiniz.

### 2. Yerel Ortamda Çalıştırma
```bash
# Gerekli kütüphaneler:
pip install tensorflow matplotlib numpy pillow

# Eğitimi başlatma ve grafikleri üretme:
python train_and_evaluate.py
```
