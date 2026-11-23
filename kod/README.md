# Yüz İfadesinden Duygu Tanıma

Bilgisayar Mühendisliği 4. sınıf, Görüntü İşleme dersi projesi.
Hoca önerisi: **Duygu Tanıma** · Yöntemler: **FER, OpenFace, AffectNet** · Alanlar: müşteri hizmetleri, terapi, oyun.

## Sistem

```
Kamera karesi
   │
   ▼
Yüz tespiti ── Haar Cascade (2001, karşılaştırma için)
   │           YuNet (2023, CNN, kutu + 5 yüz noktası)  ← kullanılan
   ▼
Kırpma (%15 pay) + BGR→RGB
   │
   ├──► HSEmotion (AffectNet ile eğitilmiş EfficientNet-B0)
   │       8 duygu olasılığı + valence + arousal         ← ana model
   │
   ├──► fer (FER-2013 ile eğitilmiş CNN, 7 duygu)          ← karşılaştırma
   │
   └──► OpenFace 2.2 (video üzerinden, 17 Action Unit)     ← açıklanabilirlik
```

## Klasör yapısı

```
kod/
├── README.md
├── requirements_fer.txt, requirements_torch.txt
├── yuz_tespit.py, yuz_tespit_yunet.py      ← yüz tespiti (Haar ve YuNet)
├── duygu_demo.py, oturum_ozeti.py           ← ana demo ve uygulama senaryosu
├── karsilastirma_canli.py, degerlendirme.py ← FER–HSEmotion karşılaştırması
├── video_kaydet.py, au_analiz.py            ← OpenFace ile Action Unit analizi
├── modeller/                                ← YuNet modeli (.onnx)
└── ciktilar/                                ← scriptlerin ürettiği CSV, PNG ve video dosyaları
```

Scriptler hangi klasörden çalıştırılırsa çalıştırılsın modeli `modeller/` klasöründen okur, bütün çıktıları `ciktilar/` klasörüne yazar.

## Dosyalar

| Dosya | Ne yapar | Çıktı (`ciktilar/` içinde) |
|---|---|---|
| `yuz_tespit.py` | Haar Cascade ile canlı yüz tespiti | ekran |
| `yuz_tespit_yunet.py` | YuNet ile canlı yüz tespiti, 5 yüz noktası, güven skoru | ekran |
| `duygu_demo.py` | **Ana demo:** canlı duygu tanıma, 8 duygu çubuğu, valence-arousal dairesi | `ekran_N.png`, `oturum_kaydi.csv` |
| `oturum_ozeti.py` | Uygulama senaryosu: görüşmenin duygu özeti | `oturum_ozeti.png/.txt` |
| `karsilastirma_canli.py` | fer ve HSEmotion yan yana; 1–7 tuşlarıyla etiketli veri toplar | `sonuclar.csv`, `karsilastirma_N.png` |
| `degerlendirme.py` | Doğruluk, F1, confusion matrix | `confusion_matrix.png`, `degerlendirme_sonuclari.txt` |
| `video_kaydet.py` | OpenFace için etiketli ifade videosu kaydeder | `ifadeler.avi`, `ifadeler_bolumler.csv` |
| `au_analiz.py` | OpenFace çıktısından Action Unit grafikleri | `au_zaman.png`, `au_isi_haritasi.png`, `au_ozet.txt` |
| `modeller/face_detection_yunet_2023mar.onnx` | YuNet modeli (OpenCV Zoo) | |

## Ortamlar

İki ayrı sanal ortam kullanıldı, çünkü TensorFlow (fer) ve PyTorch'un bağımlılıkları çakışabiliyor.

| Ortam | İçerik | Kullanım |
|---|---|---|
| `env_fer` | TensorFlow, fer, hsemotion-onnx, OpenCV 4.14, scikit-learn, matplotlib | Demo ve tüm analiz scriptleri |
| `env_torch` | PyTorch 2.11 + CUDA 12.8 | GPU doğrulaması, ileride model eğitimi |

Kurulum adımları `requirements_fer.txt` ve `requirements_torch.txt` dosyalarının başında yazıyor.

## Çalıştırma

`env_fer` ortamı etkinken:

```powershell
python duygu_demo.py          # canlı demo (q: çıkış, s: ekran görüntüsü)
python oturum_ozeti.py        # demo kapandıktan sonra görüşme özeti
```

İlk çalıştırmada HSEmotion modeli (`~/.hsemotion/` klasörüne, ~16 MB) internetten otomatik iner.

OpenFace analizi için video `video_kaydet.py` ile kaydedildikten sonra, `ciktilar/` klasöründe:

```powershell
FeatureExtraction.exe -f ifadeler.avi -out_dir openface_cikti -aus
python ..\au_analiz.py
```

## Bilinen sorunlar ve çözümleri

| Sorun | Çözüm |
|---|---|
| OpenCV 5'te `CascadeClassifier` yok | `opencv-python<5` |
| hsemotion-onnx 0.3.1: `module 'urllib' has no attribute 'request'` | Scriptin başında `import urllib.request` |
| fer 25.x: `cannot import name 'FER' from 'fer'` | `from fer.fer import FER` |
| fer: `No module named 'pkg_resources'` | `setuptools<81` |
| Aynı ortamda iki farklı sürüm OpenCV paketi | İkisini kaldır, aynı 4.x sürümünü kur, contrib'i en son kur |
