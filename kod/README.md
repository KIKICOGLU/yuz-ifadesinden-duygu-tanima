# kod/ – Ödev kodları

Projenin deneylerini ve ana demosunu içeren klasör. Kurulum, kullanım kılavuzu ve teknik açıklamalar için ana [README](../README.md) dosyasına bakın.

Bütün scriptler `env_fer` ortamında çalışır. Hangi klasörden çalıştırılırlarsa çalıştırılsınlar modeli `modeller/` klasöründen okur, çıktıları `ciktilar/` klasörüne yazarlar.

## Dosyalar

| Dosya | Ne yapar | Çıktı (`ciktilar/` içinde) |
|---|---|---|
| `duygu_demo.py` | **Ana demo:** canlı duygu tanıma, 8 duygu çubuğu, valence–arousal dairesi (`q` çıkış, `s` ekran görüntüsü) | `oturum_kaydi.csv`, `ekran_N.png` |
| `oturum_ozeti.py` | Uygulama senaryosu: görüşmenin duygu özeti (raporda 7.6) | `oturum_ozeti.png`, `oturum_ozeti.txt` |
| `yuz_tespit.py` | Haar Cascade ile canlı yüz tespiti | ekran |
| `yuz_tespit_yunet.py` | YuNet ile canlı yüz tespiti, 5 yüz noktası, güven skoru | ekran |
| `foto_indir.py` | 46 test fotoğrafını Pexels'ten `veri/fotograflar/` klasörüne indirir | `veri/kaynaklar.csv` |
| `foto_analiz.py` | Fotoğraflarda Haar–YuNet karşılaştırması ve FER–HSEmotion tahminleri (raporda 7.2, 7.3) | `foto_tespit.csv`, `foto_sonuclari.csv`, `foto_ornek_izgara.png` |
| `degerlendirme.py` | Doğruluk, makro F1 ve karışıklık matrisi (`python degerlendirme.py foto_sonuclari.csv`) | `*_confusion_matrix.png`, `*_degerlendirme.txt` |
| `au_analiz_foto.py` | OpenFace çıktısından kas hareketi (AU) ısı haritası (raporda 7.4) | `au_isi_haritasi_foto.png`, `au_ozet_foto.txt` |
| `foto_deneyleri.py` | Yukarıdaki fotoğraf deneylerini ve OpenFace adımını sırayla çalıştırır | hepsi |
| `karsilastirma_canli.py` | Kamerada FER ve HSEmotion yan yana; 1–7 tuşlarıyla etiketli kayıt | `sonuclar.csv` |
| `video_kaydet.py`, `au_analiz.py` | Video üzerinden OpenFace analizi (ek deney) | `ifadeler.avi`, `au_*.png` |
| `modeller/face_detection_yunet_2023mar.onnx` | YuNet modeli (OpenCV Zoo) | |

## Raporun sonuçlarını yeniden üretmek

```powershell
python kod\foto_indir.py        # fotoğrafları indir (bir kez)
python kod\foto_deneyleri.py    # yüz bulma, duygu tahmini, değerlendirme ve OpenFace analizi
```

OpenFace bulunamazsa ilk iki adım yine çalışır; kas hareketi adımı atlanır.

## Bilinen sorunlar ve çözümleri

| Sorun | Çözüm |
|---|---|
| OpenCV 5'te `CascadeClassifier` yok | `opencv-python<5` (requirements dosyasında sabit) |
| hsemotion-onnx 0.3.1: `module 'urllib' has no attribute 'request'` | Scriptlerin başında `import urllib.request` var |
| fer 25.x: `cannot import name 'FER' from 'fer'` | `from fer.fer import FER` |
| fer: `No module named 'pkg_resources'` | `setuptools<81` |
| Aynı ortamda iki farklı OpenCV paketi | Kurulumdaki son satır: contrib paketini aynı sürümle en son kur |
