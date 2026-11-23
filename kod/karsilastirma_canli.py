"""
FER (FER-2013) ve HSEmotion (AffectNet) canlı karşılaştırması + veri toplama
---------------------------------------------------------------------------
- Yüz tespiti: iki model için de AYNI dedektör (YuNet) kullanılır, böylece
  karşılaştırma adil olur; fark sadece duygu sınıflandırıcısından gelir.
- Sol panel: fer kütüphanesi (FER-2013 ile eğitilmiş CNN, 7 duygu)
- Sağ panel: HSEmotion (AffectNet ile eğitilmiş EfficientNet-B0, 8 duygu)
  Karşılaştırma için HSEmotion'daki "contempt" (küçümseme) sınıfı çıkarılıp
  kalan 7 olasılık yeniden normalize edilir; böylece iki model aynı 7 sınıfta
  yarışır.

Veri toplama: Bir ifade yaparken o ifadenin tuşuna bas, iki modelin tahmini
"sonuclar.csv" dosyasına eklenir. Sonra degerlendirme.py ile doğruluk ve
confusion matrix hesaplanır.

  1 = Mutluluk   2 = Uzuntu   3 = Ofke   4 = Saskinlik
  5 = Korku      6 = Igrenme  7 = Notr
  s = ekran görüntüsü   q = çıkış

Ortam: env_fer  (tensorflow + fer + hsemotion-onnx + opencv<5)
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # TensorFlow'un uzun log mesajlarını gizle

import urllib.request  # hsemotion-onnx 0.3.1 hatası: urllib.request'i import etmeyi unutuyor
import csv
from datetime import datetime

import cv2
import numpy as np
from fer.fer import FER   # fer 25.x: FER sınıfı fer/__init__.py yerine fer/fer.py içinde
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
from pathlib import Path

# Script hangi klasörden çalıştırılırsa çalıştırılsın:
# modeller kod/modeller'den okunur, çıktılar (CSV, PNG, video) kod/ciktilar'a yazılır.
KLASOR = Path(__file__).resolve().parent
CIKTI = KLASOR / "ciktilar"
CIKTI.mkdir(exist_ok=True)
os.chdir(CIKTI)

YUZ_MODELI = str(KLASOR / "modeller" / "face_detection_yunet_2023mar.onnx")
CSV_DOSYA = "sonuclar.csv"
FONT = cv2.FONT_HERSHEY_SIMPLEX

# İki modelde ortak olan 7 duygu (karşılaştırma bunlar üzerinden yapılır)
ORTAK = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
TR = {"angry": "Ofke", "disgust": "Igrenme", "fear": "Korku", "happy": "Mutluluk",
      "sad": "Uzuntu", "surprise": "Saskinlik", "neutral": "Notr"}
# HSEmotion'un çıktı sırası (8 sınıf)
HSE_SIRA = ["angry", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

TUSLAR = {ord("1"): "happy", ord("2"): "sad", ord("3"): "angry", ord("4"): "surprise",
          ord("5"): "fear", ord("6"): "disgust", ord("7"): "neutral"}


def hse_ortak(olasiliklar):
    """HSEmotion'ın 8 olasılığından contempt'i çıkarır, kalan 7'yi toplamı 1 olacak şekilde ölçekler."""
    d = {ad: float(p) for ad, p in zip(HSE_SIRA, olasiliklar)}
    d.pop("contempt")
    toplam = sum(d.values()) or 1.0
    return {k: d[k] / toplam for k in ORTAK}


def fer_ortak(fer_modeli, kare, kutu):
    """fer'e yüzü kendisi aramasın diye YuNet'in kutusunu veriyoruz."""
    sonuc = fer_modeli.detect_emotions(kare, face_rectangles=[kutu])
    if not sonuc:
        return None
    e = sonuc[0]["emotions"]
    return {k: float(e[k]) for k in ORTAK}


def en_iyi(olas):
    return max(olas, key=olas.get)


def panel_ciz(kare, baslik, olas, x0):
    """x0 konumuna, başlık + 7 duygu çubuğundan oluşan bir panel çizer."""
    yukseklik = 40 + 22 * len(ORTAK)
    cv2.rectangle(kare, (x0, 0), (x0 + 300, yukseklik), (30, 30, 30), -1)
    cv2.putText(kare, baslik, (x0 + 8, 20), FONT, 0.55, (0, 255, 255), 1)
    kazanan = en_iyi(olas)
    for i, k in enumerate(ORTAK):
        y = 45 + i * 22
        renk = (0, 255, 0) if k == kazanan else (0, 200, 255)
        cv2.putText(kare, TR[k], (x0 + 8, y), FONT, 0.5, (255, 255, 255), 1)
        cv2.rectangle(kare, (x0 + 100, y - 12), (x0 + 100 + int(olas[k] * 150), y), renk, -1)
        cv2.putText(kare, f"{olas[k] * 100:.0f}%", (x0 + 255, y), FONT, 0.45, (255, 255, 255), 1)


def csv_hazirla():
    if not os.path.exists(CSV_DOSYA):
        with open(CSV_DOSYA, "w", newline="", encoding="utf-8") as f:
            baslik = ["zaman", "gercek", "fer_tahmin", "hse_tahmin"]
            baslik += [f"fer_{k}" for k in ORTAK] + [f"hse_{k}" for k in ORTAK]
            csv.writer(f).writerow(baslik)


def csv_ekle(gercek, fer_olas, hse_olas):
    with open(CSV_DOSYA, "a", newline="", encoding="utf-8") as f:
        satir = [datetime.now().isoformat(timespec="seconds"), gercek,
                 en_iyi(fer_olas), en_iyi(hse_olas)]
        satir += [f"{fer_olas[k]:.4f}" for k in ORTAK] + [f"{hse_olas[k]:.4f}" for k in ORTAK]
        csv.writer(f).writerow(satir)


def main():
    kamera = cv2.VideoCapture(0)
    genislik = int(kamera.get(cv2.CAP_PROP_FRAME_WIDTH))
    yukseklik = int(kamera.get(cv2.CAP_PROP_FRAME_HEIGHT))

    dedektor = cv2.FaceDetectorYN.create(YUZ_MODELI, "", (genislik, yukseklik), score_threshold=0.8)
    print("Modeller yukleniyor (TensorFlow ilk acilista biraz yavas)...")
    fer_modeli = FER()
    hse_modeli = HSEmotionRecognizer(model_name="enet_b0_8_best_afew")
    csv_hazirla()

    kayit_sayisi, ekran_no = 0, 0
    mesaj, mesaj_sure = "", 0

    while True:
        basarili, kare = kamera.read()
        if not basarili:
            print("Kameradan goruntu alinamadi")
            break

        fer_olas = hse_olas = None
        _, yuzler = dedektor.detect(kare)

        if yuzler is not None:
            yuz = yuzler[0]                      # karşılaştırma tek yüz üzerinden
            x, y, w, h = yuz[:4].astype(int)
            x, y = max(0, x), max(0, y)

            # HSEmotion: %15 paylı kırpım, RGB
            H, W = kare.shape[:2]
            dx, dy = int(w * 0.15), int(h * 0.15)
            kirpik = kare[max(0, y - dy):min(H, y + h + dy), max(0, x - dx):min(W, x + w + dx)]
            if kirpik.size > 0:
                _, ham = hse_modeli.predict_emotions(cv2.cvtColor(kirpik, cv2.COLOR_BGR2RGB), logits=False)
                hse_olas = hse_ortak(ham)

            # fer: tüm kare + YuNet kutusu (fer kendi içinde gri tonlamaya çevirir)
            fer_olas = fer_ortak(fer_modeli, kare, (int(x), int(y), int(w), int(h)))

            cv2.rectangle(kare, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if fer_olas:
            panel_ciz(kare, "FER (FER-2013)", fer_olas, 0)
        if hse_olas:
            panel_ciz(kare, "HSEmotion (AffectNet)", hse_olas, kare.shape[1] - 300)

        cv2.putText(kare, f"Kayit: {kayit_sayisi}  | 1-7: etiketle  s: ekran  q: cikis",
                    (10, kare.shape[0] - 15), FONT, 0.5, (255, 255, 255), 1)
        if mesaj_sure > 0:
            cv2.putText(kare, mesaj, (10, kare.shape[0] - 40), FONT, 0.7, (0, 255, 0), 2)
            mesaj_sure -= 1

        cv2.imshow("FER vs HSEmotion", kare)
        tus = cv2.waitKey(1) & 0xFF

        if tus == ord("q"):
            break
        if tus == ord("s"):
            ekran_no += 1
            cv2.imwrite(f"karsilastirma_{ekran_no}.png", kare)
            mesaj, mesaj_sure = f"Ekran kaydedildi: karsilastirma_{ekran_no}.png", 30
        if tus in TUSLAR:
            if fer_olas and hse_olas:
                gercek = TUSLAR[tus]
                csv_ekle(gercek, fer_olas, hse_olas)
                kayit_sayisi += 1
                mesaj = (f"{TR[gercek]} kaydedildi | FER: {TR[en_iyi(fer_olas)]}"
                         f" | HSE: {TR[en_iyi(hse_olas)]}")
                mesaj_sure = 30
            else:
                mesaj, mesaj_sure = "Yuz bulunamadi, kayit yapilmadi", 30

    kamera.release()
    cv2.destroyAllWindows()
    print(f"{kayit_sayisi} kayit {CSV_DOSYA} dosyasina eklendi.")


if __name__ == "__main__":
    main()
