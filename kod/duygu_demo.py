"""
Canlı duygu tanıma demosu (ana uygulama)
=======================================
Her kamera karesinde:
  1) YuNet ile yüzü bul (kutu + 5 yüz noktası)
  2) Yüzü %15 payla kırp, BGR'den RGB'ye çevir
  3) HSEmotion (AffectNet ile eğitilmiş EfficientNet-B0) ile
     - 8 duygu için olasılık        (kategorik model, Ekman + küçümseme)
     - valence ve arousal değerleri (boyutsal model, Russell)
  4) Ekrana çiz: kutu + etiket, 8 duygu çubuğu, valence-arousal dairesi

Uygulama senaryosu (müşteri hizmetleri / terapi görüşmesi):
  Program kapanırken bütün oturum "oturum_kaydi.csv" dosyasına yazılır.
  Ardından  python oturum_ozeti.py  ile görüşmenin duygu özeti çıkarılır.

Tuşlar:  q = çıkış,  s = ekran görüntüsü kaydet (ekran_1.png, ekran_2.png, ...)
Ortam:   env_fer  (opencv-python<5, hsemotion-onnx)
"""

import urllib.request  # hsemotion-onnx 0.3.1 hatası: modeli indirirken urllib.request'i import etmeyi unutuyor
import csv
import time

import cv2
import numpy as np
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
import os
from pathlib import Path

# Script hangi klasörden çalıştırılırsa çalıştırılsın:
# modeller kod/modeller'den okunur, çıktılar (CSV, PNG, video) kod/ciktilar'a yazılır.
KLASOR = Path(__file__).resolve().parent
CIKTI = KLASOR / "ciktilar"
CIKTI.mkdir(exist_ok=True)
os.chdir(CIKTI)

YUZ_MODELI = str(KLASOR / "modeller" / "face_detection_yunet_2023mar.onnx")
# "_va_mtl" = multi-task: 8 duygu olasılığına ek olarak valence ve arousal da tahmin eder
DUYGU_MODELI = "enet_b0_8_va_mtl"
# Modelin çıktı sırası: Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise
# (OpenCV ekrana Türkçe karakter yazamadığı için ASCII kullanıyoruz)
DUYGULAR = ["Ofke", "Kucumseme", "Igrenme", "Korku",
            "Mutluluk", "Notr", "Uzuntu", "Saskinlik"]
YUMUSATMA = 0.6        # 0 = yumuşatma yok; 1'e yaklaştıkça değerler daha yavaş değişir
OTURUM_CSV = "oturum_kaydi.csv"
FONT = cv2.FONT_HERSHEY_SIMPLEX


def yuzu_kirp(kare, x, y, w, h, pay=0.15):
    """Yüz kutusunu her yönden %15 genişletip kırpar, görüntü dışına taşmayı engeller."""
    H, W = kare.shape[:2]
    dx, dy = int(w * pay), int(h * pay)
    x1, y1 = max(0, x - dx), max(0, y - dy)
    x2, y2 = min(W, x + w + dx), min(H, y + h + dy)
    return kare[y1:y2, x1:x2]


def cubuklari_ciz(kare, olasiliklar):
    """Sol üst köşeye 8 duygunun olasılık çubuklarını çizer."""
    cv2.rectangle(kare, (0, 0), (310, 20 + 22 * len(DUYGULAR)), (30, 30, 30), -1)
    en_iyi = int(np.argmax(olasiliklar))
    for i, (ad, p) in enumerate(zip(DUYGULAR, olasiliklar)):
        y = 25 + i * 22
        renk = (0, 255, 0) if i == en_iyi else (0, 200, 255)
        cv2.putText(kare, ad, (10, y), FONT, 0.5, (255, 255, 255), 1)
        cv2.rectangle(kare, (110, y - 12), (110 + int(p * 150), y), renk, -1)
        cv2.putText(kare, f"{p * 100:.0f}%", (268, y), FONT, 0.45, (255, 255, 255), 1)


def daire_ciz(kare, valence, arousal):
    """Sağ alt köşeye Russell'ın dairesel modelini çizer.
    Yatay eksen valence (olumsuz <-> olumlu), dikey eksen arousal (sakin <-> heyecanlı)."""
    H, W = kare.shape[:2]
    r = 70
    cx, cy = W - r - 25, H - r - 45
    cv2.rectangle(kare, (cx - r - 20, cy - r - 40), (cx + r + 20, cy + r + 40), (30, 30, 30), -1)
    cv2.putText(kare, "Valence - Arousal", (cx - 68, cy - r - 22), FONT, 0.45, (0, 255, 255), 1)
    cv2.circle(kare, (cx, cy), r, (200, 200, 200), 1)
    cv2.line(kare, (cx - r, cy), (cx + r, cy), (120, 120, 120), 1)
    cv2.line(kare, (cx, cy - r), (cx, cy + r), (120, 120, 120), 1)
    cv2.putText(kare, "Heyecanli", (cx - 32, cy - r - 4), FONT, 0.4, (255, 255, 255), 1)
    cv2.putText(kare, "Sakin", (cx - 18, cy + r + 14), FONT, 0.4, (255, 255, 255), 1)
    cv2.putText(kare, "-", (cx - r - 14, cy + 5), FONT, 0.6, (0, 0, 255), 2)
    cv2.putText(kare, "+", (cx + r + 4, cy + 5), FONT, 0.6, (0, 255, 0), 2)

    v = float(np.clip(valence, -1, 1))
    a = float(np.clip(arousal, -1, 1))
    # Ekranda y ekseni aşağı doğru büyür; yüksek arousal yukarıda olsun diye eksi
    nokta = (int(cx + v * r), int(cy - a * r))
    cv2.circle(kare, nokta, 7, (0, 255, 0) if v >= 0 else (0, 0, 255), -1)
    cv2.putText(kare, f"V:{v:+.2f}  A:{a:+.2f}", (cx - 58, cy + r + 32), FONT, 0.45, (255, 255, 255), 1)


def oturumu_kaydet(kayitlar):
    """Oturum boyunca toplanan ham tahminleri CSV'ye yazar."""
    with open(OTURUM_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["zaman_sn"] + DUYGULAR + ["valence", "arousal"])
        w.writerows(kayitlar)


def main():
    kamera = cv2.VideoCapture(0)
    genislik = int(kamera.get(cv2.CAP_PROP_FRAME_WIDTH))
    yukseklik = int(kamera.get(cv2.CAP_PROP_FRAME_HEIGHT))

    dedektor = cv2.FaceDetectorYN.create(YUZ_MODELI, "", (genislik, yukseklik),
                                         score_threshold=0.8)
    duygu_modeli = HSEmotionRecognizer(model_name=DUYGU_MODELI)

    kayitlar = []          # [zaman, 8 olasılık, valence, arousal]
    duzgun = None          # ilk yüz için yumuşatılmış skorlar
    baslangic = time.time()
    kayit_no = 0

    while True:
        basarili, kare = kamera.read()
        if not basarili:
            print("Kameradan görüntü alınamadı")
            break

        _, yuzler = dedektor.detect(kare)

        if yuzler is not None:
            for sira, yuz in enumerate(yuzler):
                x, y, w, h = yuz[:4].astype(int)

                kirpik = yuzu_kirp(kare, x, y, w, h)
                if kirpik.size == 0:
                    continue

                # OpenCV BGR verir, model RGB bekler
                rgb = cv2.cvtColor(kirpik, cv2.COLOR_BGR2RGB)
                # va_mtl modelinde 10 değer döner: 8 duygu olasılığı + valence + arousal
                _, skorlar = duygu_modeli.predict_emotions(rgb, logits=False)
                skorlar = np.asarray(skorlar, dtype=float)

                if sira == 0:
                    # Kareden kareye titremeyi azaltmak için üstel hareketli ortalama
                    duzgun = skorlar if duzgun is None else YUMUSATMA * duzgun + (1 - YUMUSATMA) * skorlar
                    gosterilen = duzgun
                    kayitlar.append([round(time.time() - baslangic, 3)] + [round(s, 4) for s in skorlar])
                else:
                    gosterilen = skorlar

                olasiliklar = gosterilen[:8]
                en_iyi = int(np.argmax(olasiliklar))
                etiket = f"{DUYGULAR[en_iyi]} {olasiliklar[en_iyi] * 100:.0f}%"

                cv2.rectangle(kare, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(kare, etiket, (x, y - 10), FONT, 0.7, (0, 255, 0), 2)
                for i in range(4, 14, 2):   # YuNet'in 5 yüz noktası
                    cv2.circle(kare, (int(yuz[i]), int(yuz[i + 1])), 3, (0, 0, 255), -1)

                if sira == 0:  # paneller sadece ilk yüz için
                    cubuklari_ciz(kare, olasiliklar)
                    daire_ciz(kare, gosterilen[8], gosterilen[9])

        cv2.imshow("Duygu Tanima - q: cikis, s: ekran goruntusu", kare)
        tus = cv2.waitKey(1) & 0xFF
        if tus == ord("q"):
            break
        if tus == ord("s"):
            kayit_no += 1
            cv2.imwrite(f"ekran_{kayit_no}.png", kare)
            print(f"Kaydedildi: ekran_{kayit_no}.png")

    kamera.release()
    cv2.destroyAllWindows()

    if kayitlar:
        oturumu_kaydet(kayitlar)
        print(f"{len(kayitlar)} kare {OTURUM_CSV} dosyasına kaydedildi.")
        print("Oturum özeti için:  python oturum_ozeti.py")


if __name__ == "__main__":
    main()
