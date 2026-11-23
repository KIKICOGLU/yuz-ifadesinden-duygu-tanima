"""
Fotoğraf veri seti üzerinde deneyler (kameraya gerek yok)
=========================================================
Girdi:  veri/fotograflar/<duygu>/*.jpg|png   (klasör adı = fotoğrafın gerçek etiketi)
        Fotoğraflar Pexels'ten (Pexels Lisansı); fotoğrafçılar ve bağlantılar veri/kaynaklar.csv'de.

Her fotoğraf için:
  1) Haar Cascade ve YuNet ile yüz tespiti          -> Deney 1 (dedektör karşılaştırması)
  2) YuNet'in en güvenli yüzü kırpılır (%15 pay)
  3) HSEmotion (AffectNet) ve fer (FER-2013) tahmini -> Deney 2 (model karşılaştırması)
  4) Kutulu, etiketli örnek görüntü kaydedilir      -> rapor ve sunum görselleri

Çıktılar (ciktilar/ içinde):
  foto_tespit.csv          her fotoğrafta Haar ve YuNet'in kaç yüz bulduğu
  foto_tespit_ozet.txt     dedektörlerin tespit oranları
  foto_sonuclari.csv       iki modelin tahminleri (degerlendirme.py ile aynı biçim)
  foto_ornekler/           kutu ve tahmin yazılmış fotoğraflar
  foto_ornek_izgara.png    her duygudan bir örnek, tek görselde

Ortam: env_fer        Çalıştır: python foto_analiz.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # TensorFlow'un uzun log mesajlarını gizle

import urllib.request  # hsemotion-onnx 0.3.1 hatası: urllib.request'i import etmeyi unutuyor
import csv
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from fer.fer import FER
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

KLASOR = Path(__file__).resolve().parent
FOTOGRAFLAR = KLASOR / "veri" / "fotograflar"
CIKTI = KLASOR / "ciktilar"
ORNEKLER = CIKTI / "foto_ornekler"
YUZ_MODELI = str(KLASOR / "modeller" / "face_detection_yunet_2023mar.onnx")
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Klasör adı -> ortak İngilizce etiket (degerlendirme.py ve karsilastirma_canli.py ile aynı)
KLASOR_ETIKET = {"mutluluk": "happy", "uzuntu": "sad", "ofke": "angry", "saskinlik": "surprise",
                 "korku": "fear", "igrenme": "disgust", "notr": "neutral"}
ORTAK = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
TR = {"angry": "Ofke", "disgust": "Igrenme", "fear": "Korku", "happy": "Mutluluk",
      "sad": "Uzuntu", "surprise": "Saskinlik", "neutral": "Notr"}
HSE_SIRA = ["angry", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
TR_BASLIK = {"angry": "Öfke", "disgust": "İğrenme", "fear": "Korku", "happy": "Mutluluk",
             "sad": "Üzüntü", "surprise": "Şaşkınlık", "neutral": "Nötr"}   # grafik başlıkları (matplotlib Türkçe yazabilir)


def hse_ortak(olasiliklar):
    """HSEmotion'ın 8 olasılığından contempt'i çıkarır, kalan 7'yi toplamı 1 olacak şekilde ölçekler."""
    d = {ad: float(p) for ad, p in zip(HSE_SIRA, olasiliklar)}
    d.pop("contempt")
    toplam = sum(d.values()) or 1.0
    return {k: d[k] / toplam for k in ORTAK}


def en_iyi(olas):
    return max(olas, key=olas.get)


def yuzu_kirp(goruntu, x, y, w, h, pay=0.15):
    H, W = goruntu.shape[:2]
    dx, dy = int(w * pay), int(h * pay)
    return goruntu[max(0, y - dy):min(H, y + h + dy), max(0, x - dx):min(W, x + w + dx)]


def etiket_yaz(goruntu, metin, x, y, renk):
    """Okunabilsin diye yazının arkasına koyu bir kutu çizer."""
    (gen, yuk), _ = cv2.getTextSize(metin, FONT, 0.6, 2)
    cv2.rectangle(goruntu, (x, y - yuk - 8), (x + gen + 8, y + 4), (30, 30, 30), -1)
    cv2.putText(goruntu, metin, (x + 4, y - 2), FONT, 0.6, renk, 2)


def fotograflari_bul():
    liste = []
    for klasor_adi, etiket in KLASOR_ETIKET.items():
        klasor = FOTOGRAFLAR / klasor_adi
        if not klasor.exists():
            continue
        for dosya in sorted(klasor.iterdir()):
            if dosya.suffix.lower() in (".jpg", ".jpeg", ".png"):
                liste.append((dosya, etiket))
    return liste


def main():
    fotograflar = fotograflari_bul()
    if not fotograflar:
        raise SystemExit(f"Fotoğraf bulunamadı: {FOTOGRAFLAR}")
    CIKTI.mkdir(exist_ok=True)
    ORNEKLER.mkdir(exist_ok=True)

    print("Modeller yukleniyor...")
    haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    yunet = cv2.FaceDetectorYN.create(YUZ_MODELI, "", (320, 320), score_threshold=0.6)
    hse = HSEmotionRecognizer(model_name="enet_b0_8_best_afew")
    fer_modeli = FER()

    tespit_satirlari, sonuc_satirlari, izgara = [], [], {}

    for dosya, gercek in fotograflar:
        goruntu = cv2.imread(str(dosya))
        if goruntu is None:
            print(f"  ! okunamadı: {dosya.name}")
            continue
        H, W = goruntu.shape[:2]
        goreli = dosya.relative_to(FOTOGRAFLAR).as_posix()

        # --- Deney 1: iki dedektör aynı fotoğrafta
        gri = cv2.cvtColor(goruntu, cv2.COLOR_BGR2GRAY)
        haar_yuzler = haar.detectMultiScale(gri, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        yunet.setInputSize((W, H))                 # YuNet'e her fotoğrafın kendi boyutunu söyle
        _, yunet_yuzler = yunet.detect(goruntu)
        yunet_sayi = 0 if yunet_yuzler is None else len(yunet_yuzler)
        en_skor = 0.0 if yunet_sayi == 0 else float(yunet_yuzler[:, 14].max())
        tespit_satirlari.append([goreli, gercek, len(haar_yuzler), yunet_sayi, round(en_skor, 3)])

        if yunet_sayi == 0:
            print(f"  - YuNet yüz bulamadı: {goreli}")
            continue

        # --- Deney 2: en güvenli yüz iki modele de aynı kutuyla verilir
        yuz = yunet_yuzler[int(np.argmax(yunet_yuzler[:, 14]))]
        x, y, w, h = [int(v) for v in yuz[:4]]
        x, y = max(0, x), max(0, y)
        kirpik = yuzu_kirp(goruntu, x, y, w, h)
        if kirpik.size == 0:
            continue
        _, ham = hse.predict_emotions(cv2.cvtColor(kirpik, cv2.COLOR_BGR2RGB), logits=False)
        hse_olas = hse_ortak(ham)
        fer_sonuc = fer_modeli.detect_emotions(goruntu, face_rectangles=[(x, y, w, h)])
        if not fer_sonuc:
            print(f"  - fer bu yüzü işleyemedi: {goreli}")
            continue
        fer_olas = {k: float(fer_sonuc[0]["emotions"][k]) for k in ORTAK}

        sonuc_satirlari.append([goreli, gercek, en_iyi(fer_olas), en_iyi(hse_olas)]
                               + [f"{fer_olas[k]:.4f}" for k in ORTAK]
                               + [f"{hse_olas[k]:.4f}" for k in ORTAK])

        # --- Örnek görüntü: kutu, 5 nokta, iki modelin tahmini
        ornek = goruntu.copy()
        cv2.rectangle(ornek, (x, y), (x + w, y + h), (0, 255, 0), 2)
        for i in range(4, 14, 2):
            cv2.circle(ornek, (int(yuz[i]), int(yuz[i + 1])), 3, (0, 0, 255), -1)
        hk, fk = en_iyi(hse_olas), en_iyi(fer_olas)
        etiket_yaz(ornek, f"HSEmotion: {TR[hk]} {hse_olas[hk] * 100:.0f}%", 8, 28,
                   (0, 255, 0) if hk == gercek else (0, 165, 255))
        etiket_yaz(ornek, f"FER: {TR[fk]} {fer_olas[fk] * 100:.0f}%", 8, 58,
                   (0, 255, 0) if fk == gercek else (0, 165, 255))
        etiket_yaz(ornek, f"Gercek: {TR[gercek]}", 8, H - 12, (255, 255, 255))
        cv2.imwrite(str(ORNEKLER / dosya.name), ornek)
        izgara.setdefault(gercek, ornek)          # her duygudan ilk örnek ızgaraya

    # --- Kayıtlar
    with open(CIKTI / "foto_tespit.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dosya", "gercek", "haar_yuz_sayisi", "yunet_yuz_sayisi", "yunet_en_yuksek_skor"])
        w.writerows(tespit_satirlari)

    with open(CIKTI / "foto_sonuclari.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dosya", "gercek", "fer_tahmin", "hse_tahmin"]
                   + [f"fer_{k}" for k in ORTAK] + [f"hse_{k}" for k in ORTAK])
        w.writerows(sonuc_satirlari)

    n = len(tespit_satirlari)
    haar_bir = sum(1 for s in tespit_satirlari if s[2] == 1)
    haar_hic = sum(1 for s in tespit_satirlari if s[2] == 0)
    haar_fazla = sum(1 for s in tespit_satirlari if s[2] > 1)
    yunet_var = sum(1 for s in tespit_satirlari if s[3] >= 1)
    skorlar = [s[4] for s in tespit_satirlari if s[3] >= 1]
    ozet = "\n".join([
        f"Fotoğraf sayısı                 : {n}",
        f"Haar  - tam bir yüz buldu       : {haar_bir} (%{100 * haar_bir / n:.0f})",
        f"Haar  - hiç yüz bulamadı        : {haar_hic} (%{100 * haar_hic / n:.0f})",
        f"Haar  - birden fazla 'yüz' buldu: {haar_fazla} (%{100 * haar_fazla / n:.0f})  (tek kişilik fotoğraflarda yanlış alarm)",
        f"YuNet - en az bir yüz buldu     : {yunet_var} (%{100 * yunet_var / n:.0f})",
        f"YuNet - ortalama güven skoru    : {np.mean(skorlar):.2f}" if skorlar else "YuNet - skor yok",
        f"Duygu modellerine giren yüz     : {len(sonuc_satirlari)}",
    ])
    print("\n" + ozet)
    (CIKTI / "foto_tespit_ozet.txt").write_text(ozet, encoding="utf-8")

    # --- Izgara görseli (sunum için): her duygudan bir örnek
    if izgara:
        sira = [k for k in ["happy", "sad", "angry", "surprise", "fear", "disgust", "neutral"] if k in izgara]
        fig, eksenler = plt.subplots(2, 4, figsize=(16, 8.5))
        for eksen in eksenler.flat:
            eksen.axis("off")
        for eksen, k in zip(eksenler.flat, sira):
            eksen.imshow(cv2.cvtColor(izgara[k], cv2.COLOR_BGR2RGB))
            eksen.set_title(TR_BASLIK[k], fontsize=14)
        eksenler.flat[-1].text(0.02, 0.5,
                               "Yeşil: tahmin doğru\nTuruncu: tahmin yanlış\n\nFotoğraflar: Pexels (Pexels Lisansı)\nfotoğrafçılar: veri/kaynaklar.csv",
                               fontsize=12, va="center")
        plt.tight_layout()
        plt.savefig(CIKTI / "foto_ornek_izgara.png", dpi=120)

    print("\nKaydedildi: foto_tespit.csv, foto_tespit_ozet.txt, foto_sonuclari.csv, foto_ornekler/, foto_ornek_izgara.png")
    print("Sonraki adım:  python degerlendirme.py foto_sonuclari.csv")


if __name__ == "__main__":
    main()
