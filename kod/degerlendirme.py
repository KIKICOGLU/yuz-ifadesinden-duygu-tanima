"""
Etiketli kayıtlardan iki modeli (fer ve HSEmotion) değerlendirir:
- Doğruluk (accuracy) ve makro F1
- Her duygu için precision / recall / F1 (classification report)
- Yan yana iki confusion matrix
- Rapora yapıştırmak için metin özeti

Ortam: env_fer
Çalıştır:
  python degerlendirme.py                      # canlı kayıtlar: sonuclar.csv
                                               #   -> confusion_matrix.png, degerlendirme_sonuclari.txt
  python degerlendirme.py foto_sonuclari.csv   # fotoğraf veri seti (foto_analiz.py çıktısı)
                                               #   -> foto_sonuclari_confusion_matrix.png, foto_sonuclari_degerlendirme.txt
"""

import csv
import sys
from collections import Counter

import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, classification_report,
                             ConfusionMatrixDisplay, confusion_matrix, f1_score)
import os
from pathlib import Path

# Script hangi klasörden çalıştırılırsa çalıştırılsın:
# modeller kod/modeller'den okunur, çıktılar (CSV, PNG, video) kod/ciktilar'a yazılır.
KLASOR = Path(__file__).resolve().parent
CIKTI = KLASOR / "ciktilar"
CIKTI.mkdir(exist_ok=True)
os.chdir(CIKTI)

# Komut satırında bir dosya verilirse onu kullan; çıktı adlarının başına o dosyanın adını ekle
CSV_DOSYA = sys.argv[1] if len(sys.argv) > 1 else "sonuclar.csv"
ONEK = "" if CSV_DOSYA == "sonuclar.csv" else Path(CSV_DOSYA).stem + "_"
ORTAK = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
TR = {"angry": "Öfke", "disgust": "İğrenme", "fear": "Korku", "happy": "Mutluluk",
      "sad": "Üzüntü", "surprise": "Şaşkınlık", "neutral": "Nötr"}

with open(CSV_DOSYA, encoding="utf-8") as f:
    satirlar = list(csv.DictReader(f))

if not satirlar:
    raise SystemExit(f"{CSV_DOSYA} boş. Önce karsilastirma_canli.py ya da foto_analiz.py ile veri topla.")

gercek = [s["gercek"] for s in satirlar]
fer = [s["fer_tahmin"] for s in satirlar]
hse = [s["hse_tahmin"] for s in satirlar]

# Sadece en az bir kaydı olan duyguları göster (boş satır/sütun olmasın)
etiketler = [k for k in ORTAK if k in set(gercek)]
adlar = [TR[k] for k in etiketler]

rapor = []
rapor.append(f"Toplam kayıt: {len(satirlar)}")
rapor.append("Duygu başına kayıt: " +
             ", ".join(f"{TR[k]}={n}" for k, n in Counter(gercek).most_common()))
rapor.append("")

for ad, tahmin in [("FER (FER-2013)", fer), ("HSEmotion (AffectNet)", hse)]:
    rapor.append(f"===== {ad} =====")
    rapor.append(f"Doğruluk (accuracy): {accuracy_score(gercek, tahmin):.3f}")
    rapor.append(f"Makro F1          : {f1_score(gercek, tahmin, labels=etiketler, average='macro', zero_division=0):.3f}")
    rapor.append(classification_report(gercek, tahmin, labels=etiketler,
                                       target_names=adlar, zero_division=0))

metin = "\n".join(rapor)
print(metin)
with open(f"{ONEK}degerlendirme" + ("_sonuclari" if not ONEK else "") + ".txt", "w", encoding="utf-8") as f:
    f.write(metin)

# Yan yana iki confusion matrix (satırlar normalize: her satır o duygunun yüzde kaçı neye gitti)
fig, eksenler = plt.subplots(1, 2, figsize=(14, 6))
for eksen, (ad, tahmin) in zip(eksenler, [("FER (FER-2013)", fer), ("HSEmotion (AffectNet)", hse)]):
    cm = confusion_matrix(gercek, tahmin, labels=etiketler, normalize="true")
    ConfusionMatrixDisplay(cm, display_labels=adlar).plot(
        ax=eksen, cmap="Blues", values_format=".2f", colorbar=False)
    eksen.set_title(f"{ad}\nDoğruluk: {accuracy_score(gercek, tahmin):.2f}")
    eksen.set_xlabel("Tahmin")
    eksen.set_ylabel("Gerçek etiket")
    eksen.tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig(f"{ONEK}confusion_matrix.png", dpi=150)
print(f"\nKaydedildi: {ONEK}confusion_matrix.png ve {ONEK}degerlendirme" + ("_sonuclari" if not ONEK else "") + ".txt")
