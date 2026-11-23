"""
OpenFace çıktısından Action Unit (AU) analizi
---------------------------------------------
Girdi:
  openface_cikti/ifadeler.csv  -> OpenFace FeatureExtraction çıktısı
  ifadeler_bolumler.csv        -> video_kaydet.py'nin kaydettiği ifade aralıkları
Çıktı:
  au_zaman.png          -> seçili AU'ların zamana göre yoğunluğu, ifade bölgeleri renkli
  au_isi_haritasi.png   -> her ifadede her AU'nun ortalama yoğunluğu (ısı haritası)
  au_ozet.txt           -> her ifadede en güçlü 3 AU (rapora yapıştırmak için)

Çalıştır (env_fer): python au_analiz.py
"""

import csv

import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# Script hangi klasörden çalıştırılırsa çalıştırılsın:
# modeller kod/modeller'den okunur, çıktılar (CSV, PNG, video) kod/ciktilar'a yazılır.
KLASOR = Path(__file__).resolve().parent
CIKTI = KLASOR / "ciktilar"
CIKTI.mkdir(exist_ok=True)
os.chdir(CIKTI)

OPENFACE_CSV = "openface_cikti/ifadeler.csv"
BOLUM_CSV = "ifadeler_bolumler.csv"

AU_ACIKLAMA = {
    "AU01": "Kaş içi kalkar", "AU02": "Kaş dışı kalkar", "AU04": "Kaş çatılır",
    "AU05": "Üst göz kapağı kalkar", "AU06": "Yanak kalkar", "AU07": "Göz kapağı gerilir",
    "AU09": "Burun kırışır", "AU10": "Üst dudak kalkar", "AU12": "Dudak köşesi çekilir",
    "AU14": "Gamze", "AU15": "Dudak köşesi iner", "AU17": "Çene kalkar",
    "AU20": "Dudak gerilir", "AU23": "Dudak sıkılır", "AU25": "Dudaklar ayrılır",
    "AU26": "Çene düşer", "AU45": "Göz kırpma",
}
# Zaman grafiğinde gösterilecek, duygularla en ilişkili AU'lar
SECILI = ["AU01", "AU02", "AU04", "AU06", "AU09", "AU12", "AU15", "AU26"]

# --- OpenFace CSV'sini oku (sütun adlarının başında boşluk olur, temizliyoruz)
with open(OPENFACE_CSV, encoding="utf-8") as f:
    okuyucu = csv.reader(f)
    basliklar = [b.strip() for b in next(okuyucu)]
    satirlar = [dict(zip(basliklar, s)) for s in okuyucu]

# Sadece yüzün güvenle bulunduğu kareler
satirlar = [s for s in satirlar if s["success"].strip() == "1" and float(s["confidence"]) >= 0.8]
au_adlari = [b[:-2] for b in basliklar if b.startswith("AU") and b.endswith("_r")]
kareler = np.array([int(s["frame"]) for s in satirlar])
au = {a: np.array([float(s[f"{a}_r"]) for s in satirlar]) for a in au_adlari}

with open(BOLUM_CSV, encoding="utf-8") as f:
    bolumler = [(r["ifade"], int(r["ilk_kare"]), int(r["son_kare"])) for r in csv.DictReader(f)]

# --- 1) Zaman grafiği
renkler = plt.cm.Pastel1(np.linspace(0, 1, len(bolumler)))
fig, ax = plt.subplots(figsize=(14, 6))
for (ifade, ilk, son), renk in zip(bolumler, renkler):
    ax.axvspan(ilk, son, color=renk, alpha=0.5)
    ax.text((ilk + son) / 2, 5.2, ifade, ha="center", fontsize=9)
for a in SECILI:
    if a in au:
        ax.plot(kareler, au[a], label=f"{a} ({AU_ACIKLAMA.get(a, '')})", linewidth=1.5)
ax.set_xlabel("Kare")
ax.set_ylabel("AU yoğunluğu (0-5)")
ax.set_ylim(0, 5.6)
ax.set_title("İfadelere göre Action Unit yoğunlukları (OpenFace 2.2)")
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
plt.tight_layout()
plt.savefig("au_zaman.png", dpi=150)

# --- 2) Isı haritası: her ifade için her AU'nun ortalaması
ifade_sirasi, matris = [], []
for ifade, ilk, son in bolumler:
    maske = (kareler >= ilk) & (kareler <= son)
    if maske.sum() == 0:
        continue
    ad = ifade if ifade not in ifade_sirasi else f"{ifade} (2)"
    ifade_sirasi.append(ad)
    matris.append([au[a][maske].mean() for a in au_adlari])
matris = np.array(matris)

fig, ax = plt.subplots(figsize=(14, 5))
im = ax.imshow(matris, cmap="YlOrRd", aspect="auto", vmin=0)
ax.set_xticks(range(len(au_adlari)))
ax.set_xticklabels([f"{a}\n{AU_ACIKLAMA.get(a, '')}" for a in au_adlari], rotation=60, ha="right", fontsize=7)
ax.set_yticks(range(len(ifade_sirasi)))
ax.set_yticklabels(ifade_sirasi)
for i in range(matris.shape[0]):
    for j in range(matris.shape[1]):
        ax.text(j, i, f"{matris[i, j]:.1f}", ha="center", va="center", fontsize=7)
fig.colorbar(im, ax=ax, label="Ortalama yoğunluk")
ax.set_title("Her ifadede ortalama Action Unit yoğunluğu")
plt.tight_layout()
plt.savefig("au_isi_haritasi.png", dpi=150)

# --- 3) Metin özeti
ozet = []
for ad, degerler in zip(ifade_sirasi, matris):
    en_iyi = np.argsort(degerler)[::-1][:3]
    parcalar = [f"{au_adlari[j]} {AU_ACIKLAMA.get(au_adlari[j], '')} ({degerler[j]:.2f})" for j in en_iyi]
    ozet.append(f"{ad:<14}: " + ", ".join(parcalar))
metin = "Her ifadede en güçlü 3 Action Unit\n" + "\n".join(ozet)
print(metin)
with open("au_ozet.txt", "w", encoding="utf-8") as f:
    f.write(metin)
print("\nKaydedildi: au_zaman.png, au_isi_haritasi.png, au_ozet.txt")
