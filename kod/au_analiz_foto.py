"""
OpenFace çıktısından fotoğraf veri seti için Action Unit (AU) analizi
=====================================================================
Girdi:  ciktilar/openface_foto/<duygu>/*.csv
        (OpenFace FaceLandmarkImg her fotoğraf için bir CSV yazar: face, confidence, AU01_r, ...)
Çıktı (ciktilar/ içinde):
  au_isi_haritasi_foto.png  her duyguda her AU'nun ortalama yoğunluğu (ısı haritası)
  au_ozet_foto.txt          her duyguda en güçlü 3 AU ve FACS'a göre beklenen AU'ların ölçülen değeri

Ortam: env_fer        Çalıştır: python au_analiz_foto.py   (foto_deneyleri.py bunu kendisi çağırır)
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

KLASOR = Path(__file__).resolve().parent
GIRDI = KLASOR / "ciktilar" / "openface_foto"
CIKTI = KLASOR / "ciktilar"
GUVEN_ESIGI = 0.5     # OpenFace'in yüze güveni bunun altındaysa o fotoğraf atılır

DUYGULAR = {"mutluluk": "Mutluluk", "saskinlik": "Şaşkınlık", "uzuntu": "Üzüntü", "ofke": "Öfke",
            "korku": "Korku", "igrenme": "İğrenme", "notr": "Nötr"}
AU_ACIKLAMA = {
    "AU01": "Kaş içi kalkar", "AU02": "Kaş dışı kalkar", "AU04": "Kaş çatılır",
    "AU05": "Üst göz kapağı kalkar", "AU06": "Yanak kalkar", "AU07": "Göz kapağı gerilir",
    "AU09": "Burun kırışır", "AU10": "Üst dudak kalkar", "AU12": "Dudak köşesi çekilir",
    "AU14": "Gamze", "AU15": "Dudak köşesi iner", "AU17": "Çene kalkar",
    "AU20": "Dudak gerilir", "AU23": "Dudak sıkılır", "AU25": "Dudaklar ayrılır",
    "AU26": "Çene düşer", "AU45": "Göz kırpma",
}
# FACS'a göre her duygu için beklenen AU'lar (raporun 2.3 bölümü)
BEKLENEN = {"mutluluk": ["AU06", "AU12"], "saskinlik": ["AU01", "AU02", "AU05", "AU26"],
            "uzuntu": ["AU01", "AU04", "AU15"], "ofke": ["AU04", "AU07", "AU23"],
            "korku": ["AU01", "AU02", "AU04", "AU05", "AU20"], "igrenme": ["AU09", "AU10"], "notr": []}


def fotograf_satiri(dosya):
    """Bir fotoğrafın CSV'sinden en güvenilir yüzün satırını döndürür (yoksa None)."""
    with open(dosya, encoding="utf-8") as f:
        okuyucu = csv.reader(f)
        basliklar = [b.strip() for b in next(okuyucu)]       # sütun adlarının başında boşluk olur
        satirlar = [dict(zip(basliklar, [d.strip() for d in s])) for s in okuyucu if s]
    satirlar = [s for s in satirlar if float(s.get("confidence", 0)) >= GUVEN_ESIGI]
    return max(satirlar, key=lambda s: float(s["confidence"])) if satirlar else None


def main():
    au_adlari, ifadeler, matris, sayilar = None, [], [], {}
    for klasor_adi, ad in DUYGULAR.items():
        klasor = GIRDI / klasor_adi
        if not klasor.exists():
            continue
        satirlar = [s for s in (fotograf_satiri(d) for d in sorted(klasor.glob("*.csv"))) if s]
        if not satirlar:
            continue
        if au_adlari is None:
            au_adlari = sorted(k[:-2] for k in satirlar[0] if k.startswith("AU") and k.endswith("_r"))
        ifadeler.append(klasor_adi)
        sayilar[klasor_adi] = len(satirlar)
        matris.append([np.mean([float(s[f"{a}_r"]) for s in satirlar]) for a in au_adlari])

    if not ifadeler:
        raise SystemExit(f"OpenFace çıktısı bulunamadı: {GIRDI}")
    matris = np.array(matris)

    # --- Isı haritası
    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(matris, cmap="YlOrRd", aspect="auto", vmin=0)
    ax.set_xticks(range(len(au_adlari)))
    ax.set_xticklabels([f"{a}\n{AU_ACIKLAMA.get(a, '')}" for a in au_adlari], rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(ifadeler)))
    ax.set_yticklabels([f"{DUYGULAR[i]} (n={sayilar[i]})" for i in ifadeler])
    for i in range(matris.shape[0]):
        for j in range(matris.shape[1]):
            ax.text(j, i, f"{matris[i, j]:.1f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="Ortalama yoğunluk (0-5)")
    ax.set_title("Fotoğraf veri setinde her ifadenin ortalama Action Unit yoğunluğu (OpenFace 2.2)")
    plt.tight_layout()
    plt.savefig(CIKTI / "au_isi_haritasi_foto.png", dpi=150)

    # --- Metin özeti
    satirlar = ["Her ifadede en güçlü 3 Action Unit ve FACS'a göre beklenenlerin ölçülen değeri", ""]
    for ifade, degerler in zip(ifadeler, matris):
        en_guclu = np.argsort(degerler)[::-1][:3]
        satirlar.append(f"{DUYGULAR[ifade]} (n={sayilar[ifade]})")
        satirlar.append("  En güçlü : " + ", ".join(
            f"{au_adlari[j]} {AU_ACIKLAMA.get(au_adlari[j], '')} ({degerler[j]:.2f})" for j in en_guclu))
        if BEKLENEN[ifade]:
            satirlar.append("  Beklenen : " + ", ".join(
                f"{a} ({degerler[au_adlari.index(a)]:.2f})" for a in BEKLENEN[ifade] if a in au_adlari))
    metin = "\n".join(satirlar)
    print(metin)
    (CIKTI / "au_ozet_foto.txt").write_text(metin, encoding="utf-8")
    print("\nKaydedildi: au_isi_haritasi_foto.png, au_ozet_foto.txt")


if __name__ == "__main__":
    main()
