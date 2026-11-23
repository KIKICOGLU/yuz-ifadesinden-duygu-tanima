"""
Fotoğraf veri setini indirir (Pexels)
=====================================
veri/foto_listesi.csv dosyasındaki fotoğrafları Pexels'ten indirir ve
veri/fotograflar/<duygu>/ klasörlerine koyar.

- Fotoğraflar elle (gözle) seçildi: her biri tek bir yetişkin yüzü ve
  o duyguyu açıkça gösteren, poz verilmiş stok fotoğraflardır. Klasör adı = gerçek etiket.
- Lisans: Pexels Lisansı (ücretsiz kullanım, atıf zorunlu değil). Yine de her fotoğrafın
  fotoğrafçısı ve kaynak sayfası veri/kaynaklar.csv dosyasına yazılır.
- Fotoğraflar 640 piksel genişlikte indirilir (yüz tespiti ve duygu modelleri için yeterli).

Çalıştır (internet gerekir, ~1 dakika):   python foto_indir.py
"""

import csv
import time
import urllib.request
from pathlib import Path

KLASOR = Path(__file__).resolve().parent
VERI = KLASOR / "veri"
LISTE = VERI / "foto_listesi.csv"
HEDEF = VERI / "fotograflar"
UA = "Mozilla/5.0 (DuyguTanimaOgrenciProjesi; egitim amacli)"
ADRES = "https://images.pexels.com/photos/{id}/pexels-photo-{id}.jpeg?auto=compress&cs=tinysrgb&w=640"


def main():
    with open(LISTE, encoding="utf-8") as f:
        liste = list(csv.DictReader(f))

    sayac, kaynaklar, hatalar = {}, [], []
    for satir in liste:
        duygu = satir["duygu"]
        sayac[duygu] = sayac.get(duygu, 0) + 1
        klasor = HEDEF / duygu
        klasor.mkdir(parents=True, exist_ok=True)
        dosya = klasor / f"{duygu}_{sayac[duygu]:02d}.jpg"

        if not dosya.exists():
            istek = urllib.request.Request(ADRES.format(id=satir["pexels_id"]), headers={"User-Agent": UA})
            try:
                with urllib.request.urlopen(istek, timeout=30) as yanit:
                    dosya.write_bytes(yanit.read())
                time.sleep(0.3)
            except Exception as hata:
                hatalar.append(f"{satir['pexels_id']} ({duygu}): {hata}")
                continue

        kaynaklar.append({"dosya": dosya.relative_to(HEDEF).as_posix(), "duygu": duygu,
                          "fotografci": satir["fotografci"], "kaynak": satir["kaynak"],
                          "lisans": "Pexels License"})

    with open(VERI / "kaynaklar.csv", "w", newline="", encoding="utf-8") as f:
        yazici = csv.DictWriter(f, fieldnames=["dosya", "duygu", "fotografci", "kaynak", "lisans"])
        yazici.writeheader()
        yazici.writerows(kaynaklar)

    for duygu, n in sayac.items():
        print(f"{duygu:<10}: {n} fotoğraf")
    print(f"\nToplam {len(kaynaklar)} fotoğraf hazır: {HEDEF}")
    if hatalar:
        print("\nİndirilemeyenler:\n  " + "\n  ".join(hatalar))
    print("\nSonraki adım:  python foto_deneyleri.py")


if __name__ == "__main__":
    main()
