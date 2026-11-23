"""
Fotoğraf değerlendirme
======================
İnternetten indirilen ya da önceden çekilmiş fotoğraflardaki yüzlerin duygusunu tahmin eder.
Bir ya da birkaç dosya, ya da bir klasör (alt klasörleriyle birlikte) verilebilir.
Fotoğraftaki her yüz ayrı ayrı değerlendirilir.

Çalıştırma (env_fer ortamında):
  python foto_degerlendir.py                        -> dosya seçme penceresi açılır
  python foto_degerlendir.py resim.jpg
  python foto_degerlendir.py C:\\Fotograflar          -> klasördeki bütün fotoğraflar
  python foto_degerlendir.py C:\\Fotograflar --fer    -> FER (FER-2013) ile de karşılaştırır

Çıktılar: panel/ciktilar/foto_<tarih>/
  sonuclar.csv      her yüz için tahmin, olasılıklar, valence ve arousal
  001_<ad>.jpg ...  kutu ve tahmin yazılmış fotoğraflar
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

import motor

UZANTILAR = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif", ".tif", ".tiff"}
EN_BUYUK_KENAR = 1600     # telefon fotoğrafları gibi büyük görüntüler bu boyuta küçültülür


def girdileri_topla(yollar):
    dosyalar = []
    for yol in yollar:
        p = Path(yol.strip().strip('"'))
        if p.is_dir():
            dosyalar += sorted(f for f in p.rglob("*") if f.suffix.lower() in UZANTILAR)
        elif p.is_file() and p.suffix.lower() in UZANTILAR:
            dosyalar.append(p)
        elif str(p):
            print(f"Atlandı (bulunamadı ya da fotoğraf değil): {p}")
    return dosyalar


def dosya_sec():
    """Script argümansız çalıştırılırsa (ör. VS Code'da ▶) fotoğraf seçme penceresi açar."""
    import tkinter as tk
    from tkinter import filedialog
    kok = tk.Tk()
    kok.withdraw()
    secim = filedialog.askopenfilenames(
        title="Değerlendirilecek fotoğrafları seç",
        filetypes=[("Fotoğraflar", "*.jpg *.jpeg *.png *.bmp *.webp *.jfif *.tif *.tiff"),
                   ("Tüm dosyalar", "*.*")])
    kok.destroy()
    return list(secim)


def main():
    ayr = argparse.ArgumentParser(description="Fotoğraflardaki yüzlerin duygusunu tahmin eder.")
    ayr.add_argument("girdiler", nargs="*", help="fotoğraf dosyaları ya da klasörler")
    ayr.add_argument("--fer", action="store_true", help="FER (FER-2013) ile de karşılaştır (daha yavaş)")
    ayr.add_argument("--dur-dosyasi", help="(panel için) bu dosya oluşunca iş düzgünce durdurulur")
    arg = ayr.parse_args()

    girdiler = arg.girdiler or dosya_sec()
    dosyalar = girdileri_topla(girdiler)
    if not dosyalar:
        raise SystemExit("Değerlendirilecek fotoğraf bulunamadı.")

    klasor = motor.cikti_klasoru("foto")
    motor.cikti_bildir(klasor)
    print(f"{len(dosyalar)} fotoğraf değerlendirilecek. Çıktılar: {klasor}")
    model = motor.DuyguMotoru(esik=0.6, fer_kullan=arg.fer)

    satirlar, sayac, yuzsuz = [], Counter(), 0
    for sira, dosya in enumerate(dosyalar, start=1):
        if motor.durdurulsun_mu(arg.dur_dosyasi):
            print("Durduruldu; o ana kadarki sonuçlar kaydediliyor.")
            break
        motor.ilerleme(sira - 1, len(dosyalar))
        goruntu = motor.resim_oku(dosya)
        if goruntu is None:
            print(f"  ! okunamadı: {dosya.name}")
            continue
        goruntu = motor.kucult(goruntu, EN_BUYUK_KENAR)
        isaretli = goruntu.copy()
        cikti_adi = f"{sira:03d}_{motor.guvenli_ad(dosya.stem)}.jpg"
        yuzler = model.yuzleri_bul(goruntu)

        if not yuzler:
            yuzsuz += 1
            motor.etiket_yaz(isaretli, "Yuz bulunamadi", 8, 40, motor.TURUNCU, motor.olcek_hesapla(isaretli))
            satirlar.append({"dosya": str(dosya), "isaretli": cikti_adi, "yuz_no": 0, "tahmin": "Yüz yok"})
            print(f"  - {dosya.name}: yüz bulunamadı")

        for no, yuz in enumerate(yuzler, start=1):
            skorlar = model.tahmin(goruntu, yuz)
            if skorlar is None:
                continue
            fer = model.fer_tahmin(goruntu, yuz)
            motor.yuz_ciz(isaretli, yuz, skorlar, no=no if len(yuzler) > 1 else None, fer=fer)
            i, p = motor.en_olasi(skorlar)
            sayac[motor.DUYGULAR_TR[i]] += 1
            x, y, w, h = [int(v) for v in yuz[:4]]
            satir = {"dosya": str(dosya), "isaretli": cikti_adi, "yuz_no": no,
                     "x": x, "y": y, "w": w, "h": h, "yuz_skoru": f"{float(yuz[14]):.3f}",
                     "tahmin": motor.DUYGULAR_TR[i], "olasilik": f"{p:.4f}",
                     "valence": f"{skorlar[8]:+.4f}", "arousal": f"{skorlar[9]:+.4f}"}
            satir.update({ad: f"{s:.4f}" for ad, s in zip(motor.DUYGULAR_ASCII, skorlar[:8])})
            if fer:
                k = max(fer, key=fer.get)
                satir.update({"fer_tahmin": motor.FER_TR.get(k, k), "fer_olasilik": f"{fer[k]:.4f}"})
            elif arg.fer:
                satir["fer_tahmin"] = "işlenemedi"
            satirlar.append(satir)
            print(f"  {dosya.name}" + (f" [yüz {no}]" if len(yuzler) > 1 else "")
                  + f": {motor.DUYGULAR_TR[i]} %{p * 100:.0f}  (V {skorlar[8]:+.2f}, A {skorlar[9]:+.2f})"
                  + (f"  | FER: {satir['fer_tahmin']}" if arg.fer else ""))

        motor.resim_yaz(klasor / cikti_adi, isaretli)

    motor.ilerleme(len(dosyalar), len(dosyalar))
    kolonlar = (["dosya", "isaretli", "yuz_no", "x", "y", "w", "h", "yuz_skoru", "tahmin", "olasilik",
                 "valence", "arousal"] + motor.DUYGULAR_ASCII
                + (["fer_tahmin", "fer_olasilik"] if arg.fer else []))
    # utf-8-sig: Excel'de açınca Türkçe karakterler düzgün görünsün
    with open(klasor / "sonuclar.csv", "w", newline="", encoding="utf-8-sig") as f:
        yazici = csv.DictWriter(f, fieldnames=kolonlar, restval="")
        yazici.writeheader()
        yazici.writerows(satirlar)

    yuz_sayisi = sum(sayac.values())
    print(f"\nBitti: {len(dosyalar)} fotoğraf, {yuz_sayisi} yüz" + (f", {yuzsuz} fotoğrafta yüz yok" if yuzsuz else ""))
    if sayac:
        print("Tahmin dağılımı: " + ", ".join(f"{ad} {n}" for ad, n in sayac.most_common()))
    print(f"Sonuçlar: {klasor / 'sonuclar.csv'}")


if __name__ == "__main__":
    main()
