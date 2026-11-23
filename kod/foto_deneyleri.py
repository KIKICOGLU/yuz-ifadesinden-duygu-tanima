"""
Fotoğraf veri seti üzerindeki bütün deneyleri sırayla çalıştırır
===============================================================
  1) foto_analiz.py                       Haar–YuNet tespiti, fer ve HSEmotion tahminleri
  2) degerlendirme.py foto_sonuclari.csv   doğruluk, makro F1, confusion matrix
  3) OpenFace FaceLandmarkImg              her duygu klasörü için Action Unit'ler
  4) au_analiz_foto.py                     AU ısı haritası ve özet

Bütün çıktılar ciktilar/ klasörüne yazılır.

Ortam: env_fer        Çalıştır: python foto_deneyleri.py
OpenFace başka bir yerdeyse:     python foto_deneyleri.py "D:\\Araclar\\OpenFace\\FaceLandmarkImg.exe"

OpenFace yüz noktaları için varsayılan olarak CE-CLM modelini kullanır; bu modelin dosyaları
paketle gelmez (download_models.ps1 ile Dropbox'tan ~yüzlerce MB indirilir). Bu dosyalar yoksa
paketin içinde gelen CLNF modeli (-mloc model/main_clnf_general.txt) kullanılır. Action Unit
tahmini her iki modelle de çalışır; CE-CLM yüz noktalarında biraz daha isabetlidir.
"""

import subprocess
import sys
from pathlib import Path

KLASOR = Path(__file__).resolve().parent
FOTOGRAFLAR = KLASOR / "veri" / "fotograflar"
OPENFACE_CIKTI = KLASOR / "ciktilar" / "openface_foto"
PY = sys.executable          # şu an çalışan Python (env_fer içindeki)


def adim(baslik, komut, cwd=None):
    print(f"\n{'=' * 70}\n{baslik}\n{'=' * 70}")
    sonuc = subprocess.run(komut, cwd=cwd)
    if sonuc.returncode != 0:
        print(f"!! Bu adım hata verdi (çıkış kodu {sonuc.returncode}). Yukarıdaki mesaja bak.")
        return False
    return True


def openface_bul():
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    adaylar = ([KLASOR.parent / "araclar" / "OpenFace_2.2.0_win_x64" / "FaceLandmarkImg.exe",
                Path(r"C:\OpenFace\FaceLandmarkImg.exe")]
               + sorted(Path(r"C:\OpenFace").glob("*/FaceLandmarkImg.exe")))
    return next((a for a in adaylar if a.exists()), None)


def main():
    if not adim("1/4  Yüz tespiti ve duygu tahmini (foto_analiz.py)", [PY, str(KLASOR / "foto_analiz.py")]):
        return
    adim("2/4  Model değerlendirmesi (degerlendirme.py)", [PY, str(KLASOR / "degerlendirme.py"), "foto_sonuclari.csv"])

    exe = openface_bul()
    if exe is None or not exe.exists():
        print("\n!! OpenFace bulunamadı (araclar\\OpenFace_2.2.0_win_x64 ya da C:\\OpenFace). 3. ve 4. adımlar atlandı.")
        print("   OpenFace'i kurup bu scripti tekrar çalıştır ya da exe'nin yolunu argüman olarak ver.")
        return

    print(f"\n{'=' * 70}\n3/4  OpenFace Action Unit çıkarımı ({exe})\n{'=' * 70}")
    cen_var = (exe.parent / "model" / "patch_experts" / "cen_patches_0.25_of.dat").exists()
    model_secenegi = [] if cen_var else ["-mloc", "model/main_clnf_general.txt"]
    print("  Yüz noktası modeli:", "CE-CLM" if cen_var else "CLNF (paketle gelen)")
    for klasor in sorted(p for p in FOTOGRAFLAR.iterdir() if p.is_dir()):
        hedef = OPENFACE_CIKTI / klasor.name
        hedef.mkdir(parents=True, exist_ok=True)
        print(f"  {klasor.name} ...")
        # OpenFace modellerini exe'nin yanındaki klasörde aradığı için onun klasöründen çalıştırıyoruz
        subprocess.run([str(exe), "-fdir", str(klasor), "-out_dir", str(hedef), "-aus"] + model_secenegi,
                       cwd=str(exe.parent), stdout=subprocess.DEVNULL)
        print(f"    {len(list(hedef.glob('*.csv')))} fotoğraf işlendi")

    adim("4/4  Action Unit analizi (au_analiz_foto.py)", [PY, str(KLASOR / "au_analiz_foto.py")])
    print("\nBitti. Sonuçlar: kod\\ciktilar\\")


if __name__ == "__main__":
    main()
