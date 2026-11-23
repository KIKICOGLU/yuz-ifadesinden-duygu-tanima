"""
Video dosyası değerlendirme
===========================
Önceden çekilmiş bir videoyu kare kare işler: en büyük yüzün 8 duygu olasılığını,
valence ve arousal değerlerini zamana göre kaydeder ve bir görüşme özeti çıkarır
(ödevdeki oturum_ozeti.py ile aynı ölçütler).

Çalıştırma (env_fer ortamında):
  python video_degerlendir.py                       -> video seçme penceresi açılır
  python video_degerlendir.py gorusme.mp4
  python video_degerlendir.py gorusme.mp4 --adim 3  -> her 3 karede bir analiz (daha hızlı)
  python video_degerlendir.py gorusme.mp4 --video-yok  -> işaretli video kaydetme

Çıktılar: panel/ciktilar/video_<tarih>_<ad>/
  kayit.csv      zaman, 8 duygu olasılığı, valence, arousal (oturum_kaydi.csv ile aynı biçim)
  ozet.png/.txt  valence-arousal zaman grafiği, baskın duygular, sayısal özet
  isaretli.mp4   kutu, duygu çubukları ve valence-arousal dairesi çizilmiş video
"""

import argparse
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np

import motor

EN_GENIS = 1280     # daha geniş videolar bu genişliğe küçültülür (hız için)


def video_sec():
    import tkinter as tk
    from tkinter import filedialog
    kok = tk.Tk()
    kok.withdraw()
    secim = filedialog.askopenfilename(
        title="Değerlendirilecek videoyu seç",
        filetypes=[("Videolar", "*.mp4 *.avi *.mov *.mkv *.wmv *.webm *.m4v"), ("Tüm dosyalar", "*.*")])
    kok.destroy()
    return secim


def video_ac(yol):
    """OpenCV, Windows'ta Türkçe karakterli yollardaki videoları açamayabilir.
    Açılmazsa video geçici bir klasöre kopyalanıp oradan açılır."""
    kamera = cv2.VideoCapture(str(yol))
    if kamera.isOpened():
        return kamera, None
    if str(yol).isascii():
        return kamera, None
    gecici = Path(tempfile.mkdtemp(prefix="duygu_panel_")) / ("video" + Path(yol).suffix.lower())
    print("Video yolu Türkçe karakter içeriyor; geçici bir kopyadan açılıyor...")
    shutil.copyfile(yol, gecici)
    return cv2.VideoCapture(str(gecici)), gecici


def main():
    ayr = argparse.ArgumentParser(description="Kayıtlı bir videonun duygu özetini çıkarır.")
    ayr.add_argument("video", nargs="?", help="video dosyası (mp4, avi, mov, ...)")
    ayr.add_argument("--adim", type=int, default=2, help="her N karede bir analiz (varsayılan 2)")
    ayr.add_argument("--video-yok", action="store_true", help="işaretli video kaydetme")
    ayr.add_argument("--dur-dosyasi", help="(panel için) bu dosya oluşunca iş düzgünce durdurulur")
    arg = ayr.parse_args()

    yol = arg.video or video_sec()
    if not yol or not Path(yol).is_file():
        raise SystemExit(f"Video bulunamadı: {yol}")
    yol = Path(yol)
    adim = max(1, arg.adim)

    kamera, gecici = video_ac(yol)
    if not kamera.isOpened():
        raise SystemExit("Video açılamadı. Desteklenen biçimler: mp4, avi, mov, mkv, wmv, webm.")
    fps = kamera.get(cv2.CAP_PROP_FPS)
    if not fps or np.isnan(fps) or fps > 240:
        fps = 30.0
    toplam = int(kamera.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(kamera.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(kamera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    oran = min(1.0, EN_GENIS / W) if W else 1.0
    boyut = (int(W * oran) // 2 * 2, int(H * oran) // 2 * 2)   # kodlayıcılar çift sayı ister

    klasor = motor.cikti_klasoru("video", yol.stem)
    motor.cikti_bildir(klasor)
    sure_tahmini = f"{toplam / fps:.0f} sn, " if toplam > 0 else ""
    print(f"Video: {yol.name}  ({W}x{H}, {fps:.1f} kare/sn, {sure_tahmini}{max(toplam, 0)} kare)")
    print(f"Her {adim} karede bir analiz yapılacak. Çıktılar: {klasor}")
    model = motor.DuyguMotoru(esik=0.6)

    yazici = None
    if not arg.video_yok:
        yazici = cv2.VideoWriter(str(klasor / "isaretli.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, boyut)
        if not yazici.isOpened():
            print("Uyarı: işaretli video yazılamıyor; yalnızca özet çıkarılacak.")
            yazici = None

    kayitlar, duzgun, son = [], None, None
    analiz, yuzlu, k = 0, 0, 0
    adim_ilerleme = max(1, toplam // 100) if toplam > 0 else 30
    while True:
        basarili, kare = kamera.read()
        if not basarili:
            break
        if kare.shape[1] != boyut[0] or kare.shape[0] != boyut[1]:
            kare = cv2.resize(kare, boyut, interpolation=cv2.INTER_AREA)

        if k % adim == 0:
            analiz += 1
            yuzler = model.yuzleri_bul(kare)
            son = None
            if yuzler:
                yuz = yuzler[0]                         # en büyük yüz
                skorlar = model.tahmin(kare, yuz)
                if skorlar is not None:
                    yuzlu += 1
                    kayitlar.append([round(k / fps, 3)] + [round(float(s), 4) for s in skorlar])
                    duzgun = skorlar if duzgun is None else motor.YUMUSATMA * duzgun + (1 - motor.YUMUSATMA) * skorlar
                    son = (yuz, duzgun)
        if yazici is not None:
            if son is not None:
                motor.kare_ciz(kare, *son)
            yazici.write(kare)

        k += 1
        if k % adim_ilerleme == 0:
            motor.ilerleme(k, max(toplam, 0))
            if motor.durdurulsun_mu(arg.dur_dosyasi):
                print("Durduruldu; o ana kadar işlenen kısım kaydediliyor.")
                break

    kamera.release()
    if yazici is not None:
        yazici.release()
    if gecici is not None:
        shutil.rmtree(gecici.parent, ignore_errors=True)
    motor.ilerleme(k, k)

    if not kayitlar:
        raise SystemExit("Videoda hiç yüz bulunamadı; özet çıkarılamadı.")
    motor.kayit_yaz(klasor, kayitlar)
    ek = [f"Kaynak                : {yol.name}",
          f"İşlenen kare          : {k} (her {adim} karede bir analiz: {analiz} kare)",
          f"Yüz bulunan analiz    : {yuzlu} / {analiz} (%{yuzlu / max(analiz, 1) * 100:.0f})"]
    metin = motor.ozet_cikar(klasor, kayitlar, f"Video: {yol.name}", ek)
    if metin:
        print("\n" + metin)
    print(f"\nBitti. Çıktılar: {klasor}")


if __name__ == "__main__":
    main()
