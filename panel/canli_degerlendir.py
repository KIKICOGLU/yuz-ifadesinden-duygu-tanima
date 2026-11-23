"""
Canlı kamera değerlendirme
==========================
Kameradan anlık görüntü alıp her karede duygu tahmini yapar (ödevdeki duygu_demo.py ile
aynı yöntem). Fark: kayıt ödevin oturum_kaydi.csv dosyasının üzerine YAZILMAZ; her oturum
panel/ciktilar/ altında kendi klasörüne kaydedilir ve özet otomatik çıkarılır.

Varsayılan olarak yalnızca sayılar kaydedilir, görüntü kaydedilmez.

Çalıştırma (env_fer ortamında):
  python canli_degerlendir.py                    -> 0 numaralı kamera
  python canli_degerlendir.py --kamera 1         -> başka bir kamera
  python canli_degerlendir.py --video-kaydet     -> işaretli görüntüyü videoya da kaydet

Tuşlar: q = bitir ve kaydet,  s = ekran görüntüsü.  Pencereyi kapatmak da kaydeder.

Çıktılar: panel/ciktilar/canli_<tarih>/
  kayit.csv, ozet.png, ozet.txt  (ve istenirse isaretli.mp4, ekran_*.png)
"""

import argparse
import time

import cv2

import motor

PENCERE = "Duygu Tanima (panel) - q: bitir ve kaydet, s: ekran goruntusu"


def main():
    ayr = argparse.ArgumentParser(description="Kameradan canlı duygu tanıma ve oturum özeti.")
    ayr.add_argument("--kamera", default="0", help="kamera numarası (varsayılan 0)")
    ayr.add_argument("--video-kaydet", action="store_true", help="işaretli görüntüyü videoya da kaydet")
    ayr.add_argument("--dur-dosyasi", help="(panel için) bu dosya oluşunca oturum düzgünce bitirilir")
    arg = ayr.parse_args()

    kaynak = int(arg.kamera) if str(arg.kamera).isdigit() else arg.kamera
    klasor = motor.cikti_klasoru("canli")
    motor.cikti_bildir(klasor)
    model = motor.DuyguMotoru(esik=0.8)            # canlı demoda olduğu gibi 0,8 eşiği

    print(f"Kamera açılıyor ({kaynak})...", flush=True)
    kamera = cv2.VideoCapture(kaynak)
    if not kamera.isOpened():
        raise SystemExit(f"Kamera açılamadı (numara {kaynak}). Başka bir program kamerayı kullanıyor "
                         "olabilir ya da farklı bir kamera numarası denenmeli.")

    yazici = None
    kayitlar, duzgun, ekran_no = [], None, 0
    baslangic = time.time()
    print("Kamera açık. Bitirmek için kamera penceresindeyken q tuşuna bas.", flush=True)

    while True:
        basarili, kare = kamera.read()
        if not basarili:
            print("Kameradan görüntü alınamadı.")
            break

        yuzler = model.yuzleri_bul(kare)
        for sira, yuz in enumerate(yuzler):
            skorlar = model.tahmin(kare, yuz)
            if skorlar is None:
                continue
            if sira == 0:                             # en büyük yüz: kayıt, çubuklar ve daire
                kayitlar.append([round(time.time() - baslangic, 3)] + [round(float(s), 4) for s in skorlar])
                duzgun = skorlar if duzgun is None else motor.YUMUSATMA * duzgun + (1 - motor.YUMUSATMA) * skorlar
                motor.kare_ciz(kare, yuz, duzgun)
            else:                                     # diğer yüzler: yalnızca kutu ve etiket
                motor.yuz_ciz(kare, yuz, skorlar, olcek=0.8)

        sure = time.time() - baslangic
        cv2.putText(kare, f"{sure:5.1f} sn", (kare.shape[1] - 110, 28), motor.FONT, 0.7, (255, 255, 255), 2)

        if arg.video_kaydet:
            if yazici is None:
                fps = kamera.get(cv2.CAP_PROP_FPS) or 20.0
                yazici = cv2.VideoWriter(str(klasor / "isaretli.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                         fps if 1 <= fps <= 60 else 20.0, (kare.shape[1], kare.shape[0]))
            yazici.write(kare)

        cv2.imshow(PENCERE, kare)
        tus = cv2.waitKey(1) & 0xFF
        if tus == ord("q"):
            break
        if tus == ord("s"):
            ekran_no += 1
            motor.resim_yaz(klasor / f"ekran_{ekran_no}.png", kare)
            print(f"Ekran görüntüsü kaydedildi: ekran_{ekran_no}.png", flush=True)
        if cv2.getWindowProperty(PENCERE, cv2.WND_PROP_VISIBLE) < 1:      # pencere kapatıldı
            break
        if motor.durdurulsun_mu(arg.dur_dosyasi):
            break

    kamera.release()
    if yazici is not None:
        yazici.release()
    cv2.destroyAllWindows()

    if not kayitlar:
        raise SystemExit("Oturumda hiç yüz bulunamadı; kayıt yapılmadı.")
    motor.kayit_yaz(klasor, kayitlar)
    print(f"{len(kayitlar)} ölçüm kaydedildi.")
    metin = motor.ozet_cikar(klasor, kayitlar, "Canlı oturum boyunca duygu durumu")
    if metin:
        print("\n" + metin)
    print(f"\nBitti. Çıktılar: {klasor}")


if __name__ == "__main__":
    main()
