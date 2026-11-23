"""
OpenFace analizi için etiketli ifade videosu kaydeder.
------------------------------------------------------
Ekranda sırayla hangi ifadeyi yapman gerektiği yazar (her biri ~4 sn).
Çıktılar:
  ifadeler.avi          -> OpenFace'e verilecek video
  ifadeler_bolumler.csv -> her ifadenin hangi kare aralığında yapıldığı

Çalıştır (env_fer): python video_kaydet.py
"""

import csv
import time

import cv2
import os
from pathlib import Path

# Script hangi klasörden çalıştırılırsa çalıştırılsın:
# modeller kod/modeller'den okunur, çıktılar (CSV, PNG, video) kod/ciktilar'a yazılır.
KLASOR = Path(__file__).resolve().parent
CIKTI = KLASOR / "ciktilar"
CIKTI.mkdir(exist_ok=True)
os.chdir(CIKTI)

ASAMALAR = [("Notr", 4), ("Mutluluk", 4), ("Saskinlik", 4), ("Uzuntu", 4),
            ("Ofke", 4), ("Korku", 4), ("Igrenme", 4), ("Notr", 3)]
VIDEO = "ifadeler.avi"
BOLUM_CSV = "ifadeler_bolumler.csv"
FONT = cv2.FONT_HERSHEY_SIMPLEX

kamera = cv2.VideoCapture(0)
genislik = int(kamera.get(cv2.CAP_PROP_FRAME_WIDTH))
yukseklik = int(kamera.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 1) Geri sayım sırasında kameranın gerçek FPS'ini ölç (videonun hızı doğru olsun)
kare_sayisi, baslangic = 0, time.time()
while time.time() - baslangic < 3:
    ok, kare = kamera.read()
    if not ok:
        raise SystemExit("Kameradan goruntu alinamadi")
    kare_sayisi += 1
    kalan = 3 - int(time.time() - baslangic)
    goster = kare.copy()
    cv2.putText(goster, f"Hazirlan... {kalan}", (20, 50), FONT, 1.2, (0, 255, 255), 3)
    cv2.imshow("Kayit", goster)
    cv2.waitKey(1)
fps = max(5.0, kare_sayisi / 3.0)
print(f"Olculen FPS: {fps:.1f}")

yazici = cv2.VideoWriter(VIDEO, cv2.VideoWriter_fourcc(*"XVID"), fps, (genislik, yukseklik))

# 2) Kayıt: her aşamada ekrana yapılacak ifadeyi yaz, kare numaralarını tut
bolumler = []
kare_no = 0
for ifade, sure in ASAMALAR:
    ilk_kare = kare_no
    bitis = time.time() + sure
    while time.time() < bitis:
        ok, kare = kamera.read()
        if not ok:
            break
        yazici.write(kare)            # videoya yazı OLMADAN kaydet
        kare_no += 1
        goster = kare.copy()
        cv2.putText(goster, f"Simdi: {ifade}  ({bitis - time.time():.0f} sn)",
                    (20, 50), FONT, 1.0, (0, 255, 0), 2)
        cv2.circle(goster, (genislik - 30, 30), 10, (0, 0, 255), -1)  # kayıt işareti
        cv2.imshow("Kayit", goster)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    # OpenFace kareleri 1'den numaralar
    bolumler.append((ifade, ilk_kare + 1, kare_no))

yazici.release()
kamera.release()
cv2.destroyAllWindows()

with open(BOLUM_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["ifade", "ilk_kare", "son_kare"])
    w.writerows(bolumler)

print(f"Kaydedildi: {VIDEO} ({kare_no} kare) ve {BOLUM_CSV}")
