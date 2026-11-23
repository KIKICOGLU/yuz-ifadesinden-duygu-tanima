import cv2
import os
from pathlib import Path

# Script hangi klasörden çalıştırılırsa çalıştırılsın:
# modeller kod/modeller'den okunur, çıktılar (CSV, PNG, video) kod/ciktilar'a yazılır.
KLASOR = Path(__file__).resolve().parent
CIKTI = KLASOR / "ciktilar"
CIKTI.mkdir(exist_ok=True)
os.chdir(CIKTI)

# 1) Kamerayı aç. 0 = bilgisayardaki ilk kamera
kamera = cv2.VideoCapture(0)

# 2) OpenCV ile birlikte gelen hazır, eğitilmiş yüz dedektörünü yükle
dosya = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
yuz_dedektoru = cv2.CascadeClassifier(dosya)

while True:
    # 3) Kameradan bir kare oku. basarili: okuma oldu mu, kare: görüntü matrisi
    basarili, kare = kamera.read()
    if not basarili:
        print("Kameradan görüntü alınamadı")
        break

    # 4) Haar renkle değil parlaklıkla çalışır, gri tonlamaya çevir
    gri = cv2.cvtColor(kare, cv2.COLOR_BGR2GRAY)

    # 5) Yüzleri bul. Her yüz için (x, y, genişlik, yükseklik) döner
    yuzler = yuz_dedektoru.detectMultiScale(
        gri,
        scaleFactor=1.1,   # görüntü her adımda %10 küçültülerek taranır
        minNeighbors=5,    # bir bölge en az kaç kez "yüz" denirse kabul edilir
        minSize=(60, 60),  # bundan küçük yüzleri dikkate alma
    )

    # 6) Her yüzün etrafına yeşil dikdörtgen çiz (renk BGR: (0, 255, 0) = yeşil)
    for (x, y, w, h) in yuzler:
        cv2.rectangle(kare, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 7) Sonucu pencerede göster
    cv2.imshow("Yuz Tespiti - cikmak icin q", kare)

    # 8) 1 ms bekle; q tuşuna basılırsa döngüden çık
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 9) Kamerayı serbest bırak ve pencereleri kapat
kamera.release()
cv2.destroyAllWindows()