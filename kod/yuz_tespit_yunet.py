import cv2
import os
from pathlib import Path

# Script hangi klasörden çalıştırılırsa çalıştırılsın:
# modeller kod/modeller'den okunur, çıktılar (CSV, PNG, video) kod/ciktilar'a yazılır.
KLASOR = Path(__file__).resolve().parent
CIKTI = KLASOR / "ciktilar"
CIKTI.mkdir(exist_ok=True)
os.chdir(CIKTI)

MODEL = str(KLASOR / "modeller" / "face_detection_yunet_2023mar.onnx")

kamera = cv2.VideoCapture(0)
genislik = int(kamera.get(cv2.CAP_PROP_FRAME_WIDTH))
yukseklik = int(kamera.get(cv2.CAP_PROP_FRAME_HEIGHT))

# YuNet'e girdi boyutunu baştan söylüyoruz; kameradan gelen kareyle aynı olmalı
dedektor = cv2.FaceDetectorYN.create(
    MODEL, "", (genislik, yukseklik),
    score_threshold=0.8,   # bundan düşük skorlu tespitleri at
)

while True:
    basarili, kare = kamera.read()
    if not basarili:
        break

    # Dikkat: Haar'ın aksine RENKLİ kareyi veriyoruz
    _, yuzler = dedektor.detect(kare)

    # Hiç yüz yoksa YuNet None döndürür, kontrol etmezsek döngü hata verir
    if yuzler is not None:
        for yuz in yuzler:
            # Her satır 15 sayı: [x, y, w, h, 5 nokta × (x, y), skor]
            x, y, w, h = yuz[:4].astype(int)
            skor = yuz[14]

            cv2.rectangle(kare, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(kare, f"{skor:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 5 yüz noktası: indeks 4-13 arası (x1, y1, x2, y2, ...)
            # i = 4, 6, 8, 10, 12 -> x;  i + 1 -> aynı noktanın y'si
            for i in range(4, 14, 2):
                nx = int(yuz[i])
                ny = int(yuz[i + 1])
                cv2.circle(kare, (nx, ny), 3, (0, 0, 255), -1)

    cv2.imshow("YuNet - cikmak icin q", kare)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

kamera.release()
cv2.destroyAllWindows()
