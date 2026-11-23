"""
Görüşme (oturum) duygu özeti  -  uygulama senaryosu
===================================================
duygu_demo.py'nin kaydettiği "oturum_kaydi.csv" dosyasını okuyup bir müşteri
hizmetleri / terapi görüşmesinin duygu özetini çıkarır:

  oturum_ozeti.png  ->  (1) valence ve arousal'ın zamana göre değişimi,
                         olumsuz anlar kırmızı ile işaretli
                        (2) baskın duyguların süre payları
                        (3) bütün anların valence-arousal düzlemindeki dağılımı
  oturum_ozeti.txt  ->  rapora yapıştırılabilecek sayısal özet

Ortam: env_fer (matplotlib gerekir)    Çalıştır: python oturum_ozeti.py
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

OTURUM_CSV = "oturum_kaydi.csv"
DUYGULAR = ["Öfke", "Küçümseme", "İğrenme", "Korku", "Mutluluk", "Nötr", "Üzüntü", "Şaşkınlık"]
OLUMSUZ_ESIK = -0.2     # valence bu değerin altındaysa an "olumsuz" sayılır
PENCERE = 5             # grafikte gürültüyü azaltmak için hareketli ortalama (kare sayısı)

with open(OTURUM_CSV, encoding="utf-8") as f:
    okuyucu = csv.reader(f)
    next(okuyucu)                                   # başlık satırını atla
    veri = np.array([[float(d) for d in satir] for satir in okuyucu])

if len(veri) < 2:
    raise SystemExit("Oturum kaydı çok kısa. duygu_demo.py'yi biraz daha uzun çalıştır.")

zaman = veri[:, 0]
olasiliklar = veri[:, 1:9]
valence = np.clip(veri[:, 9], -1, 1)
arousal = np.clip(veri[:, 10], -1, 1)


def hareketli_ortalama(x, n=PENCERE):
    if len(x) < n:
        return x
    # Kenarları uç değerlerle doldur; yoksa başta ve sonda değerler yapay olarak sıfıra çekilir
    dolgulu = np.pad(x, (n // 2, n - 1 - n // 2), mode="edge")
    return np.convolve(dolgulu, np.ones(n) / n, mode="valid")


v_duz = hareketli_ortalama(valence)
a_duz = hareketli_ortalama(arousal)
baskin = olasiliklar.argmax(axis=1)                 # her karede en olası duygu
paylar = np.bincount(baskin, minlength=8) / len(baskin)

# Kareler arası süreler (kameranın FPS'i sabit olmayabilir, gerçek zamanı kullanıyoruz)
sureler = np.diff(zaman, append=zaman[-1])
toplam_sure = zaman[-1] - zaman[0]
olumsuz_sure = sureler[v_duz < OLUMSUZ_ESIK].sum()

# ---------- Grafik ----------
fig = plt.figure(figsize=(14, 8))
izgara = fig.add_gridspec(2, 2, height_ratios=[1, 1])

ax1 = fig.add_subplot(izgara[0, :])
ax1.plot(zaman, v_duz, label="Valence (olumsuz ↔ olumlu)", color="tab:green")
ax1.plot(zaman, a_duz, label="Arousal (sakin ↔ heyecanlı)", color="tab:orange")
ax1.axhline(0, color="gray", linewidth=0.8)
ax1.fill_between(zaman, -1, 1, where=v_duz < OLUMSUZ_ESIK, color="red", alpha=0.12,
                 label=f"Olumsuz an (valence < {OLUMSUZ_ESIK})")
ax1.set_ylim(-1, 1)
ax1.set_xlabel("Zaman (sn)")
ax1.set_title("Görüşme boyunca duygu durumu")
ax1.legend(loc="upper right", fontsize=8)

ax2 = fig.add_subplot(izgara[1, 0])
sira = np.argsort(paylar)[::-1]
ax2.barh([DUYGULAR[i] for i in sira][::-1], (paylar[sira] * 100)[::-1], color="tab:blue")
ax2.set_xlabel("Sürenin yüzdesi (%)")
ax2.set_title("Baskın duygu payları")

ax3 = fig.add_subplot(izgara[1, 1])
nokta = ax3.scatter(valence, arousal, c=zaman, cmap="viridis", s=8)
ax3.add_patch(plt.Circle((0, 0), 1, fill=False, color="gray"))
ax3.axhline(0, color="gray", linewidth=0.8)
ax3.axvline(0, color="gray", linewidth=0.8)
ax3.set_xlim(-1.1, 1.1)
ax3.set_ylim(-1.1, 1.1)
ax3.set_aspect("equal")
ax3.set_xlabel("Valence")
ax3.set_ylabel("Arousal")
ax3.set_title("Valence-Arousal düzlemi (Russell)")
fig.colorbar(nokta, ax=ax3, label="Zaman (sn)")

plt.tight_layout()
plt.savefig("oturum_ozeti.png", dpi=150)

# ---------- Metin özeti ----------
en_olumsuz = int(np.argmin(v_duz))
satirlar = [
    f"Oturum süresi           : {toplam_sure:.1f} sn ({len(veri)} kare)",
    f"Ortalama valence        : {valence.mean():+.2f}",
    f"Ortalama arousal        : {arousal.mean():+.2f}",
    f"Olumsuz anların payı    : %{100 * olumsuz_sure / max(toplam_sure, 1e-6):.0f}",
    f"En olumsuz an           : {zaman[en_olumsuz]:.1f}. sn (valence {v_duz[en_olumsuz]:+.2f})",
    "Baskın duygu payları    : " + ", ".join(f"{DUYGULAR[i]} %{paylar[i] * 100:.0f}" for i in sira if paylar[i] > 0),
]
metin = "\n".join(satirlar)
print(metin)
with open("oturum_ozeti.txt", "w", encoding="utf-8") as f:
    f.write(metin)
print("\nKaydedildi: oturum_ozeti.png ve oturum_ozeti.txt")
