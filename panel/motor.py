"""
Panel motoru: yüz bulma, duygu tahmini, çizim ve özet
=====================================================
Panelin üç analiz scripti bu dosyayı ortak kullanır:
  foto_degerlendir.py   (fotoğraflar)
  video_degerlendir.py  (kayıtlı video dosyaları)
  canli_degerlendir.py  (anlık kamera)

Ödev kodlarına (kod/) hiçbir şey yazmaz; oradan yalnızca YuNet model dosyasını OKUR.
Bütün çıktılar panel/ciktilar/<tür>_<tarih>/ klasörlerine gider.

Kullanılan yöntem ödevdeki ana demoyla (kod/duygu_demo.py) aynıdır:
  YuNet ile yüz -> %15 payla kırp -> BGR'den RGB'ye -> HSEmotion enet_b0_8_va_mtl
  (AffectNet ile eğitilmiş; 8 duygu olasılığı + valence + arousal)
Fotoğraflarda istenirse FER (FER-2013) de aynı yüz kutusuyla çalıştırılır.

Ortam: env_fer
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")   # TensorFlow'un uzun log mesajlarını gizle

import urllib.request  # noqa: F401  hsemotion-onnx 0.3.1 hatası: modeli indirirken bu import'u unutuyor
import csv
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# Panel çıktıları Türkçe karakter içerir; Windows konsolunda da düzgün görünsün
for akis in (sys.stdout, sys.stderr):
    try:
        akis.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

PANEL = Path(__file__).resolve().parent
PROJE = PANEL.parent
KOD = PROJE / "kod"
CIKTILAR = PANEL / "ciktilar"
YUZ_MODELI = KOD / "modeller" / "face_detection_yunet_2023mar.onnx"
DUYGU_MODELI = "enet_b0_8_va_mtl"

# HSEmotion'ın çıktı sırası: Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise
DUYGULAR_TR = ["Öfke", "Küçümseme", "İğrenme", "Korku", "Mutluluk", "Nötr", "Üzüntü", "Şaşkınlık"]
# OpenCV görüntünün üstüne Türkçe karakter yazamadığı için çizimlerde ASCII adlar kullanılır
DUYGULAR_ASCII = ["Ofke", "Kucumseme", "Igrenme", "Korku", "Mutluluk", "Notr", "Uzuntu", "Saskinlik"]
FER_TR = {"angry": "Öfke", "disgust": "İğrenme", "fear": "Korku", "happy": "Mutluluk",
          "sad": "Üzüntü", "surprise": "Şaşkınlık", "neutral": "Nötr"}
FER_ASCII = {"angry": "Ofke", "disgust": "Igrenme", "fear": "Korku", "happy": "Mutluluk",
             "sad": "Uzuntu", "surprise": "Saskinlik", "neutral": "Notr"}

# Zaman kayıtları ödevdeki oturum_kaydi.csv ile aynı biçimde tutulur
KAYIT_BASLIK = ["zaman_sn"] + DUYGULAR_ASCII + ["valence", "arousal"]
OLUMSUZ_ESIK = -0.2     # valence bu değerin altındaysa an "olumsuz" sayılır
YUMUSATMA = 0.6         # ekrandaki değerler için üstel hareketli ortalama
FONT = cv2.FONT_HERSHEY_SIMPLEX
YESIL, TURUNCU, KIRMIZI = (0, 200, 0), (0, 165, 255), (0, 0, 255)
BEYAZ, KOYU = (255, 255, 255), (30, 30, 30)

_TR_ASCII = str.maketrans("çğıöşüÇĞİÖŞÜâîûÂÎÛ", "cgiosuCGIOSUaiuAIU")


# ---------------------------------------------------------------- panel ile haberleşme
# Panel bu satırları okuyup ilerleme çubuğunu ve sonuç ekranını günceller.
# Script VS Code'dan çalıştırıldığında da zararsızdır, sadece ekrana yazılır.
def ilerleme(yapilan, toplam):
    print(f"ILERLEME {yapilan}/{toplam}", flush=True)


def cikti_bildir(klasor):
    print(f"CIKTI {klasor}", flush=True)


def durdurulsun_mu(dur_dosyasi):
    """Panel 'Durdur' düğmesine basılınca bu dosyayı oluşturur; script işini düzgünce bitirir."""
    return bool(dur_dosyasi) and Path(dur_dosyasi).exists()


# ---------------------------------------------------------------- dosya işlemleri
def guvenli_ad(metin, uzunluk=40):
    """Dosya adlarını ASCII'ye çevirir (OpenCV, Windows'ta Türkçe karakterli yollara yazamaz)."""
    metin = str(metin).translate(_TR_ASCII)
    ad = "".join(c if (c.isascii() and (c.isalnum() or c in "-_")) else "_" for c in metin)
    return ad[:uzunluk].strip("_") or "dosya"


def cikti_klasoru(tur, ek=""):
    zaman = datetime.now().strftime("%Y%m%d_%H%M%S")
    ad = f"{tur}_{zaman}" + (f"_{guvenli_ad(ek, 30)}" if ek else "")
    klasor = CIKTILAR / ad
    klasor.mkdir(parents=True, exist_ok=True)
    return klasor


def resim_oku(yol):
    """cv2.imread Windows'ta 'Masaüstü' gibi Türkçe karakterli yolları okuyamaz; bu yüzden
    dosya önce byte olarak okunup sonra çözülür."""
    try:
        veri = np.fromfile(str(yol), dtype=np.uint8)
    except OSError:
        return None
    if veri.size == 0:
        return None
    return cv2.imdecode(veri, cv2.IMREAD_COLOR)


def resim_yaz(yol, goruntu, kalite=92):
    yol = Path(yol)
    parametre = [cv2.IMWRITE_JPEG_QUALITY, kalite] if yol.suffix.lower() in (".jpg", ".jpeg") else []
    basarili, tampon = cv2.imencode(yol.suffix or ".jpg", goruntu, parametre)
    if basarili:
        tampon.tofile(str(yol))
    return basarili


def kucult(goruntu, en_buyuk_kenar):
    """Çok büyük fotoğrafları (telefon fotoğrafı gibi) işlemeden önce küçültür."""
    H, W = goruntu.shape[:2]
    oran = en_buyuk_kenar / max(H, W)
    if oran >= 1:
        return goruntu
    return cv2.resize(goruntu, (int(W * oran), int(H * oran)), interpolation=cv2.INTER_AREA)


def yuzu_kirp(kare, x, y, w, h, pay=0.15):
    """Yüz kutusunu her yönden %15 genişletip kırpar (kod/duygu_demo.py ile aynı)."""
    H, W = kare.shape[:2]
    dx, dy = int(w * pay), int(h * pay)
    x1, y1 = max(0, x - dx), max(0, y - dy)
    x2, y2 = min(W, x + w + dx), min(H, y + h + dy)
    return kare[y1:y2, x1:x2]


# ---------------------------------------------------------------- modeller
class DuyguMotoru:
    """YuNet + HSEmotion (ve istenirse FER) modellerini bir kez yükler, her karede kullanır."""

    def __init__(self, esik=0.6, fer_kullan=False):
        if not YUZ_MODELI.exists():
            raise SystemExit(f"YuNet modeli bulunamadı: {YUZ_MODELI}\n"
                             "Panel klasörü, kod klasörüyle aynı proje klasöründe olmalı.")
        print("Modeller yükleniyor...", flush=True)
        try:
            from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
        except ImportError:
            raise SystemExit(
                "hsemotion_onnx bulunamadı: bu script env_fer ortamında çalıştırılmalı.\n"
                f"  Şu an kullanılan Python: {sys.executable}\n"
                f"  Doğrusu: {PROJE / 'env_fer' / 'Scripts' / 'python.exe'}\n"
                "  VS Code'da sağ alttan yorumlayıcı olarak env_fer'i seç ya da terminalde önce\n"
                "  .\\env_fer\\Scripts\\Activate.ps1 komutunu çalıştır.") from None
        self.dedektor = cv2.FaceDetectorYN.create(str(YUZ_MODELI), "", (320, 320), score_threshold=esik)
        self.hse = HSEmotionRecognizer(model_name=DUYGU_MODELI)
        self.fer = None
        if fer_kullan:
            print("FER (TensorFlow) yükleniyor, biraz sürebilir...", flush=True)
            try:
                from fer.fer import FER
            except ImportError as hata:
                raise SystemExit(f"FER yüklenemedi ({hata}). FER seçeneği env_fer ortamında çalışır.") from None
            self.fer = FER()
        self._boyut = None

    def yuzleri_bul(self, kare):
        """Yüzleri büyükten küçüğe sıralı döndürür. Her yüz: x, y, w, h, 5 nokta (10 sayı), skor."""
        H, W = kare.shape[:2]
        if self._boyut != (W, H):
            self.dedektor.setInputSize((W, H))     # YuNet'e görüntünün kendi boyutunu söyle
            self._boyut = (W, H)
        _, yuzler = self.dedektor.detect(kare)
        if yuzler is None:
            return []
        return sorted(yuzler, key=lambda y: -float(y[2] * y[3]))

    def tahmin(self, kare, yuz):
        """10 değer döndürür: 8 duygu olasılığı + valence + arousal (yüz kırpılamazsa None)."""
        x, y, w, h = [int(v) for v in yuz[:4]]
        kirpik = yuzu_kirp(kare, x, y, w, h)
        if kirpik.size == 0:
            return None
        _, skorlar = self.hse.predict_emotions(cv2.cvtColor(kirpik, cv2.COLOR_BGR2RGB), logits=False)
        skorlar = np.asarray(skorlar, dtype=float)
        # Model valence ve arousal'ı bazen 1'in biraz dışında verir; Russell modelinin
        # [-1, 1] aralığına kırpılır (kod/oturum_ozeti.py de aynısını yapar)
        skorlar[8:10] = np.clip(skorlar[8:10], -1, 1)
        return skorlar

    def fer_tahmin(self, kare, yuz):
        """FER'e aynı yüz kutusunu verir; {'happy': 0.8, ...} döndürür (FER yoksa None)."""
        if self.fer is None:
            return None
        H, W = kare.shape[:2]
        x, y, w, h = [int(v) for v in yuz[:4]]
        x, y = max(0, x), max(0, y)
        w, h = min(w, W - x), min(h, H - y)
        if w <= 0 or h <= 0:
            return None
        sonuc = self.fer.detect_emotions(kare, face_rectangles=[(x, y, w, h)])
        if not sonuc:
            return None
        return {k: float(v) for k, v in sonuc[0]["emotions"].items()}


def en_olasi(skorlar):
    i = int(np.argmax(skorlar[:8]))
    return i, float(skorlar[i])


# ---------------------------------------------------------------- çizim
def olcek_hesapla(goruntu):
    """Yazı boyutunu görüntünün büyüklüğüne göre ayarlar."""
    return float(np.clip(min(goruntu.shape[:2]) / 700, 0.5, 1.4))


def etiket_yaz(goruntu, metin, x, y, renk, olcek=0.6):
    """Okunabilsin diye yazının arkasına koyu bir kutu çizer. (x, y) yazının sol alt köşesi."""
    kalinlik = max(1, int(round(olcek * 2)))
    (gen, yuk), _ = cv2.getTextSize(metin, FONT, olcek, kalinlik)
    H, W = goruntu.shape[:2]
    x = int(np.clip(x, 0, max(0, W - gen - 8)))
    y = int(np.clip(y, yuk + 8, H - 4))
    cv2.rectangle(goruntu, (x, y - yuk - 8), (x + gen + 8, y + 4), KOYU, -1)
    cv2.putText(goruntu, metin, (x + 4, y - 2), FONT, olcek, renk, kalinlik, cv2.LINE_AA)
    return yuk + 14


def yuz_ciz(goruntu, yuz, skorlar, no=None, fer=None, olcek=None):
    """Yüz kutusu, 5 yüz noktası ve tahmin etiketlerini çizer."""
    olcek = olcek or olcek_hesapla(goruntu)
    x, y, w, h = [int(v) for v in yuz[:4]]
    cv2.rectangle(goruntu, (x, y), (x + w, y + h), YESIL, max(2, int(olcek * 3)))
    for i in range(4, 14, 2):
        cv2.circle(goruntu, (int(yuz[i]), int(yuz[i + 1])), max(2, int(olcek * 4)), KIRMIZI, -1)
    if skorlar is None:
        return
    i, p = en_olasi(skorlar)
    satirlar = [((f"{no}: " if no else "") + f"{DUYGULAR_ASCII[i]} {p * 100:.0f}%", YESIL),
                (f"V {skorlar[8]:+.2f}  A {skorlar[9]:+.2f}", BEYAZ)]
    if fer:
        k = max(fer, key=fer.get)
        satirlar.append((f"FER: {FER_ASCII.get(k, k)} {fer[k] * 100:.0f}%", TURUNCU))
    # Etiketler kutunun üstüne; yer yoksa kutunun altına
    satir_yuk = int(32 * max(olcek, 0.7))
    yukari = y - 6 - satir_yuk * (len(satirlar) - 1)
    if yukari > satir_yuk:                                         # kutunun üstünde yer var
        baslangic = yukari
    elif y + h + satir_yuk * len(satirlar) < goruntu.shape[0]:     # kutunun altında yer var
        baslangic = y + h + satir_yuk
    else:                                                          # yüz bütün kareyi kaplıyor: kutunun içine
        baslangic = max(y, 0) + satir_yuk
    yazi = max(0.55, olcek * 0.8)
    for k, (metin, renk) in enumerate(satirlar):
        etiket_yaz(goruntu, metin, x, baslangic + k * satir_yuk, renk, yazi)


def cubuklari_ciz(kare, olasiliklar):
    """Sol üst köşeye 8 duygunun olasılık çubuklarını çizer (kod/duygu_demo.py ile aynı görünüm)."""
    cv2.rectangle(kare, (0, 0), (310, 20 + 22 * len(DUYGULAR_ASCII)), KOYU, -1)
    en_iyi = int(np.argmax(olasiliklar))
    for i, (ad, p) in enumerate(zip(DUYGULAR_ASCII, olasiliklar)):
        y = 25 + i * 22
        renk = (0, 255, 0) if i == en_iyi else (0, 200, 255)
        cv2.putText(kare, ad, (10, y), FONT, 0.5, BEYAZ, 1, cv2.LINE_AA)
        cv2.rectangle(kare, (110, y - 12), (110 + int(p * 150), y), renk, -1)
        cv2.putText(kare, f"{p * 100:.0f}%", (268, y), FONT, 0.45, BEYAZ, 1, cv2.LINE_AA)


def daire_ciz(kare, valence, arousal):
    """Sağ alt köşeye Russell'ın valence-arousal dairesini çizer."""
    H, W = kare.shape[:2]
    r = 70
    cx, cy = W - r - 25, H - r - 45
    cv2.rectangle(kare, (cx - r - 20, cy - r - 40), (cx + r + 20, cy + r + 40), KOYU, -1)
    cv2.putText(kare, "Valence - Arousal", (cx - 68, cy - r - 22), FONT, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(kare, (cx, cy), r, (200, 200, 200), 1)
    cv2.line(kare, (cx - r, cy), (cx + r, cy), (120, 120, 120), 1)
    cv2.line(kare, (cx, cy - r), (cx, cy + r), (120, 120, 120), 1)
    cv2.putText(kare, "Heyecanli", (cx - 32, cy - r - 4), FONT, 0.4, BEYAZ, 1, cv2.LINE_AA)
    cv2.putText(kare, "Sakin", (cx - 18, cy + r + 14), FONT, 0.4, BEYAZ, 1, cv2.LINE_AA)
    cv2.putText(kare, "-", (cx - r - 14, cy + 5), FONT, 0.6, (0, 0, 255), 2)
    cv2.putText(kare, "+", (cx + r + 4, cy + 5), FONT, 0.6, (0, 255, 0), 2)
    v = float(np.clip(valence, -1, 1))
    a = float(np.clip(arousal, -1, 1))
    nokta = (int(cx + v * r), int(cy - a * r))       # ekranda y aşağı doğru büyür
    cv2.circle(kare, nokta, 7, (0, 255, 0) if v >= 0 else (0, 0, 255), -1)
    cv2.putText(kare, f"V:{v:+.2f}  A:{a:+.2f}", (cx - 58, cy + r + 32), FONT, 0.45, BEYAZ, 1, cv2.LINE_AA)


def kare_ciz(kare, yuz, skorlar):
    """Video ve canlı mod için: en büyük yüzün kutusu + çubuklar + daire."""
    i, p = en_olasi(skorlar)
    x, y, w, h = [int(v) for v in yuz[:4]]
    cv2.rectangle(kare, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(kare, f"{DUYGULAR_ASCII[i]} {p * 100:.0f}%", (x, max(20, y - 10)), FONT, 0.7,
                (0, 255, 0), 2, cv2.LINE_AA)
    for k in range(4, 14, 2):
        cv2.circle(kare, (int(yuz[k]), int(yuz[k + 1])), 3, KIRMIZI, -1)
    cubuklari_ciz(kare, skorlar[:8])
    if kare.shape[0] >= 260 and kare.shape[1] >= 420:
        daire_ciz(kare, skorlar[8], skorlar[9])


# ---------------------------------------------------------------- zaman kaydı ve özet
def kayit_yaz(klasor, kayitlar):
    """[zaman, 8 olasılık, valence, arousal] satırlarını kayit.csv'ye yazar."""
    yol = Path(klasor) / "kayit.csv"
    with open(yol, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(KAYIT_BASLIK)
        w.writerows(kayitlar)
    return yol


def _hareketli_ortalama(x, n=5):
    if len(x) < n:
        return x
    dolgulu = np.pad(x, (n // 2, n - 1 - n // 2), mode="edge")
    return np.convolve(dolgulu, np.ones(n) / n, mode="valid")


def ozet_cikar(klasor, kayitlar, baslik, ek_satirlar=()):
    """Zaman kaydından ozet.png ve ozet.txt üretir (kod/oturum_ozeti.py ile aynı ölçütler).
    Özet metnini döndürür; kayıt çok kısaysa None."""
    import matplotlib
    matplotlib.use("Agg")                          # pencere açmadan dosyaya çiz
    import matplotlib.pyplot as plt

    if len(kayitlar) < 2:
        print("Özet çıkarmak için yeterli kayıt yok (en az 2 karede yüz bulunmalı).", flush=True)
        return None
    veri = np.asarray(kayitlar, dtype=float)
    zaman = veri[:, 0]
    olasiliklar = veri[:, 1:9]
    valence = np.clip(veri[:, 9], -1, 1)
    arousal = np.clip(veri[:, 10], -1, 1)

    v_duz, a_duz = _hareketli_ortalama(valence), _hareketli_ortalama(arousal)
    baskin = olasiliklar.argmax(axis=1)
    paylar = np.bincount(baskin, minlength=8) / len(baskin)
    sureler = np.diff(zaman, append=zaman[-1])     # kareler arası gerçek süre
    toplam_sure = zaman[-1] - zaman[0]
    olumsuz_pay = sureler[v_duz < OLUMSUZ_ESIK].sum() / toplam_sure if toplam_sure > 0 else 0.0
    en_kotu = int(np.argmin(v_duz))

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
    ax1.set_title(baslik)
    ax1.legend(loc="upper right", fontsize=8)

    ax2 = fig.add_subplot(izgara[1, 0])
    sira = np.argsort(paylar)
    ax2.barh([DUYGULAR_TR[i] for i in sira], paylar[sira] * 100, color="tab:blue")
    ax2.set_xlabel("Sürenin yüzdesi (%)")
    ax2.set_title("Baskın duygu payları")

    ax3 = fig.add_subplot(izgara[1, 1])
    ax3.add_patch(plt.Circle((0, 0), 1, fill=False, color="gray"))
    ax3.axhline(0, color="gray", linewidth=0.8)
    ax3.axvline(0, color="gray", linewidth=0.8)
    nokta = ax3.scatter(valence, arousal, c=zaman, cmap="viridis", s=8)
    fig.colorbar(nokta, ax=ax3, label="Zaman (sn)")
    ax3.set_xlim(-1.1, 1.1)
    ax3.set_ylim(-1.1, 1.1)
    ax3.set_aspect("equal")
    ax3.set_xlabel("Valence")
    ax3.set_ylabel("Arousal")
    ax3.set_title("Valence-Arousal düzlemi (Russell)")

    plt.tight_layout()
    plt.savefig(Path(klasor) / "ozet.png", dpi=110)
    plt.close(fig)

    baskinlar = ", ".join(f"{DUYGULAR_TR[i]} %{paylar[i] * 100:.0f}"
                          for i in np.argsort(-paylar) if paylar[i] > 0)
    metin = "\n".join([
        f"Süre                  : {toplam_sure:.1f} sn ({len(veri)} ölçüm)",
        f"Ortalama valence      : {valence.mean():+.2f}",
        f"Ortalama arousal      : {arousal.mean():+.2f}",
        f"Olumsuz anların payı  : %{olumsuz_pay * 100:.0f}  (valence < {OLUMSUZ_ESIK})",
        f"En olumsuz an         : {zaman[en_kotu]:.1f}. sn (valence {v_duz[en_kotu]:+.2f})",
        f"Baskın duygular       : {baskinlar}",
        *ek_satirlar,
    ])
    (Path(klasor) / "ozet.txt").write_text(metin + "\n", encoding="utf-8")
    return metin
