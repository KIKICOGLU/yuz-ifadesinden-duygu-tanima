"""
Duygu Tanıma – Kontrol Paneli
=============================
Bütün analizleri ve ödev scriptlerini tek pencereden çalıştırmak için.

Sekmeler
  Fotoğraf          internetten indirilen ya da çekilmiş fotoğraflar   -> foto_degerlendir.py
  Video dosyası     önceden çekilmiş videolar                          -> video_degerlendir.py
  Canlı kamera      anlık kamera görüntüsü                             -> canli_degerlendir.py
  Ödev scriptleri   kod/ klasöründeki teslim edilen scriptler (değiştirilmeden çalıştırılır)
  Geçmiş            panel/ciktilar altındaki önceki analizler

Panel her işi ayrı bir Python süreci olarak başlatır. Yani her script panel olmadan,
VS Code'dan da aynen çalışır; panelin günlük alanında her işin komutu yazılır.

Çalıştırma (env_fer ortamında):  python panel.py
"""

import csv
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from datetime import datetime
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    from PIL import Image, ImageTk
except ImportError:            # Pillow matplotlib ile birlikte gelir; yine de yoksa resimsiz çalış
    Image = ImageTk = None

PANEL = Path(__file__).resolve().parent
PROJE = PANEL.parent
KOD = PROJE / "kod"
CIKTILAR = PANEL / "ciktilar"
YEDEKLER = PANEL / "yedekler"


def ortam_pythonu():
    """Projedeki env_fer sanal ortamının Python'unu bulur (Windows ve Linux/macOS yolları)."""
    for aday in (PROJE / "env_fer" / "Scripts" / "python.exe", PROJE / "env_fer" / "bin" / "python"):
        if aday.exists():
            return aday
    return None


ENV_PYTHON = ortam_pythonu()
# Scriptler her zaman env_fer ile çalıştırılır; panel başka bir Python ile açılmış olsa bile.
# (env_fer bulunamazsa panelin kendi Python'u kullanılır.)
CALISTIRICI = str(ENV_PYTHON) if ENV_PYTHON else sys.executable.replace("pythonw.exe", "python.exe")


def env_fer_icinde_mi():
    if ENV_PYTHON is None:
        return True
    try:
        return Path(sys.prefix).resolve() == (PROJE / "env_fer").resolve()
    except OSError:
        return False

DUYGULAR_TR = ["Öfke", "Küçümseme", "İğrenme", "Korku", "Mutluluk", "Nötr", "Üzüntü", "Şaşkınlık"]
DUYGULAR_ASCII = ["Ofke", "Kucumseme", "Igrenme", "Korku", "Mutluluk", "Notr", "Uzuntu", "Saskinlik"]
TUR_ADI = {"foto": "Fotoğraf", "video": "Video", "canli": "Canlı kamera"}

# (dosya, argümanlar, başlık, açıklama, üzerine yazdığı dosyalar -> çalıştırmadan önce yedeklenir)
ODEV_SCRIPTLERI = [
    ("foto_deneyleri.py", [], "Bütün fotoğraf deneyleri",
     "46 fotoğrafta yüz tespiti karşılaştırması, FER–HSEmotion değerlendirmesi ve OpenFace analizi "
     "(rapordaki 7.2, 7.3 ve 7.4). Aynı fotoğraflarla aynı sonuçları yeniden üretir.", []),
    ("duygu_demo.py", [], "Canlı duygu demosu (ana uygulama)",
     "Kamera penceresinde q ile çık. Kapanınca ödevin oturum kaydının üzerine yazar; "
     "panel çalıştırmadan önce eski kaydı yedekler.",
     ["oturum_kaydi.csv", "oturum_ozeti.png", "oturum_ozeti.txt"]),
    ("oturum_ozeti.py", [], "Oturum özeti",
     "oturum_kaydi.csv'den görüşme özeti çıkarır (rapordaki 7.6).", []),
    ("degerlendirme.py", ["foto_sonuclari.csv"], "Değerlendirme",
     "Fotoğraf sonuçlarından doğruluk, makro F1 ve karmaşıklık matrisi.", []),
    ("yuz_tespit.py", [], "Yüz tespiti: Haar Cascade",
     "Kamerada klasik yüz tespiti. q ile çık.", []),
    ("yuz_tespit_yunet.py", [], "Yüz tespiti: YuNet",
     "Kamerada CNN tabanlı yüz tespiti ve 5 yüz noktası. q ile çık.", []),
    ("karsilastirma_canli.py", [], "Canlı FER–HSEmotion karşılaştırması",
     "1–7 tuşlarıyla etiketli kayıt alır. sonuclar.csv'nin üzerine yazar; panel önce yedekler.",
     ["sonuclar.csv"]),
]

ILERLEME = re.compile(r"^ILERLEME (\d+)/(\d+)\s*$")
CIKTI = re.compile(r"^CIKTI (.+?)\s*$")


def klasoru_ac(yol):
    yol = Path(yol)
    if not yol.exists():
        messagebox.showinfo("Bulunamadı", f"Henüz oluşturulmamış:\n{yol}")
        return
    if os.name == "nt":
        os.startfile(str(yol))                            # noqa: S606  (Windows Gezgini / varsayılan program)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(yol)])
    else:
        subprocess.Popen(["xdg-open", str(yol)])


# ======================================================================== küçük bileşenler
class ResimAlani(tk.Canvas):
    """Bir resmi, alanın boyutuna sığacak şekilde gösterir; pencere büyüyünce yeniden çizer."""

    def __init__(self, ebeveyn, bos_yazi="", **kw):
        super().__init__(ebeveyn, background="#202124", highlightthickness=0, **kw)
        self.bos_yazi = bos_yazi
        self._resim = None
        self._tk = None
        self.bind("<Configure>", lambda _e: self._ciz())

    def goster(self, yol):
        self._resim = None
        if Image is not None and yol and Path(yol).exists():
            try:
                with Image.open(yol) as r:
                    self._resim = r.convert("RGB")
            except OSError:
                self._resim = None
        self._ciz()

    def temizle(self, yazi=None):
        self._resim = None
        if yazi is not None:
            self.bos_yazi = yazi
        self._ciz()

    def _ciz(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 10 or h < 10:
            return
        if self._resim is None:
            self.create_text(w // 2, h // 2, text=self.bos_yazi, fill="#9AA0A6",
                             font=("Segoe UI", 11), width=w - 40, justify="center")
            return
        kopya = self._resim.copy()
        kopya.thumbnail((max(1, w - 12), max(1, h - 12)), Image.LANCZOS)
        self._tk = ImageTk.PhotoImage(kopya)
        self.create_image(w // 2, h // 2, image=self._tk)


class OzetAlani(ttk.Frame):
    """Video ve canlı sekmelerinde: solda özet grafiği, sağda sayısal özet ve düğmeler."""

    def __init__(self, ebeveyn, bos_yazi, video_dugmesi=True):
        super().__init__(ebeveyn)
        self.klasor = None
        govde = ttk.PanedWindow(self, orient="horizontal")
        govde.pack(fill="both", expand=True)
        self.resim = ResimAlani(govde, bos_yazi)
        govde.add(self.resim, weight=3)
        sag = ttk.Frame(govde, padding=(10, 0, 0, 0))
        govde.add(sag, weight=1)
        ttk.Label(sag, text="Sayısal özet", style="Kalin.TLabel").pack(anchor="w")
        self.metin = tk.Text(sag, height=12, width=44, wrap="word", relief="flat",
                             font=("Consolas", 10), background="#F6F6F6")
        self.metin.pack(fill="both", expand=True, pady=(4, 8))
        self.metin.configure(state="disabled")
        dugmeler = ttk.Frame(sag)
        dugmeler.pack(fill="x")
        if video_dugmesi:
            self.video_btn = ttk.Button(dugmeler, text="İşaretli videoyu aç", state="disabled",
                                        command=lambda: klasoru_ac(self.klasor / "isaretli.mp4"))
            self.video_btn.pack(fill="x", pady=2)
        else:
            self.video_btn = None
        self.klasor_btn = ttk.Button(dugmeler, text="Çıktı klasörünü aç", state="disabled",
                                     command=lambda: klasoru_ac(self.klasor))
        self.klasor_btn.pack(fill="x", pady=2)

    def yukle(self, klasor):
        self.klasor = Path(klasor) if klasor else None
        if self.klasor is None or not self.klasor.exists():
            self.resim.temizle("Çıktı bulunamadı.")
            self._metin_yaz("")
            return
        ozet_png, ozet_txt = self.klasor / "ozet.png", self.klasor / "ozet.txt"
        if ozet_png.exists():
            self.resim.goster(ozet_png)
        else:
            self.resim.temizle("Özet çıkarılamadı (yeterli yüz bulunamamış olabilir). Günlüğe bak.")
        self._metin_yaz(ozet_txt.read_text(encoding="utf-8") if ozet_txt.exists() else "")
        self.klasor_btn.configure(state="normal")
        if self.video_btn is not None:
            self.video_btn.configure(state="normal" if (self.klasor / "isaretli.mp4").exists() else "disabled")

    def _metin_yaz(self, metin):
        self.metin.configure(state="normal")
        self.metin.delete("1.0", "end")
        self.metin.insert("1.0", metin)
        self.metin.configure(state="disabled")


# ======================================================================== ana pencere
class Panel(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Duygu Tanıma – Kontrol Paneli")
        self.geometry("1280x840")
        self.minsize(1020, 700)
        self.surec = None
        self.kuyruk = queue.Queue()
        self.dur_dosyasi = None
        self.son_cikti = None
        self.bitince = None
        self.calisma_dugmeleri = []
        self.foto_satirlari = {}
        self._stiller()

        ust = ttk.Frame(self, padding=(14, 10, 14, 4))
        ust.pack(fill="x")
        ttk.Label(ust, text="Duygu Tanıma – Kontrol Paneli", style="Baslik.TLabel").pack(side="left")
        ttk.Label(ust, text=f"Scriptleri çalıştıran Python: {CALISTIRICI}", style="Soluk.TLabel").pack(side="right")

        dikey = ttk.PanedWindow(self, orient="vertical")
        dikey.pack(fill="both", expand=True, padx=10)
        self.defter = ttk.Notebook(dikey)
        dikey.add(self.defter, weight=5)
        self._foto_sekmesi()
        self._video_sekmesi()
        self._canli_sekmesi()
        self._odev_sekmesi()
        self._gecmis_sekmesi()
        self.defter.bind("<<NotebookTabChanged>>", self._sekme_degisti)

        alt = ttk.Frame(dikey)
        dikey.add(alt, weight=1)
        durum = ttk.Frame(alt, padding=(0, 6, 0, 4))
        durum.pack(fill="x")
        self.ilerleme = ttk.Progressbar(durum, length=320, mode="determinate")
        self.ilerleme.pack(side="left")
        self.durum_yazi = tk.StringVar(value="Hazır.")
        ttk.Label(durum, textvariable=self.durum_yazi).pack(side="left", padx=10)
        self.dur_btn = ttk.Button(durum, text="Durdur", state="disabled", command=self._durdur)
        self.dur_btn.pack(side="right")
        ttk.Button(durum, text="Günlüğü temizle", command=self._gunluk_temizle).pack(side="right", padx=6)
        gunluk_cerceve = ttk.Frame(alt)
        gunluk_cerceve.pack(fill="both", expand=True, pady=(0, 10))
        self.gunluk = tk.Text(gunluk_cerceve, height=9, wrap="word", font=("Consolas", 9),
                              background="#1E1E1E", foreground="#D4D4D4", insertbackground="#D4D4D4",
                              relief="flat")
        kaydirma = ttk.Scrollbar(gunluk_cerceve, command=self.gunluk.yview)
        self.gunluk.configure(yscrollcommand=kaydirma.set, state="disabled")
        self.gunluk.tag_configure("komut", foreground="#4FC1FF")
        self.gunluk.tag_configure("hata", foreground="#F48771")
        self.gunluk.tag_configure("tamam", foreground="#89D185")
        kaydirma.pack(side="right", fill="y")
        self.gunluk.pack(side="left", fill="both", expand=True)

        self._yaz("Panel hazır. Her işin komutu burada görünür; aynı komutu VS Code terminalinde "
                  "de çalıştırabilirsin.\n", "tamam")
        if not KOD.exists():
            self._yaz(f"Uyarı: ödev klasörü bulunamadı: {KOD}\n", "hata")
        if ENV_PYTHON is None:
            self._yaz(f"Uyarı: {PROJE / 'env_fer'} bulunamadı; scriptler şu Python ile çalışacak: "
                      f"{CALISTIRICI}\n", "hata")
        self.after(100, self._kuyrugu_isle)
        self.protocol("WM_DELETE_WINDOW", self._kapat)

    # ------------------------------------------------------------------ görünüm
    def _stiller(self):
        stil = ttk.Style(self)
        if os.name != "nt" and "clam" in stil.theme_names():
            stil.theme_use("clam")
        temel = ("Segoe UI", 10)
        self.option_add("*Font", temel)
        stil.configure(".", font=temel)
        stil.configure("Baslik.TLabel", font=("Segoe UI", 15, "bold"))
        stil.configure("Kalin.TLabel", font=("Segoe UI", 10, "bold"))
        stil.configure("Soluk.TLabel", foreground="#6B6B6B", font=("Segoe UI", 9))
        stil.configure("Aciklama.TLabel", foreground="#444444")
        stil.configure("Ana.TButton", font=("Segoe UI", 10, "bold"), padding=(14, 4))
        stil.configure("Treeview", rowheight=24)

    def _aciklama(self, ebeveyn, metin):
        ttk.Label(ebeveyn, text=metin, style="Aciklama.TLabel", wraplength=1150,
                  justify="left").pack(anchor="w", pady=(0, 6))

    # ------------------------------------------------------------------ Fotoğraf sekmesi
    def _foto_sekmesi(self):
        s = ttk.Frame(self.defter, padding=12)
        self.defter.add(s, text="  Fotoğraf  ")
        self._aciklama(s, "İnternetten indirdiğin ya da kendi çektiğin fotoğraflardaki her yüzün duygusunu "
                          "tahmin eder. Bir ya da birkaç dosya, ya da bütün bir klasör seçebilirsin.")
        satir = ttk.Frame(s)
        satir.pack(fill="x")
        self.foto_yol = tk.StringVar()
        ttk.Entry(satir, textvariable=self.foto_yol).pack(side="left", fill="x", expand=True)
        ttk.Button(satir, text="Dosya seç…", command=self._foto_dosya_sec).pack(side="left", padx=(6, 0))
        ttk.Button(satir, text="Klasör seç…", command=self._foto_klasor_sec).pack(side="left", padx=(6, 0))

        secenek = ttk.Frame(s)
        secenek.pack(fill="x", pady=(6, 8))
        self.foto_fer = tk.BooleanVar(value=False)
        ttk.Checkbutton(secenek, variable=self.foto_fer,
                        text="FER (FER-2013) ile de karşılaştır  (TensorFlow yüklendiği için daha yavaş)").pack(side="left")
        btn = ttk.Button(secenek, text="Değerlendir", style="Ana.TButton", command=self._foto_baslat)
        btn.pack(side="right")
        self.calisma_dugmeleri.append(btn)
        self.foto_klasor_btn = ttk.Button(secenek, text="Çıktı klasörünü aç", state="disabled",
                                          command=lambda: klasoru_ac(self.foto_klasor))
        self.foto_klasor_btn.pack(side="right", padx=6)
        self.foto_klasor = None

        govde = ttk.PanedWindow(s, orient="horizontal")
        govde.pack(fill="both", expand=True)
        sol = ttk.Frame(govde)
        govde.add(sol, weight=2)
        kolonlar = [("dosya", "Dosya", 190), ("yuz", "Yüz", 40), ("tahmin", "Tahmin", 90),
                    ("olasilik", "Olasılık", 70), ("valence", "Valence", 65), ("arousal", "Arousal", 65),
                    ("fer", "FER", 100)]
        self.foto_tablo = ttk.Treeview(sol, columns=[k for k, _, _ in kolonlar], show="headings",
                                       selectmode="browse")
        for k, baslik, gen in kolonlar:
            self.foto_tablo.heading(k, text=baslik)
            self.foto_tablo.column(k, width=gen, anchor="w" if k == "dosya" else "center",
                                   stretch=(k == "dosya"))
        kay = ttk.Scrollbar(sol, command=self.foto_tablo.yview)
        self.foto_tablo.configure(yscrollcommand=kay.set)
        kay.pack(side="right", fill="y")
        self.foto_tablo.pack(fill="both", expand=True)
        self.foto_tablo.bind("<<TreeviewSelect>>", self._foto_secildi)

        sag = ttk.Frame(govde, padding=(10, 0, 0, 0))
        govde.add(sag, weight=3)
        self.foto_resim = ResimAlani(sag, "Fotoğrafları seçip “Değerlendir”e bas.\n"
                                          "Sonuçlar solda listelenir; bir satıra tıklayınca işaretli fotoğraf burada görünür.")
        self.foto_resim.pack(fill="both", expand=True)
        self.foto_cubuk = tk.Canvas(sag, height=190, background="#FFFFFF", highlightthickness=0)
        self.foto_cubuk.pack(fill="x", pady=(8, 0))
        self.foto_cubuk.bind("<Configure>", lambda _e: self._foto_cubuk_ciz())
        self._secili_foto = None

    def _foto_dosya_sec(self):
        secim = filedialog.askopenfilenames(
            title="Fotoğrafları seç",
            filetypes=[("Fotoğraflar", "*.jpg *.jpeg *.png *.bmp *.webp *.jfif *.tif *.tiff"),
                       ("Tüm dosyalar", "*.*")])
        if secim:
            self.foto_yol.set(" ; ".join(secim))

    def _foto_klasor_sec(self):
        secim = filedialog.askdirectory(title="Fotoğraf klasörünü seç")
        if secim:
            self.foto_yol.set(secim)

    def _foto_baslat(self):
        girdiler = [p.strip().strip('"') for p in self.foto_yol.get().split(";") if p.strip()]
        if not girdiler:
            messagebox.showinfo("Fotoğraf", "Önce bir fotoğraf ya da klasör seç.")
            return
        eksik = [g for g in girdiler if not Path(g).exists()]
        if eksik:
            messagebox.showerror("Fotoğraf", "Bulunamadı:\n" + "\n".join(eksik[:5]))
            return
        self.foto_tablo.delete(*self.foto_tablo.get_children())
        self.foto_satirlari.clear()
        self._secili_foto = None
        self.foto_resim.temizle("Değerlendiriliyor…")
        self._foto_cubuk_ciz()
        komut = [CALISTIRICI, "-u", str(PANEL / "foto_degerlendir.py"), *girdiler]
        if self.foto_fer.get():
            komut.append("--fer")
        self._baslat(komut, PANEL, "Fotoğraf değerlendirme", self._foto_bitti, durdurulabilir=True)

    def _foto_bitti(self, kod, klasor):
        self.foto_klasor = klasor
        if klasor:
            self.foto_klasor_btn.configure(state="normal")
        csv_yol = Path(klasor) / "sonuclar.csv" if klasor else None
        if not csv_yol or not csv_yol.exists():
            self.foto_resim.temizle("Sonuç üretilemedi. Ayrıntı için alttaki günlüğe bak.")
            return
        self._foto_tabloyu_doldur(csv_yol)

    def _foto_tabloyu_doldur(self, csv_yol):
        klasor = csv_yol.parent
        with open(csv_yol, encoding="utf-8-sig") as f:
            satirlar = list(csv.DictReader(f))
        self.foto_tablo.delete(*self.foto_tablo.get_children())
        self.foto_satirlari.clear()
        for k, s in enumerate(satirlar):
            yuz_var = s.get("yuz_no") not in ("", "0", None)
            olas = f"%{float(s['olasilik']) * 100:.0f}" if s.get("olasilik") else ""
            fer = s.get("fer_tahmin", "")
            if fer and s.get("fer_olasilik"):
                fer += f" %{float(s['fer_olasilik']) * 100:.0f}"
            iid = f"s{k}"
            self.foto_tablo.insert("", "end", iid=iid, values=(
                Path(s["dosya"]).name, s.get("yuz_no", "") if yuz_var else "–", s.get("tahmin", ""), olas,
                s.get("valence", "")[:5] if yuz_var else "", s.get("arousal", "")[:5] if yuz_var else "", fer))
            s["_klasor"] = str(klasor)
            self.foto_satirlari[iid] = s
        if satirlar:
            self.foto_tablo.selection_set("s0")
            self.foto_tablo.see("s0")

    def _foto_secildi(self, _e=None):
        secim = self.foto_tablo.selection()
        if not secim:
            return
        s = self.foto_satirlari.get(secim[0])
        if not s:
            return
        self._secili_foto = s
        self.foto_resim.goster(Path(s["_klasor"]) / s["isaretli"])
        self._foto_cubuk_ciz()

    def _foto_cubuk_ciz(self):
        c = self.foto_cubuk
        c.delete("all")
        s = self._secili_foto
        w = max(c.winfo_width(), 300)
        if not s or s.get("yuz_no") in ("", "0", None):
            c.create_text(w // 2, 95, fill="#888888",
                          text="Seçili yüzün 8 duygu olasılığı burada görünür." if not s else "Bu fotoğrafta yüz bulunamadı.")
            return
        olasiliklar = [float(s.get(a) or 0) for a in DUYGULAR_ASCII]
        en = max(range(8), key=lambda i: olasiliklar[i])
        sol, cubuk_gen = 110, w - 110 - 70
        for i, (ad, p) in enumerate(zip(DUYGULAR_TR, olasiliklar)):
            y = 14 + i * 21
            c.create_text(sol - 8, y, text=ad, anchor="e", font=("Segoe UI", 9, "bold" if i == en else "normal"))
            c.create_rectangle(sol, y - 7, sol + cubuk_gen, y + 7, fill="#EEEEEE", outline="")
            c.create_rectangle(sol, y - 7, sol + max(1, int(cubuk_gen * p)), y + 7,
                               fill="#2E7D32" if i == en else "#5B9BD5", outline="")
            c.create_text(sol + cubuk_gen + 8, y, text=f"%{p * 100:.0f}", anchor="w", font=("Segoe UI", 9))
        alt = (f"Valence {float(s['valence']):+.2f}   ·   Arousal {float(s['arousal']):+.2f}"
               + (f"   ·   FER: {s['fer_tahmin']}" if s.get("fer_tahmin") else ""))
        c.create_text(sol, 14 + 8 * 21, text=alt, anchor="w", font=("Segoe UI", 9), fill="#444444")

    # ------------------------------------------------------------------ Video sekmesi
    def _video_sekmesi(self):
        s = ttk.Frame(self.defter, padding=12)
        self.defter.add(s, text="  Video dosyası  ")
        self._aciklama(s, "Önceden çekilmiş bir videoyu kare kare işler: en büyük yüzün duygusunu, valence ve "
                          "arousal değerlerini zamana göre kaydeder, görüşme özeti ve işaretli bir video üretir.")
        satir = ttk.Frame(s)
        satir.pack(fill="x")
        self.video_yol = tk.StringVar()
        ttk.Entry(satir, textvariable=self.video_yol).pack(side="left", fill="x", expand=True)
        ttk.Button(satir, text="Video seç…", command=self._video_sec).pack(side="left", padx=(6, 0))

        secenek = ttk.Frame(s)
        secenek.pack(fill="x", pady=(6, 8))
        ttk.Label(secenek, text="Her").pack(side="left")
        self.video_adim = tk.IntVar(value=2)
        ttk.Spinbox(secenek, from_=1, to=10, width=4, textvariable=self.video_adim).pack(side="left", padx=4)
        ttk.Label(secenek, text="karede bir analiz  (1 = her kare, en ayrıntılı ama en yavaş)").pack(side="left")
        self.video_kaydet = tk.BooleanVar(value=True)
        ttk.Checkbutton(secenek, text="İşaretli video kaydet", variable=self.video_kaydet).pack(side="left", padx=20)
        btn = ttk.Button(secenek, text="Değerlendir", style="Ana.TButton", command=self._video_baslat)
        btn.pack(side="right")
        self.calisma_dugmeleri.append(btn)

        self.video_ozet = OzetAlani(s, "Bir video seçip “Değerlendir”e bas.\n"
                                       "Bitince valence–arousal zaman grafiği ve özet burada görünür.")
        self.video_ozet.pack(fill="both", expand=True)

    def _video_sec(self):
        secim = filedialog.askopenfilename(
            title="Videoyu seç",
            filetypes=[("Videolar", "*.mp4 *.avi *.mov *.mkv *.wmv *.webm *.m4v"), ("Tüm dosyalar", "*.*")])
        if secim:
            self.video_yol.set(secim)

    def _video_baslat(self):
        yol = self.video_yol.get().strip().strip('"')
        if not yol or not Path(yol).is_file():
            messagebox.showinfo("Video", "Önce bir video dosyası seç.")
            return
        try:
            adim = max(1, int(self.video_adim.get()))
        except (tk.TclError, ValueError):
            adim = 2
        komut = [CALISTIRICI, "-u", str(PANEL / "video_degerlendir.py"), yol, "--adim", str(adim)]
        if not self.video_kaydet.get():
            komut.append("--video-yok")
        self.video_ozet.resim.temizle("Video işleniyor… İlerleme alttaki çubukta.")
        self._baslat(komut, PANEL, "Video değerlendirme",
                     lambda kod, klasor: self.video_ozet.yukle(klasor), durdurulabilir=True)

    # ------------------------------------------------------------------ Canlı sekmesi
    def _canli_sekmesi(self):
        s = ttk.Frame(self.defter, padding=12)
        self.defter.add(s, text="  Canlı kamera  ")
        self._aciklama(s, "Kameradan anlık duygu tanıma. Ayrı bir kamera penceresi açılır; q tuşuyla ya da "
                          "“Bitir ve kaydet”le bitirince özet otomatik çıkar. Varsayılan olarak yalnızca sayılar "
                          "kaydedilir, görüntü kaydedilmez. Ödevin oturum kaydına dokunmaz.")
        secenek = ttk.Frame(s)
        secenek.pack(fill="x", pady=(0, 8))
        ttk.Label(secenek, text="Kamera numarası").pack(side="left")
        self.canli_kamera = tk.IntVar(value=0)
        ttk.Spinbox(secenek, from_=0, to=5, width=4, textvariable=self.canli_kamera).pack(side="left", padx=6)
        self.canli_video = tk.BooleanVar(value=False)
        ttk.Checkbutton(secenek, variable=self.canli_video,
                        text="İşaretli görüntüyü videoya da kaydet (yüzün kaydedilir)").pack(side="left", padx=20)
        btn = ttk.Button(secenek, text="Başlat", style="Ana.TButton", command=self._canli_baslat)
        btn.pack(side="right")
        self.calisma_dugmeleri.append(btn)
        self.canli_bitir_btn = ttk.Button(secenek, text="Bitir ve kaydet", state="disabled", command=self._durdur)
        self.canli_bitir_btn.pack(side="right", padx=6)

        self.canli_ozet = OzetAlani(s, "“Başlat”a bas, kamera penceresi açılsın.\n"
                                       "Oturumu bitirince özet burada görünür.")
        self.canli_ozet.pack(fill="both", expand=True)

    def _canli_baslat(self):
        try:
            kamera = int(self.canli_kamera.get())
        except (tk.TclError, ValueError):
            kamera = 0
        komut = [CALISTIRICI, "-u", str(PANEL / "canli_degerlendir.py"), "--kamera", str(kamera)]
        if self.canli_video.get():
            komut.append("--video-kaydet")
        self.canli_ozet.resim.temizle("Kamera penceresi açık. Bitirmek için q'ya ya da “Bitir ve kaydet”e bas.")

        def bitti(kod, klasor):
            self.canli_bitir_btn.configure(state="disabled")
            self.canli_ozet.yukle(klasor)

        if self._baslat(komut, PANEL, "Canlı kamera", bitti, durdurulabilir=True):
            self.canli_bitir_btn.configure(state="normal")

    # ------------------------------------------------------------------ Ödev scriptleri
    def _odev_sekmesi(self):
        s = ttk.Frame(self.defter, padding=12)
        self.defter.add(s, text="  Ödev scriptleri  ")
        self._aciklama(s, "Bunlar ödevin teslim edilen kodları (kod/ klasörü). Panel onları değiştirmez; VS Code'da "
                          "▶ ile çalıştırmakla aynı şekilde çalıştırır. Çıktılar her zamanki gibi kod/ciktilar/ "
                          "klasörüne yazılır. Kamera açan scriptlerden q ile çık.")
        liste = ttk.Frame(s)
        liste.pack(fill="both", expand=True)
        liste.columnconfigure(0, weight=1)
        satir_no = 0
        for dosya, argumanlar, baslik, aciklama, yedekler in ODEV_SCRIPTLERI:
            yol = KOD / dosya
            if not yol.exists():
                continue
            kutu = ttk.Frame(liste, padding=(4, 6))
            kutu.grid(row=satir_no, column=0, sticky="ew")
            kutu.columnconfigure(0, weight=1)
            ttk.Label(kutu, text=f"{baslik}   ·   {dosya} {' '.join(argumanlar)}".rstrip(),
                      style="Kalin.TLabel").grid(row=0, column=0, sticky="w")
            ttk.Label(kutu, text=aciklama, style="Aciklama.TLabel", wraplength=860,
                      justify="left").grid(row=1, column=0, sticky="w")
            btn = ttk.Button(kutu, text="Çalıştır",
                             command=lambda d=dosya, a=argumanlar, b=baslik, y=yedekler: self._odev_calistir(d, a, b, y))
            btn.grid(row=0, column=1, rowspan=2, padx=(10, 4))
            self.calisma_dugmeleri.append(btn)
            ttk.Button(kutu, text="VS Code'da aç", command=lambda p=yol: self._vscode_ac(p)).grid(
                row=0, column=2, rowspan=2)
            ttk.Separator(liste).grid(row=satir_no + 1, column=0, sticky="ew")
            satir_no += 2
        alt = ttk.Frame(s)
        alt.pack(fill="x", pady=(8, 0))
        ttk.Button(alt, text="kod/ciktilar klasörünü aç", command=lambda: klasoru_ac(KOD / "ciktilar")).pack(side="left")
        ttk.Button(alt, text="Yedekler klasörünü aç", command=lambda: klasoru_ac(YEDEKLER)).pack(side="left", padx=6)

    def _odev_calistir(self, dosya, argumanlar, baslik, yedekler):
        mevcut = [KOD / "ciktilar" / y for y in yedekler if (KOD / "ciktilar" / y).exists()]
        if mevcut:
            hedef = YEDEKLER / f"{Path(dosya).stem}_{datetime.now():%Y%m%d_%H%M%S}"
            if not messagebox.askokcancel(
                    baslik, f"Bu script şu dosyaların üzerine yazacak:\n  "
                            + "\n  ".join(p.name for p in mevcut)
                            + f"\n\nÖnce şuraya yedeklenecek:\n  {hedef}\n\nDevam edilsin mi?"):
                return
            hedef.mkdir(parents=True, exist_ok=True)
            for p in mevcut:
                shutil.copy2(p, hedef / p.name)
            self._yaz(f"Yedeklendi: {', '.join(p.name for p in mevcut)} -> {hedef}\n", "tamam")
        komut = [CALISTIRICI, "-u", str(KOD / dosya), *argumanlar]
        self._baslat(komut, KOD, baslik, None, durdurulabilir=False)

    def _vscode_ac(self, yol):
        try:
            subprocess.Popen(["code", str(yol)], shell=(os.name == "nt"),
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError:
            messagebox.showinfo("VS Code", "VS Code komut satırı (code) bulunamadı. Dosyanın klasörü açılıyor.")
            klasoru_ac(Path(yol).parent)

    # ------------------------------------------------------------------ Geçmiş sekmesi
    def _gecmis_sekmesi(self):
        s = ttk.Frame(self.defter, padding=12)
        self.defter.add(s, text="  Geçmiş  ")
        self._aciklama(s, "Panelden yapılan bütün analizler panel/ciktilar/ altında tarihli klasörlerde durur.")
        ust = ttk.Frame(s)
        ust.pack(fill="x", pady=(0, 6))
        ttk.Button(ust, text="Yenile", command=self._gecmisi_yukle).pack(side="left")
        ttk.Button(ust, text="Seçili klasörü aç", command=self._gecmis_klasor_ac).pack(side="left", padx=6)
        ttk.Button(ust, text="panel/ciktilar klasörünü aç", command=lambda: klasoru_ac(CIKTILAR)).pack(side="left")

        govde = ttk.PanedWindow(s, orient="horizontal")
        govde.pack(fill="both", expand=True)
        sol = ttk.Frame(govde)
        govde.add(sol, weight=2)
        self.gecmis_tablo = ttk.Treeview(sol, columns=("tur", "tarih", "ad"), show="headings", selectmode="browse")
        for k, b, gen in [("tur", "Tür", 100), ("tarih", "Tarih", 140), ("ad", "Klasör", 260)]:
            self.gecmis_tablo.heading(k, text=b)
            self.gecmis_tablo.column(k, width=gen, anchor="w")
        kay = ttk.Scrollbar(sol, command=self.gecmis_tablo.yview)
        self.gecmis_tablo.configure(yscrollcommand=kay.set)
        kay.pack(side="right", fill="y")
        self.gecmis_tablo.pack(fill="both", expand=True)
        self.gecmis_tablo.bind("<<TreeviewSelect>>", self._gecmis_secildi)
        self.gecmis_tablo.bind("<Double-1>", lambda _e: self._gecmis_klasor_ac())

        sag = ttk.Frame(govde, padding=(10, 0, 0, 0))
        govde.add(sag, weight=3)
        self.gecmis_resim = ResimAlani(sag, "Soldan bir analiz seç.")
        self.gecmis_resim.pack(fill="both", expand=True)
        self.gecmis_metin = ttk.Label(sag, text="", justify="left", font=("Consolas", 9))
        self.gecmis_metin.pack(anchor="w", pady=(6, 0))

    def _gecmisi_yukle(self):
        self.gecmis_tablo.delete(*self.gecmis_tablo.get_children())
        if not CIKTILAR.exists():
            return
        klasorler = sorted((p for p in CIKTILAR.iterdir() if p.is_dir()), key=lambda p: p.name.split("_", 1)[-1],
                           reverse=True)
        for p in klasorler:
            parcalar = p.name.split("_")
            tur = TUR_ADI.get(parcalar[0], parcalar[0])
            tarih = ""
            if len(parcalar) >= 3:
                try:
                    tarih = datetime.strptime(parcalar[1] + parcalar[2], "%Y%m%d%H%M%S").strftime("%d.%m.%Y %H:%M")
                except ValueError:
                    pass
            self.gecmis_tablo.insert("", "end", iid=str(p), values=(tur, tarih, p.name))

    def _gecmis_secildi(self, _e=None):
        secim = self.gecmis_tablo.selection()
        if not secim:
            return
        p = Path(secim[0])
        resim = p / "ozet.png"
        if not resim.exists():
            jpgler = sorted(p.glob("*.jpg"))
            resim = jpgler[0] if jpgler else None
        if resim:
            self.gecmis_resim.goster(resim)
        else:
            self.gecmis_resim.temizle("Bu klasörde gösterilecek görsel yok.")
        ozet = p / "ozet.txt"
        if ozet.exists():
            self.gecmis_metin.configure(text=ozet.read_text(encoding="utf-8"))
        elif (p / "sonuclar.csv").exists():
            with open(p / "sonuclar.csv", encoding="utf-8-sig") as f:
                satirlar = list(csv.DictReader(f))
            fotolar = len({s["dosya"] for s in satirlar})
            yuzler = sum(1 for s in satirlar if s.get("yuz_no") not in ("", "0"))
            self.gecmis_metin.configure(text=f"{fotolar} fotoğraf, {yuzler} yüz  ·  sonuclar.csv")
        else:
            self.gecmis_metin.configure(text="")

    def _gecmis_klasor_ac(self):
        secim = self.gecmis_tablo.selection()
        klasoru_ac(Path(secim[0]) if secim else CIKTILAR)

    def _sekme_degisti(self, _e=None):
        if self.defter.index("current") == 4:
            self._gecmisi_yukle()

    # ------------------------------------------------------------------ süreç yönetimi
    def _baslat(self, komut, cwd, baslik, bitince, durdurulabilir):
        if self.surec is not None:
            messagebox.showinfo("Meşgul", "Şu anda başka bir iş çalışıyor. Bitmesini bekle ya da durdur.")
            return False
        self.son_cikti = None
        self.bitince = bitince
        self.dur_dosyasi = None
        if durdurulabilir:
            self.dur_dosyasi = Path(tempfile.gettempdir()) / f"duygu_panel_dur_{os.getpid()}"
            self.dur_dosyasi.unlink(missing_ok=True)
            komut = [*komut, "--dur-dosyasi", str(self.dur_dosyasi)]
        gosterilen = ["python" if k == CALISTIRICI else k for k in komut if k != "-u"]
        gosterilen = [k for k in gosterilen if k != "--dur-dosyasi" and k != str(self.dur_dosyasi)]
        self._yaz(f"\n▶ {baslik}   ({datetime.now():%H:%M:%S})\n", "komut")
        self._yaz(f"  {subprocess.list2cmdline(gosterilen)}\n", "komut")
        ortam = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", PYTHONUNBUFFERED="1",
                     TF_CPP_MIN_LOG_LEVEL="3")
        bayrak = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        try:
            self.surec = subprocess.Popen(komut, cwd=str(cwd), stdout=subprocess.PIPE,
                                          stderr=subprocess.STDOUT, env=ortam, creationflags=bayrak)
        except OSError as hata:
            self._yaz(f"Başlatılamadı: {hata}\n", "hata")
            self.surec = None
            return False
        threading.Thread(target=self._oku, args=(self.surec,), daemon=True).start()
        self.ilerleme.configure(mode="indeterminate")
        self.ilerleme.start(12)
        self.durum_yazi.set(f"Çalışıyor: {baslik}")
        self.dur_btn.configure(state="normal")
        for b in self.calisma_dugmeleri:
            b.configure(state="disabled")
        return True

    def _oku(self, surec):
        for ham in iter(surec.stdout.readline, b""):
            self.kuyruk.put(("satir", ham.decode("utf-8", errors="replace").rstrip("\r\n")))
        surec.wait()
        self.kuyruk.put(("bitti", surec.returncode))

    def _kuyrugu_isle(self):
        try:
            while True:
                tur, veri = self.kuyruk.get_nowait()
                if tur == "satir":
                    self._satir_isle(veri)
                else:
                    self._is_bitti(veri)
        except queue.Empty:
            pass
        self.after(100, self._kuyrugu_isle)

    def _satir_isle(self, satir):
        es = ILERLEME.match(satir)
        if es:
            yapilan, toplam = int(es.group(1)), int(es.group(2))
            if toplam > 0:
                if str(self.ilerleme.cget("mode")) != "determinate":
                    self.ilerleme.stop()
                    self.ilerleme.configure(mode="determinate", maximum=100)
                self.ilerleme["value"] = min(100, yapilan * 100 / toplam)
            return
        es = CIKTI.match(satir)
        if es:
            self.son_cikti = Path(es.group(1))
            return
        etiket = "hata" if ("Traceback" in satir or "Error" in satir or satir.startswith("  ! ")) else None
        self._yaz(satir + "\n", etiket)

    def _is_bitti(self, kod):
        self.surec = None
        self.ilerleme.stop()
        self.ilerleme.configure(mode="determinate", maximum=100)
        self.ilerleme["value"] = 100 if kod == 0 else 0
        if self.dur_dosyasi is not None:
            self.dur_dosyasi.unlink(missing_ok=True)
        self.dur_btn.configure(state="disabled", text="Durdur")
        for b in self.calisma_dugmeleri:
            b.configure(state="normal")
        if kod == 0:
            self.durum_yazi.set("Bitti.")
            self._yaz("✔ Bitti.\n", "tamam")
        else:
            self.durum_yazi.set(f"Hata ya da durduruldu (çıkış kodu {kod}). Günlüğe bak.")
            self._yaz(f"✖ Çıkış kodu {kod}\n", "hata")
        if self.bitince is not None:
            try:
                self.bitince(kod, self.son_cikti)
            except Exception as hata:                     # sonuç ekranı hatası paneli kapatmasın
                self._yaz(f"Sonuç gösterilemedi: {hata}\n", "hata")
        self.bitince = None

    def _durdur(self):
        if self.surec is None:
            return
        if self.dur_dosyasi is not None and not self.dur_dosyasi.exists():
            self.dur_dosyasi.touch()                  # script işini düzgünce bitirir ve kaydeder
            self.durum_yazi.set("Durduruluyor… (tekrar basarsan hemen sonlandırılır)")
            self.dur_btn.configure(text="Hemen sonlandır")
            return
        if self.dur_dosyasi is None and not messagebox.askyesno(
                "Durdur", "Bu script hemen sonlandırılacak ve o ana kadarki kaydı yazamayabilir.\n"
                          "Kamera açan scriptlerde kaydın yazılması için kamera penceresinde q'ya basman daha iyi.\n\n"
                          "Yine de sonlandırılsın mı?"):
            return
        self.surec.terminate()
        self._yaz("İş sonlandırıldı.\n", "hata")

    def _yaz(self, metin, etiket=None):
        self.gunluk.configure(state="normal")
        self.gunluk.insert("end", metin, etiket or ())
        if int(self.gunluk.index("end-1c").split(".")[0]) > 4000:
            self.gunluk.delete("1.0", "1000.0")
        self.gunluk.see("end")
        self.gunluk.configure(state="disabled")

    def _gunluk_temizle(self):
        self.gunluk.configure(state="normal")
        self.gunluk.delete("1.0", "end")
        self.gunluk.configure(state="disabled")

    def _kapat(self):
        if self.surec is not None:
            if not messagebox.askyesno("Çıkış", "Çalışan bir iş var. Durdurup çıkılsın mı?"):
                return
            self.surec.terminate()
        if self.dur_dosyasi is not None:
            self.dur_dosyasi.unlink(missing_ok=True)
        self.destroy()


def main():
    # Panel sistemdeki Python ile açıldıysa (ör. VS Code'da yanlış yorumlayıcı seçiliyse)
    # kendini env_fer ile yeniden başlatır; böylece resimler (Pillow) ve modeller hep bulunur.
    if not env_fer_icinde_mi() and "--yeniden" not in sys.argv:
        pythonw = ENV_PYTHON.with_name("pythonw.exe")
        hedef = pythonw if pythonw.exists() else ENV_PYTHON
        print(f"Panel env_fer ortamında yeniden açılıyor: {hedef}")
        subprocess.Popen([str(hedef), str(Path(__file__).resolve()), "--yeniden"], cwd=str(PANEL))
        return
    if os.name == "nt":                               # yüksek çözünürlüklü ekranlarda bulanıklığı önler
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass
    CIKTILAR.mkdir(exist_ok=True)
    Panel().mainloop()


if __name__ == "__main__":
    main()
