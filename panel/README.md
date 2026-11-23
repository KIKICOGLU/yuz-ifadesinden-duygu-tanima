# Duygu Tanıma – Kontrol Paneli (ek arayüz)

Bu klasör ödevin teslim edilen kodlarından (`kod/`) **ayrıdır**. Panel, `kod/` içindeki hiçbir
dosyayı değiştirmez; oradan yalnızca YuNet model dosyasını okur ve istenirse ödev scriptlerini
VS Code'dan çalıştırır gibi çalıştırır. Kendi çıktılarını `panel/ciktilar/` klasörüne yazar.

```
duygu-tanima/
├── kod/        ← ödev (teslim edilen kodlar, dokunulmaz)
└── panel/      ← bu klasör (ek arayüz)
    ├── panel.py               kontrol paneli penceresi
    ├── motor.py               ortak motor: YuNet + HSEmotion, çizim, özet
    ├── foto_degerlendir.py    fotoğraf değerlendirme
    ├── video_degerlendir.py   kayıtlı video değerlendirme
    ├── canli_degerlendir.py   anlık kamera değerlendirme
    ├── ciktilar/              panelin bütün çıktıları (otomatik oluşur)
    └── yedekler/              ödev kayıtlarının yedekleri (otomatik oluşur)
```

## Paneli açmak

PowerShell'de:

```powershell
cd C:\Projeler\duygu-tanima
.\env_fer\Scripts\Activate.ps1
python panel\panel.py
```

VS Code'da: `panel/panel.py` dosyasını aç, sağ alttan yorumlayıcı olarak **env_fer**'i seç ve ▶ düğmesine bas.

## Sekmeler

| Sekme | Ne yapar | Çalıştırdığı script |
|---|---|---|
| Fotoğraf | İnternetten indirilen ya da çekilmiş fotoğraflardaki her yüzün duygusunu tahmin eder. Bir ya da birkaç dosya, ya da bütün bir klasör seçilebilir. İstenirse FER ile karşılaştırır. | `foto_degerlendir.py` |
| Video dosyası | Kayıtlı bir videoyu kare kare işler; valence–arousal zaman grafiği, görüşme özeti ve işaretli video üretir. | `video_degerlendir.py` |
| Canlı kamera | Kameradan anlık duygu tanıma. q ile ya da “Bitir ve kaydet” ile bitince özet otomatik çıkar. Ödevin `oturum_kaydi.csv` dosyasına dokunmaz. | `canli_degerlendir.py` |
| Ödev scriptleri | `kod/` klasöründeki scriptleri değiştirmeden çalıştırır, “VS Code'da aç” ile dosyayı açar. | `kod/*.py` |
| Geçmiş | Panelden yapılan önceki analizleri listeler ve gösterir. | – |

Alttaki günlük alanında her işin tam komutu yazılır; aynı komutu VS Code terminaline yapıştırıp
panelsiz de çalıştırabilirsin.

**Durdur düğmesi:** İlk basışta iş düzgünce durdurulur ve o ana kadarki sonuçlar kaydedilir.
İkinci basışta iş hemen sonlandırılır.

## Panelsiz çalıştırma (VS Code)

Her script tek başına çalışır. Argüman vermeden çalıştırılırsa (ör. VS Code'da ▶) dosya seçme
penceresi açılır.

```powershell
cd C:\Projeler\duygu-tanima\panel

python foto_degerlendir.py                              # dosya seçme penceresi
python foto_degerlendir.py C:\Resimler\resim.jpg  
python foto_degerlendir.py C:\Fotograflar --fer         # klasör + FER karşılaştırması

python video_degerlendir.py                             # video seçme penceresi
python video_degerlendir.py gorusme.mp4 --adim 3        # her 3 karede bir analiz (daha hızlı)
python video_degerlendir.py gorusme.mp4 --video-yok     # işaretli video kaydetme

python canli_degerlendir.py                             # 0 numaralı kamera
python canli_degerlendir.py --kamera 1 --video-kaydet   # başka kamera, görüntüyü de kaydet
```

## Çıktılar

Her analiz `panel/ciktilar/` altında tarihli bir klasöre yazılır:

| Klasör | İçerik |
|---|---|
| `foto_<tarih>/` | `sonuclar.csv` (her yüz için tahmin, 8 olasılık, valence, arousal) ve işaretli fotoğraflar |
| `video_<tarih>_<ad>/` | `kayit.csv`, `ozet.png`, `ozet.txt`, `isaretli.mp4` |
| `canli_<tarih>/` | `kayit.csv`, `ozet.png`, `ozet.txt` (istenirse `isaretli.mp4`, `ekran_*.png`) |

`kayit.csv`, ödevdeki `oturum_kaydi.csv` ile aynı biçimdedir; özetteki ölçütler de
`oturum_ozeti.py` ile aynıdır (olumsuz an: valence < −0,2).

## Ödev dosyalarıyla ilişkisi

- Panelin analizleri ödevdeki ana demoyla aynı yöntemi kullanır: YuNet → %15 payla kırpma →
  HSEmotion `enet_b0_8_va_mtl` (8 duygu + valence + arousal).
- “Ödev scriptleri” sekmesi `kod/` içindeki scriptleri kendi klasörlerinde çalıştırır; çıktıları her
  zamanki gibi `kod/ciktilar/` içine yazılır.
- `duygu_demo.py` ve `karsilastirma_canli.py` çalıştırılmadan önce, üzerine yazacakları kayıtlar
  (`oturum_kaydi.csv`, `oturum_ozeti.*`, `sonuclar.csv`) `panel/yedekler/` klasörüne kopyalanır.
  Rapordaki sayıları değiştirmek istemiyorsan bu scriptleri çalıştırdıktan sonra yedeği geri kopyala.

## Notlar

- Fotoğraf biçimleri: jpg, png, bmp, webp, tif. iPhone'un HEIC biçimi desteklenmez; önce jpg'ye çevir.
- Video biçimleri: mp4, avi, mov, mkv, wmv, webm.
- Türkçe karakterli klasörlerdeki (ör. `Masaüstü`) dosyalar da okunur.
- FER seçeneği TensorFlow yüklediği için ilk açılışta 10–20 saniye bekletir.
- Fotoğraf ve videolarda valence ile arousal, Russell modelinin [−1, 1] aralığına kırpılır.
