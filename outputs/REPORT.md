# Dört kaynağın yakın okuması ve canonical teze katkı değerlendirmesi

**Tez:** Mevsimlerin Ötesinde / *Beyond Seasons* (Samet Baytar)
**Görev türü:** Değerlendirme; revizyon değil. Canonical, kapanmış çerçeve (05) ve kaynak PDF'leri değiştirilmedi.
**Tarih:** 25 Eylül 2026
**Kaynak dosyaları:** [K1 Sagers](dossiers/K1-Sagers.md) · [K2 Adamson](dossiers/K2-Adamson.md) · [K3 Hulme](dossiers/K3-Hulme.md) · [K4 Carlill](dossiers/K4-Carlill.md)

**Gösterim.**
- Canonical'a her atıf `l. N («cümlenin ilk sözcükleri»)` biçimindedir; satır numaraları SHA-256 `4391c0a3…` sürümüne aittir.
- Kaynaktan harfiyen alıntılar “…” içindedir. Sayfa, kaynağın basılı folyosudur; Hulme için sayfa yoktur (§0).
- Kanıt etiketleri: `VERIFIED`, `PARTIALLY VERIFIED`, `PLAUSIBLE BUT UNVERIFIED`, `CONTRADICTED`, `ACCESS BLOCKED`, `NOT READ`.
- `inputs/novels/` bulunmadığı için roman sayfalarına ilişkin her iddia `NOT VERIFIED`dır.

---

## §0. Okunanlar ve son denetim

### Dosyalar

| Dosya | SHA-256 | Okunan kısım | Sürüm ve sayfa notu |
|---|---|---|---|
| `inputs/canonical/Dissertation-Draft-Live.md` | `4391c0a3ec229913c9ebf5df5e319ef915e4729a13eda3373ebe2647a3c3801e` | Tamamı (1146 satır) | 05'in yazıldığı sürüm `dfe4ef24…` idi. 05 §12'deki satır numaraları o sürüme aittir ve burada kullanılmadı. |
| `inputs/framework/05-KAPANMIS-CERCEVE.md` | `40ff5e77227789c9af7e5f57c6839bb24d9659267cf2b3e3c6b9c953c2bdb293` | Tamamı, §0–§15 | — |
| `inputs/sources/01-Sagers-2024-ch7-with-front-matter.pdf` | `5c20dd068732ce7201b5da37845328e0c4ffc34c4068a7b4cc68c7be2816f511` | 26 PDF sayfasının tamamı: ön sayfalar ve bölüm, basılı folyo 159–176 | Yayımlanmış sürüm; her sayfada folyo var |
| `inputs/sources/02-Adamson-et-al-2026-GEC.pdf` | `ac12dee159c84247d9c4de30596834a23f5ee7c7b1b20fdd9ae06a0c71a3acf6` | Tamamı, ss. 1–11 | Yayımlanmış sürüm; s. 1'de folyo basılı değil |
| `inputs/sources/02b-Adamson-Rapson-2024-WIREs.pdf` | `93eabced77dd42cccf514f1ac92856767878cd8cb12a7b3a3ac7b39223dfe440` | Tamamı, “1 of 8”–“8 of 8” | Yayımlanmış sürüm |
| `inputs/sources/03-Hulme-2018-Weber-submitted-version.pdf` | `757d78c5a67d0e6898d85a1d4eea2ca3e987733b034f1e0ee157ce0b21cce7f5` | 12 PDF sayfasının tamamı | Gönderim sürümü (3 Kasım 2017); yayımlanmış sayfa yok, alıntılar sayfasız |
| `inputs/sources/04-Carlill-2024-English-Studies.pdf` | `148b72355830c268f59cc2495eb99e64c7cec0bef186d3e4b17a46014e364050` | 23 PDF sayfasının tamamı: kapak ve folyo 1–22 | Çevrimiçi ilk yayım; folyo 1 basılı değil |
| `inputs/novels/` | — | yok | Roman sayfaları NOT VERIFIED |

**Dosya düzeni.** `inputs/` altındaki dosyalar yüklenen dosyaların bayt düzeyinde aynı kopyalarıdır; SHA-256 değerleri karşılaştırıldı. Bu dosyalar commit edilmedi ve yerel `.git/info/exclude` ile dışarıda tutuldu. Commit edilen tek şey `outputs/`tır.

**Okuma yöntemi.**
- Her PDF sayfa sayfa metne çevrildi (PyMuPDF). Metin katmanı beş dosyada da vardı.
- Kaynaklar ana oturumda baştan sona okundu. Alt ajan kullanılmadı.
- Folyolar sayfa üst ve alt bilgisinden alındı. İlk sayfaların folyo durumu görüntüyle denetlendi (Carlill ve Adamson ilk sayfaları).

### Son denetimin sonucu

Denetim, bu raporun ve dört dosyanın tamamı üzerinde `verify.py` betiğiyle yapıldı. Betik oturumun scratchpad dizinindedir ve commit edilmedi.

- **Canonical satırları.** `l. N («…»)` biçimindeki her atıf için «…» parçası canonical'ın N. satırında aranır. Sonuç: 198 atıf denetlendi; hepsi doğru satıra düştü, **hata 0**.
- **Çıplak atıflar.** Yanında «…» parçası olmayan `l. N` atıfları sayılır. Sonuç: **0**.
- **Kaynak alıntıları.** Her “…” alıntısı (15 karakterden uzun) ve her blok alıntı, beş PDF'in sayfa metinlerinde ve canonical'da aranır. Karşılaştırmadan önce boşluklar, satır sonu tireleri ve tırnak biçimleri normalleştirilir. Bulunduğu sayfanın folyosu, raporda verilen sayfayla karşılaştırılır. Sonuç: **238 alıntı** kaynak PDF'lerde bulundu; hepsinde verilen sayfa, bulunduğu folyoyla uyuşuyor (**uyuşmazlık 0**; aynı alıntı birden çok dosyada geçiyorsa her geçiş ayrı sayıldı). 9 alıntı yalnız canonical'da bulundu; bunlar canonical'ın roman alıntılarıdır ve roman düzeyinde NOT VERIFIED'dır. 22 dize hiçbir kaynakta yoktur ve hepsi beklenen türdendir: web ve künye başlıkları (15), PDF üst verisi (1), APA biçiminde küçük harfe çevrilmiş makale başlığı (2), terim (3; lieux de mémoire), iki harfiyen parçanın üç noktayla birleştirilmesi (1). **Kaynağa atfedilip kaynakta bulunmayan alıntı yoktur.**

Denetimde bulunan ve düzeltilen hatalar:
- Bir sayfa hatası: Carlill'deki Sally cümlesi s. 17 değil **s. 16**.
- İki satır parçası yanlış satıra bağlanmıştı.
- Birkaç alıntıda büyük harf ve üç nokta farkı vardı.

---

## §1. Hüküm

Dört kaynağın hiçbiri tezin sorusunu, savını, yöntemini ya da mimarisini yanlışlamaz. Hiçbiri tezin ana bulgusunu, yani asimetriyi, öncelemez.

- **Sagers.** Dörtlemeyi Antroposen ve derin zaman üzerinden okuyan en yakın çalışmadır. Romanlardaki havayı, iklim nedenini, sesleri ve bağı kimin kurduğunu hiç ele almaz; bölümde “weather” sözcüğü bir kez bile geçmez. Canonical'ın Sagers aktarımı büyük ölçüde doğrudur. İki yerde küçük doğruluk düzeltmesi gerekir: öneren öznenin kim olduğu (M2) ve Groom atfının sayfası (M3). Sagers'ın mevsimlerin her cildin şimdisiyle örtüştüğü genellemesi (s. 168) canonical'ın tarihli şimdileriyle çelişir, ama tez lehine; bu bir savunma notudur.
- **Adamson ve diğerleri.** İçerik düzeyindeki tek katkı buradan gelir. Tezin kaynaksız rakip açıklaması «olağan ihtiyat», bu çalışmayla hem kaynağa kavuşur hem daralır. Çalışmadaki katılımcıların çoğu, sorulmadan ve ihtiyatla da olsa, kendi yaşadıkları hava değişimini iklime bağlar. Olağan ihtiyat bu yüzden tek bir olayın bağlanmamasını açıklayabilir, ama romanlarda kendi yaşadığı havayı iklime bağlayan hiçbir sesin bulunmamasını tek başına açıklamaz (M1).
- **Hulme.** Metne gerekmez. İşi canonical'da Smith ve Liu, Dimick ve Clark ile zaten yapılıyor. Promptun varsaydığı açıklama («iklim soyuttur, hava yaşanır») Hulme'un savı değildir. Elde yalnız gönderim sürümü var.
- **Carlill.** Metne gerekmez. *The High House* karşılaştırması asimetrinin gerçekçiliğin zorunlu sonucu olmadığını gösterir, ama bir yazar tercihini kanıtlamaz.

**Toplam üç müdahale öneriliyor.** Biri iki cümlelik bir sınır eklemesi (M1), ikisi tek ifade ya da sayfa düzeyinde doğruluk düzeltmesi (M2, M3).

| Kaynak | Katkı türü | Öneri | Güven |
|---|---|---|---|
| K1 Sagers 2024 | Canonical'ın aktarımını doğrular; iki küçük doğruluk sorunu; özgünlüğü öncelemez; s. 168 genellemesi canonical'la çelişir (tez lehine) | **M2**, l. 143 («Sagers dizisel düzeni»); **M3**, l. 240 («Sagers onu s. 160'ta atıfsız kullanır») | Yüksek: basılı folyo, VERIFIED |
| K2 Adamson ve diğerleri 2026 (+ Adamson ve Rapson 2024) | Rakip açıklamayı sınırlar (olağan ihtiyat) | **M1**, l. 1022 («Tek bir olayı iklime bağlamaktan»); WIREs için öneri yok | Orta-yüksek: bulgular VERIFIED; örneklem küçük ve özgül |
| K3 Hulme 2018 | Kısmen açıklama (canonical'daki kaynaklarla aynı iş), kısmen rakip (sinekdoki), büyük ölçüde ilgisiz | Yok; savunma notu | Orta: gönderim sürümü; yayımlanmış sayfa ACCESS BLOCKED |
| K4 Carlill 2024/2025 | Karşılaştırma noktası; tür zorunluluğunu zayıflatır | Yok; savunma notu | Orta: *The High House* NOT READ; sayı sayfaları doğrulanmadı |

---

## §2. Kaynak başına özet

### K1 — Sagers 2024 ([dosya](dossiers/K1-Sagers.md))

- **Künye.** D. Lloyd ve W. Mortimer (ed.), *Digressions in Deep Time*, Lexington Books, 2024, ss. 159–176; VERIFIED. Promptun «D. Lloyd ve ark.» ifadesi yanlıştır: kitabın iki editörü var.
- **Savı.** Sagers'a göre Smith, beş kitaplık dizide bir “serialism” estetiği kurar. Bu estetik gerileme ve yenilenme döngülerini bir araya getirir ve şimdiyi kalınlaştırır (ss. 159–160). Sagers bölümün sonunda bu reparatif jesti sorgular (ss. 172, 174).
- **Olumsuz bulguyu etkilemez.** Romanlardaki havayı iklime bağlamaz; iklim yalnız genel bağlamlarda geçer (ss. 161, 166, 168, 172).
- **Canonical'daki aktarımlar.** Şu satırlar doğru: l. 143 («Yakın Smith eleştirisinde»), l. 203 («Sagers doğal, tarihsel, siyasal»), l. 241 («Sagers yayın süreçlerinin yapısını da»), l. 885 («Sagers da *Summer*'ın iki kampı») ve l. 966 («Sagers, dizinin reparatif görünen dönüşünün»).
- **İki kesinlik sorunu.** l. 143'te («Sagers dizisel düzeni») öneren özne Sagers olarak yazılmış; kaynakta Smith'tir. l. 240'ta («Sagers onu s. 160'ta atıfsız kullanır») Groom atfı yalnız s. 171'e verilmiş; atıf s. 170'te başlar.
- **2023 sempozyum sözü (s. 162).** Tırnaksız, sayfasız ve sözdizimsel olarak bozuk bir aktarımdır. Kullanılmamalı.

### K2 — Adamson ve diğerleri 2026 ([dosya](dossiers/K2-Adamson.md))

- **Künye.** *Global Environmental Change* 99, 103182; DOI 10.1016/j.gloenvcha.2026.103182; VERIFIED.
- **Tasarım.** Güneydoğu İngiltere'de 1942–1961 doğumlu 16 kişiyle yaşam öyküsü görüşmesi ve 101 çevrimiçi katkı. İklim değişikliği bilerek sorulmadı (s. 2).
- **Bulgu.** On altı kişiden on ikisi algıladıkları değişimi kendiliğinden insan kaynaklı iklim değişikliğine bağladı (ss. 3–4). Bunu çoğu kez bellekten kuşku duyarak ve ihtiyatla yaptı (ss. 4–5).
- **Özetteki “wary of generalisations” ifadesi (s. 1).** Bölüm 3.3'te görüldüğü gibi ulusal karakter hakkındaki genellemelere ilişkindir (s. 8), iklim atfına değil. Olağan ihtiyat için bu cümleye dayanmak yanlış okuma olur.
- **Canonical'a etkisi.** l. 93'teki («Romanların iklimi yaşanan bir güne») ve l. 1022'deki («Tek bir olayı iklime bağlamaktan») rakip açıklamayı daraltır. Promptta geçen §2.5'te bu rakip açıklama yoktur.
- **Kavramlar.** “weather-heritage” ve “prototype” kullanılmadan yalnız bulgular kullanılabilir. WIREs yazısı kavramın kuramsal önerisidir ve metne gerekmez.

### K3 — Hulme 2018, gönderim sürümü ([dosya](dossiers/K3-Hulme.md))

- **Savı.** İklim, hava deneyimiyle kültür arasında aracılık eden, normalleştirici ve güvence veren bir **fikir**dir. Anomaliyi tanımak ancak bu fikirle mümkündür. Antroposen'de istikrar kalmaz: “Climate-change” bir sinekdokiye döner ve iklim fikri bir “zombie idea” olur.
- **Tezle ilişkisi.** Hulme iklimi istatistiksel bir soyutlama olarak tanımlamayı açıkça reddeder. Promptun varsaydığı «iklim soyuttur, hava yaşanır» açıklaması bu yüzden Hulme'a atfedilemez; Adamson ve diğerleri bu formülü Jasanoff'a bağlar (s. 1). Hulme'un tezle ilgili fikirleri canonical'da zaten kaynaklıdır: Smith ve Liu, l. 105 («Smith ve Liu mevsimin genel bir tanımını verir»); Dimick, l. 737 («Dimick bir zamansal uyuşmazlığın»).
- **En güçlü karşı okuma.** Her anomali kaydı zaten iklimseldir; iklimi ayrı bir neden olarak söylememek Antroposen'in uygun biçimi olabilir. Tezin cevabı: bulgu ad ve neden düzeyindedir; sinekdoki savı da yakın nedenlerin açıkça ayrılmasını açıklamaz. l. 727 («Asimetri olayın yerinde değil») bunu zaten söyler.
- **Künye.** Yayımlanmış sayfalar (63–74) PLAUSIBLE BUT UNVERIFIED. Alıntılar sayfasız verildi.

### K4 — Carlill 2024 ([dosya](dossiers/K4-Carlill.md))

- **Künye.** *English Studies*, çevrimiçi 26 Kasım 2024; VERIFIED. Web kayıtlarına göre sonradan 106(3), 373–394 sayısına atanmış: PLAUSIBLE BUT UNVERIFIED.
- **Savı.** Carlill “climate realism”i (Badia, Cetinić ve Diamanti) Ghosh'a karşı savunur. *The High House*'u LeMenager'in “everyday Anthropocene” kavramı ve Berlant'tan türettiği “climate crisis ordinariness” üzerinden okur: yas, felç ve hareketsizlik.
- **Yaşanan hava ve iklim.** Carlill'in alıntıladığı parçalara göre roman yaşanan sel ve mevsim bozulmasını açık bir kriz çerçevesine yerleştirir (ss. 6, 12). Bunu “future anterior” bir geriye bakışla yapar; belirli bir günü “because” ile bağlayan bir cümle alıntılanmamıştır.
- **Tür mü, tercih mi?** Karşılaştırma bir tür zorunluluğunu zayıflatır, ama bir tercih kanıtlamaz; iki romanın öncülü ve zamansal bakışı farklıdır.
- **§4.4 ile.** Konu örtüşür, yöntem farklıdır.
- **Rakip açıklama.** Canonical'da «romanın realist kipi bunu gerektirir» diye bir rakip açıklama yoktur. Ghosh sorusunun cevabı l. 727'de («Asimetri olayın yerinde değil») zaten vardır.

---

## §3. Web taraması

### §3.1 Erişim durumu

**Çalışan tek araç WebSearch'tü.** Arama sonuçlarının başlık ve URL'lerini, bir de makinenin ürettiği özetleri verir. Bu bilgi burada yalnız **metadata** olarak kullanıldı; hiçbir arama sonucu «okunmuş metin» sayılmadı.

Aşağıdaki erişimler reddedildi (**ACCESS BLOCKED**):

| Denenen adres | Araç | Hata |
|---|---|---|
| `https://api.crossref.org/works/10.1080/0013838X.2024.2428932` | curl ve WebFetch | `CONNECT tunnel failed, response 403` / `EGRESS_BLOCKED` |
| `https://doi.org/10.1016/j.gloenvcha.2024.102822`, `https://doi.org/10.1080/0013838X.2024.2428932` | curl ve WebFetch | `CONNECT 403` / `EGRESS_BLOCKED` |
| `https://api.openalex.org/works/doi:10.1080/0013838X.2024.2428932` | WebFetch | `EGRESS_BLOCKED` |
| `https://www.tandfonline.com/doi/full/10.1080/0013838X.2024.2428932` | WebFetch | `EGRESS_BLOCKED` |
| `https://www.repository.cam.ac.uk/handle/1810/287739` | WebFetch | `EGRESS_BLOCKED` |
| `https://mikehulme.org/weather-worlds-in-the-anthropocene-and-the-end-of-climate/` | WebFetch | `EGRESS_BLOCKED` |
| `https://rowman.com/Action/Search/_/digressions%20in%20deep%20time` | WebFetch | `EGRESS_BLOCKED` |
| `https://www.lrb.co.uk/the-paper/v40/n05/christian-lorentzen/the-collage-police` | WebFetch | `EGRESS_BLOCKED` |
| `https://www.cambridge.org/core/books/cambridge-companion-to-british-postmodern-fiction/alternative-realisms/` | WebFetch | `EGRESS_BLOCKED` |

Bu yüzden Crossref ve OpenAlex ile atıf araması yapılamadı. Aşağıdaki olumsuz sonuçlar («bulunmadı») zayıftır.

### §3.2 Künyelerin doğrulanması

| Kaynak | PDF'in kendisinden | Web (yalnız metadata) | Sonuç |
|---|---|---|---|
| Sagers 2024 | Editörler Declan Lloyd ve Warren Mortimer; Lexington Books, Lanham, © 2024; ISBN 9781666948417 / 9781666948424; ss. 159–176: VERIFIED | Kitapçı ve Google Books listeleri aynı ISBN'i ve editörleri verir; yayın ayı Haziran 2024 (özet) | Canonical kaynakçası doğru: l. 1104 («Sagers, F. (2024).») |
| Adamson ve diğerleri 2026 | GEC 99, 103182; DOI 10.1016/j.gloenvcha.2026.103182; çevrimiçi 28 Mayıs 2026: VERIFIED | ScienceDirect kaydı, PII S0959378026000713 | Doğru |
| Adamson ve Rapson 2024 | *WIREs Climate Change* 15(6), e913; DOI 10.1002/wcc.913 (“How to cite this article”): VERIFIED | — | Doğru |
| Hulme 2018 | Gönderim sürümü; “scheduled for Fall 2018 (Issue 34.1)”: PARTIALLY VERIFIED | Sayfalar 63–74 (arama özetleri); başlık varyantı “in”/“of” | Sayfa verilemez: PLAUSIBLE BUT UNVERIFIED |
| Carlill 2024 | *English Studies*, çevrimiçi 26 Kasım 2024; DOI 10.1080/0013838X.2024.2428932: VERIFIED | Taylor & Francis sayfa başlığı “English Studies: Vol 106, No 3”; özet ss. 373–394, 2025 | Sayı kaydı: PLAUSIBLE BUT UNVERIFIED |

### §3.3 Bu dört kaynaktan birini Ali Smith ile birlikte anan çalışmalar (2019–2026)

- **Sagers, F. (2021). “Time on Our Hands in Ali Smith’s *Summer*.” *Moveable Type*, 13(1).** UCL Discovery 10138480. NOT READ. Arama özetinde bu yazıdan aktarılan iki cümle Sagers 2024'ün s. 163'teki cümleleriyle neredeyse aynıdır; yazı 2024 bölümünün bir öncülü görünüyor. Tez bakımından önemi yalnız künye düzeyindedir. Sagers'ın ilk kez 2021'de yazdığı bir iddiaya 2024 bölümüyle atıf yapmak bir öncelik sorunu yaratmaz, çünkü tez Sagers'ı öncelik için değil konum için anar.
- **Adamson (2024, 2026), Hulme (2018) ya da Carlill'i (2024) Ali Smith ile birlikte anan bir çalışma bulunmadı.** Yalnız WebSearch kullanıldı; atıf dizinleri açılamadı. Sonuç zayıftır.

### §3.4 Canonical kaynakçasında olmayan yeni adaylar (2023–2026)

Hepsi **NOT READ**. Bilgiler yalnız arama sonuçlarından alındı ve öneriye dönüştürülmedi.

| Künye (metadata) | Neden önemli olabilir | Öncelik |
|---|---|---|
| Schrag, N. (2023). Metamodernism and counterpublics: Politics, aesthetics, and porosity in Ali Smith’s *Seasonal Quartet*. *Textual Practice*, 37(12), 2019–2038. https://doi.org/10.1080/0950236X.2022.2150295 | Sanat ve siyaset konuşmalarını “counterpublic” olarak okur (özet). §4.4'teki karşılanış sahneleriyle (Florence, Iris) örtüşebilir | Düşük–orta |
| Andeweg, A., & van Amelsvoort, J. (2024). Introduction: The narrative ethics of Ali Smith’s *Seasonal Quartet*. *C21 Literature* (özel sayı girişi; cilt ve sayı belirsiz) | Canonical'ın kullandığı *C21* dosyasının (Andeweg ve Janković; van Amelsvoort; Wilson) çerçeve yazısı. Özete göre konukseverlik üzerinedir | Düşük (iklim tezi için) |
| Tate, A. (2025). Alternative realisms: Speculation, magic, and miracle in British postmodern fiction. In B. Nicol (Ed.), *The Cambridge Companion to British Postmodern Fiction*. Cambridge University Press | Özete göre Mitchell, Ali Smith ve Ishiguro'dan yararlanır. Dörtlemeyi ele alıp almadığı belirlenemedi. Gerçekçilik ve Ghosh sorusunda (§5, S12) işe yarayabilir | Orta (açılırsa) |
| “Time Travel Is Real”: Navigating the metamodernist oscillations in Ali Smith’s *Autumn* (2016). *Journal of English Studies* (Universidad de La Rioja); yıl muhtemelen 2024, yazar belirlenemedi | Metamodern salınım; iklim odağı görünmüyor | Düşük |
| “Oral Collage: A Study of the Storyteller-Migrant in Ali Smith’s *Seasonal Quartet*” (ResearchGate kaydı, yaklaşık 2024; yayın yeri belirsiz) | Göç ve anlatıcı; Beşinci Bölümle olası örtüşme | Düşük |
| “‘Going to Collage’: Ali Smith’s *Autumn* and Post-Liberal Democratic Imagination” (Academia.edu kaydı; yıl ve yer belirsiz) | Kolaj ve demokrasi; Lorentzen'in kolaj eleştirisiyle ilişkili olabilir | Düşük |
| “Between Myth and Reality: The Metamodern Oscillation of Sincerity and Irony in Ali Smith’s *Seasonal Quartet*” (Zenodo 13355250, yaklaşık 2024) | Metamodern okuma | Düşük |

İklim, hava, mevsim, Antroposen ya da gerçekçilik üzerinden okuyan ve canonical kaynakçasında olmayan yeni bir 2023–2026 çalışması bulunmadı. Bu alandaki arama sonuçları hep van Amelsvoort 2024'e, Bernard 2024'e ve Byrne 2020'ye döndü; üçü de canonical'da var.

### §3.5 Özellikle sorulan iki metin

- **Andrew Tate, “Alternative Realisms”.** Kitap 2025 tarihli, Cambridge University Press. Arama özetine göre bölüm Ali Smith'ten yararlanır; aynı kitabın başka bir bölümü *The Accidental*'ı ele alır. Dörtlemeyi ele alıp almadığı belirlenemedi (NOT READ; `cambridge.org` ACCESS BLOCKED). İlginç bir rastlantı: Andrew Tate, Sagers'ın bulunduğu *Digressions in Deep Time* cildinin açılış bölümünü de yazmıştır (“Introductory Keynote: Deep Time Poetics”, içindekiler s. vii).
- **Christian Lorentzen, “The Collage Police”.** *London Review of Books* 40(5), 8 Mart 2018. Arama özetine göre *Autumn* ve *Winter*'ı Brexit krizine “rapid-response” bir edebî yorum olarak ele alan bir inceleme. İklim okumasına karşı bir ses olup olmadığı **belirlenemedi** (NOT READ; `lrb.co.uk` ACCESS BLOCKED). 2023–2026 aralığının dışındadır.

### §3.6 Aralık dışında kalan ama kayda geçen metinler

- Conway, T. L. “Feminist Forms and Borderless Landscapes in Ali Smith’s *Seasonal Quartet*.” *Iowa Journal of Cultural Studies*, no. 21 (2021). Ekofeminist bir okumadır.
- Peace, G. J. *Living Entanglements and the Ecological Thought in the Works of Paul Kingsnorth, Tom McCarthy, and Ali Smith*. Yüksek lisans tezi, University of Tennessee at Chattanooga, 2021.
- “Ali Smith’s Poetic Attentions.” *Post45*, Mayıs 2022; yazar belirlenemedi.
- Massey University yüksek lisans tezi, “Ali Smith and the Seasonal Quartet: encounters with art”; yıl belirlenemedi.
- Amsterdam Üniversitesi AIHR, Mart 2023 etkinlik sayfası: “Climate Fiction, Realism, and Ali Smith’s Seasonal Quartet”. Bir yayın değildir.

Hepsi NOT READ.

---

## §4. Önerilen canonical müdahaleleri

Öneri eşiğinin dört koşulu şunlardır: (1) canonical'da adı konmuş bir boşluk ya da risk olmalı; (2) kaynak bir iddiayı güçlendirmeli, sınırlamalı ya da düzeltmeli; (3) yeni kavram, alt bölüm ya da soru, sav ve yöntem değişikliği gerekmemeli; (4) dayanak alıntı VERIFIED olmalı ve basılı sayfası belli olmalı. Üç önerinin üçü de dört koşulu karşılar.

### M1 — K2 · l. 1022 («Bu cevabın sınırları ve rakipleri de yazılmalıdır.») · olağan ihtiyat

- **Konum:** Sonuç, sınırlar ve rakipler paragrafı. Mevcut cümle l. 1022'de («Tek bir olayı iklime bağlamaktan kaçınan olağan ihtiyat») başlar.
- **Mevcut ifade:**
  > «Tek bir olayı iklime bağlamaktan kaçınan olağan ihtiyat, romanların siyasal odağı ve korunaklı yerlerde sapmanın küçük kalması (Dimick, 2024, s. 2) aynı mesafeyi açıklayabilir.»
- **Önerilen Türkçe ifade** (bu cümlenin hemen arkasına eklenecek iki cümle):
  > Bu açıklamaların ilki sınırlıdır: tehlike riskinin görece düşük olduğu güneydoğu İngiltere'de 1942–1961 doğumlu on altı kişiyle yapılan bir bellek çalışmasında, iklim değişikliği sorulmadığı hâlde katılımcıların on ikisi algıladıkları hava değişimini kendiliğinden insan kaynaklı iklim değişikliğine bağlamış, bunu çoğu kez belleklerinin doğruluğuna ilişkin bir kuşku kaydıyla yapmıştır (Adamson ve diğerleri, 2026, ss. 2–4). Olağan ihtiyat bu yüzden tek bir olayın iklime bağlanmamasını açıklayabilir; konuşanın kendi yaşadığı havadaki değişimi ihtiyatla da olsa iklime bağlayan bir sesin taramada ve yakın okumada bulunmamasını ise tek başına açıklamaz (§2.5).
- **Kaynakça kaydı (APA 7).** Kaynakçanın başına, Andersen'den önce eklenir:
  `KAYNAKÇA +` Adamson, G., Rapson, J., Woodham, A., Annabell, T., & Cantillon, L. (2026). “But they hardly ever freeze now”: Exploring weather-heritage, memory, and change in southeastern England. *Global Environmental Change*, *99*, 103182. https://doi.org/10.1016/j.gloenvcha.2026.103182
- **Dayanak alıntılar:**
  - “Because of our interest in how participants attribute meaning to their experiences, we intentionally did not ask participants to read their memories through climate change.” (s. 2)
  - “birth years ranging from 1942 to 1961” ve “leaving a sample of 16”, ayrıca bölge için “where the risk of hydrometeorological hazards is relatively low and social resilience relatively high” (s. 2)
  - “Whilst we deliberately did not mention climate change in life history prompts or questions on the portal, perceived changes to weather were a regular feature of the memories, and 12 of the life history participants specifically attributed this to anthropogenic climate change.” (ss. 3–4)
  - “Participants segued between memories of the past and experience of change in the present, often qualifying their responses with uncertainty over the accuracy of their memories.” (s. 4)

  Hepsi VERIFIED.
- **Hangi iddiayı nasıl değiştirir?** l. 1022 («Tek bir olayı iklime bağlamaktan») şu an kaynaksız bir önerme içerir: olağan ihtiyat «aynı mesafeyi açıklayabilir». Müdahale bu önermeyi kaynağa bağlar ve kapsamını daraltır. Olağan ihtiyat artık asimetrinin yalnız bir kısmını açıklar: tek bir olayın bağlanmamasını. Kalan kısım tezin kendi bulgusu olarak daha kesin yerini bulur: bu çalışmadaki katılımcıların ihtiyatla da olsa kurduğu türden bir bağ bile romanlarda kurulmaz. Bu bulgu l. 333'e («Tezin hükümleri belirli durumlarda») ve l. 335'e («Tarama, konuşanın kendi yerinde yaşanan belirli bir hava») dayanır. Tezin sorusu, savı ve yöntemi değişmez; asimetri bir tasarım iddiasına da dönüşmez, çünkü l. 93 («Tez yazarın tasarımı hakkında değil») yerinde kalır.
- **Risk.**
  - *Jüri itirazı:* Ampirik bir sosyal bilim çalışmasının edebî bir tezde yeri ne? Cevap: müdahale hiçbir pasajı doğrulamaz. Tezin kendi öne sürdüğü, gündelik davranışa ilişkin bir önermenin kapsamını ölçer.
  - *Örneklem:* Küçüktür; katılımcıların çoğu yaşlı, beyaz İngiliz ve yükseköğrenimlidir (s. 9). Görüşmeler 2020–2022 tarihlidir, bir kısmı romanların tarihli şimdilerinden sonradır. Cümle bu yüzden «on altı kişiyle» ve «bir bellek çalışmasında» der ve genelleme yapmaz.
  - *Tek olay atfı:* Çalışma bunu ayrıca sınamaz. Cümle bu yüzden tek olay konusunda yalnız tezin kendi önermesini korur («açıklayabilir»).
  - *Başlıktaki kavram:* Kaynakça başlığında “weather-heritage” geçer. Kavram metne alınmaz.

### M2 — K1 · l. 143 («Sagers dizisel düzeni») · öneren özne

- **Konum:** §1.3. Mevcut cümle l. 143'te («Sagers dizisel düzeni ve genel olarak anlatıyı “harmonizing and unifying forces” olarak önerir») başlar.
- **Mevcut ifade:**
  > «Sagers dizisel düzeni ve genel olarak anlatıyı “harmonizing and unifying forces” olarak önerir; aynı cümlede insan ve doğa döngülerinin “held together and in conflict” tutulduğunu söyler (s. 160) ve s. 172'de dizinin reparatif dönüşünün sınırlarını kendisi sorgular.»
- **Önerilen Türkçe ifade** (dipnot işareti yerinde kalır):
  > Sagers'a göre Smith dizisel düzeni ve genel olarak anlatıyı “harmonizing and unifying forces” olarak önerir; Sagers aynı cümlede insan ve doğa döngülerinin “held together and in conflict” tutulduğunu söyler (s. 160) ve s. 172'de dizinin reparatif dönüşünün sınırlarını kendisi sorgular.
- **Kaynakça kaydı:** Değişiklik yok.
- **Dayanak alıntı:** “I argue that in unifying these cyclical structures, or serialisms, within the quartet and coda of her cultural project, Smith proposes serialisms in particular, and narrative, in general, as harmonizing and unifying forces: the cycles of the human and natural world are held together and in conflict, by serialist artistic practices which share a rhythmic, cyclical, structure.” (s. 160) — VERIFIED
- **Hangi iddiayı nasıl değiştirir?** Önerinin kime ait olduğu düzelir: Smith'e aittir, Sagers bunu savunur. Bu, tezin kendi ilkesiyle, yani bağı kimin kurduğunu ayırmakla uyumludur. Cümlenin sonundaki «kendisi sorgular» da artık tutarlıdır: Sagers Smith'e yüklediği bir öneriyi sorgular. l. 143 («açıklanan nesneye dayanır») konumlandırması değişmez.
- **Risk:** Yok denecek kadar az.

### M3 — K1 · l. 240 («Sagers onu s. 160'ta atıfsız kullanır») · Groom atfının sayfası

- **Konum:** §1.7 dipnotları, `[^k1-sagers]`.
- **Mevcut ifade:**
  > «Sagers onu s. 160'ta atıfsız kullanır, s. 171'de ise çağdaş sanat bağlamından Groom'a atfeder ve “acknowledge[s] its multiple, interwoven temporalities” (çoklu, iç içe geçmiş zamansallıklarını kabul eden) glosunu oradan alır; Groom'a ilk göndermesi s. 161'dedir (Groom, 2013; Sagers'ın verdiği gönderme s. 16).»
- **Önerilen Türkçe ifade:**
  > Sagers onu s. 160'ta atıfsız kullanır, ss. 170–171'de ise çağdaş sanat bağlamından Groom'a atfeder ve “acknowledge[s] its multiple, interwoven temporalities” (çoklu, iç içe geçmiş zamansallıklarını kabul eden) glosunu oradan alır (s. 171); Groom'a ilk göndermesi s. 161'dedir (Groom, 2013; Sagers'ın verdiği gönderme s. 16).
- **Kaynakça kaydı:** Değişiklik yok.
- **Dayanak alıntılar:**
  - s. 170:
    > Smith’s engagement “with the remnant of previous times mark[s] a thickening of the present to acknowledge its multiple, interwoven temporalities” (16). This “thickening” is, to me, a distinctly Anthropocenic aesthetic.

    Bu cümleden hemen önce “(Groom 2013, 16)” göndermesi vardır.
  - s. 171:
    > through the ironic wider “thickening of the present” which “acknowledge[s] its multiple, interwoven temporalities” (Groom 2013, 16)

  Her ikisi VERIFIED.
- **Hangi iddiayı nasıl değiştirir?** Dipnot ifadenin kaynak içindeki kökenini sayfa sayfa belirtme iddiasındadır. Düzeltme bu iddiayı doğru hâle getirir. Aynı bilgi üst verideki revizyon kaydında da geçer: l. 9 («(2024, ss. 160, 171; Groom atfı)»). Orası metin gövdesi değildir ve isteğe bağlı olarak uyumlanabilir.
- **Risk:** Yok.

---

## §5. Savunma notları

Aşağıdakiler jürinin sorabileceği ve bu okumaların cevap verdiği sorulardır. Hiçbiri metne girmek zorunda değildir.

**S1. «Mevsim başlığı zaten her cildin şimdisidir; Sagers da öyle diyor.»**
Sagers bunu söyler: “the seasons align with the “present” narrative time-period of each of the original, seasonally-titled quartet” (s. 168; VERIFIED). Ama örnek olarak yalnız *Winter*'ı verir. Canonical'a göre *Spring*'in şimdisi Ekim 2018'dir ve *Autumn*'unki yazda başlar: l. 157 («Bir cilt de tek bir mevsim değildir») ve l. 314 («“October 2018” (Smith, 2019, s. 11)»). En yakın eleştirmenin bu genellemesi, tezin ad ile tarihli şimdi arasındaki ayrımının gerekli olduğunu gösterir. Roman düzeyi NOT VERIFIED.

**S2. «Sagers dörtlemeyi Antroposen ve derin zaman üzerinden zaten okumadı mı? Özgünlük nerede?»**
Sagers'ın kalınlaşması kendi sözüyle bir estetiktir (“This “thickening” is, to me, a distinctly Anthropocenic aesthetic.”, s. 170). Bölümde hava, iklim nedeni, ses ya da kip incelenmez; “weather” sözcüğü hiç geçmez. Asimetrinin bir öncülü yoktur. Tezin öncülleri canonical'da zaten anılır: N. Smith, Byrne ve van Amelsvoort, l. 205 («Tezin olumsuz bulgusunun da öncülleri vardır.»).

**S3. «Kobalt sahnesi okumanız Sagers'ta var mı?»**
Hayır. Sagers yenilenebilir enerjinin “continue to exploit humans, and require the continued mining of the earth’s resources” olduğunu söyler (s. 172). Bu Smith'in sözcük oyunlarına yaptığı bir benzetmedir; *Spring* ss. 250–251 pasajını okumaz. Tezin kobalt okuması pasaj düzeyindedir: l. 585 («Sonraki sahne bu bağımlılığın eşitsiz yüzünü Richard'ın şimdisine taşır.»).

**S4. «Olağan ihtiyat asimetriyi açıklamaz mı?»**
M1 kabul edilmese bile cevap aynıdır. Adamson ve diğerlerinin çalışmasında sorulmadan on altı kişiden on ikisi değişimi iklime bağlar (ss. 3–4), çoğu kez ihtiyatla: “it’s probably a climate change thing, isn’t it?” (s. 4). Bu çalışmaya göre olağan ihtiyat bağın yokluğunu değil, ihtiyatlı bir bağı öngörür.

*Uyarı:* Özetteki “Participants were wary of generalisations” ifadesi (s. 1) bu cevaba dayanak yapılmamalıdır. Bölüm 3.3'e göre bu ihtiyat ulusal karakter hakkındaki genellemelerle ilgilidir (s. 8).

**S5. «Korunaklı yerlerde sapma küçük kalır (Dimick); karakterler bu yüzden bağ kurmuyor olabilir.»**
Adamson ve diğerlerinin bölgesi “where the risk of hydrometeorological hazards is relatively low and social resilience relatively high” diye tanımlanır (s. 2). Orada da değişim algılanır, iklime bağlanır ve yer duygusunun aşınması olarak yaşanır (ss. 8–10). Canonical'da bu bileşen yorum olarak etiketlidir: l. 741 («Bu mesafe, gündelik deneyim ile»). Bu kadarı yeterlidir.

**S6. «Annenin *monoseason* yakınması gerçekçi bir halk söylemi mi, abartı mı?»**
Dış bağlam olarak: katılımcılar “the seasons just seem to be kind of merging” (LHI5) ve seasonality “doesn’t seem to be proper” (LHI9) diye yakınır (s. 5). Aynı söylemde neden çoğu kez adlandırılır; annenin cümlesinde adlandırılmaz, l. 405 («Annenin *monoseason* yakınması da nedensizdir»). Bu dış bağlam romanı doğrulamaz. Yalnız, nedensizliğin dikkat çekici bir özellik olduğunu destekler.

**S7. «1961'in sıcak şubatı ve bellek.»**
Katılımcılar uç yılları (1976) hem bir ölçüt hem “a harbinger of hotter weather to come” olarak anar (s. 7). Yazarlar ise bu yılların normu pekiştirdiğini söyler (s. 9). Bellek düzeltilir: “However, in my memory, it always snowed in December. I was surprised to realise later, when looking at weather records, that this wasn’t the case.” (s. 6). Tezin anomali ile eğilim ayrımı (l. 127 «Yıllar arasındaki olağan değişkenlik ile») ve Grace'in bellek düzeltmesi (l. 799 «Aynı sahne dizisi belleğin yanlış da olabileceğini gösterir») bu gündelik örüntüyle uyumludur.

**S8. «Aralıkta nergis: *Spring* s. 8 bir erken çiçeklenme mi anlatıyor?»**
Bir katılımcı 2018'in Boxing Day'inde nergis gördüğünü anlatır: “And this is Boxing Day. You don’t get daffodils on Boxing Day!” (s. 6). Bu, erken çiçeklenmenin tanınır bir gündelik işaret olduğunu gösterir; ama romandaki belirsizliği çözmez. Canonical iki okumayı da açık tutar: l. 533 («Misilleme listesi tehdidi belirli aylara ve evlere yerleştirir»). Böyle kalmalıdır.

**S9. «Hulme'a göre her anomali kaydı zaten iklimseldir. İklim yaşanan günden nasıl ayrı tutulur?»**
Tezin hükmü ad ve neden düzeyindedir. Bu düzey §2.3'te kurulur (l. 296 «Beş soru, nedenlerin ve iklimin romanlarda nasıl söylendiğini»), “nedensiz” sözcüğü de l. 739'da («*Nedensiz* ve *atfedilmemiş*») daraltılır. Norm Dimick ile ele alınır: l. 737 («Dimick bir zamansal uyuşmazlığın»).

*İsteğe bağlı sözcük düzeltmesi (öneri değildir):* l. 727 («Dizinin ana bulgusu bu asimetridir») ve l. 1010 («Dizinin ana bulgusu bu asimetridir») cümlelerindeki «iklim ise yaşanan günden ayrı tutulur» ifadesi «iklimin adı ise yaşanan günden ayrı tutulur» diye açılabilir. Eşik koşulu 1 karşılanmadığı için öneri olarak yazılmadı.

**S10. «Hulme'a göre iklim değişikliği bir sinekdokidir. Yan yana durma Antroposen'in doğru biçimi değil mi?»**
Hulme şunu söyler: “‘Climate-change’ is simply a synecdoche, a short-hand for a manifestation of aggregated changes which are at one and the same time environmental, economic, technological, social and cultural.” (gönderim sürümü, sayfasız). Bu sav iklimin neden ayrılmadığını açıklayabilir. Kimyasal nedenin neden açıkça ayrıldığını açıklamaz (“because”, W s. 119): l. 727 («Asimetri olayın yerinde değil»). Tez asimetriyi değerlendirmez: l. 743 («Bu çalışma asimetriyi ne bir eksiklik»).

*Dikkat:* «İklim soyuttur, hava yaşanır» formülünü Hulme'a atfetmeyin. Hulme iklimi yaşanan ve örtük bir fikir olarak tanımlar; soyutluk formülü Adamson ve diğerlerinde Jasanoff'a bağlanır (s. 1).

**S11. «Hulme'un ‘end of climate’ savı van Amelsvoort'un *monoseason* ufkunu desteklemiyor mu?»**
Destekler. Tez bilerek iki kutbun arasında durur: l. 725 («Tez böylece Wiemann'ın aldırmadan süren kadansı»). Hulme'u eklemek dengeyi bir kutba çeker.

**S12. «Neden Ghosh yok?»**
Carlill ve Sagers Ghosh'u tartışır. Carlill şöyle der: “Without doubt the most-cited criticism of literary realism in a climate changed and changing present belongs to Amitav Ghosh” (s. 1). Sagers Ghosh'u ss. 161 ve 166'da anar. Cevap:
- **(a)** Tez bir tür iddiası kurmaz.
- **(b)** Ghosh'un «felaketi dışarıda bırakma» tezi asimetriyi açıklamaz. *Winter*'daki kimyasal felaket de başka yerde yaşanır, ama “because” ile bağlanır: l. 727 («Asimetri olayın yerinde değil»).
- **(c)** Tezin gerçekçilik dayanağı Thieme'dir: l. 133 («Thieme gerçekçi romanın *long present*ini»). Carlill'e göre Thieme Ghosh'a itiraz eder (s. 3); Thieme'nin kendi sayfası NOT READ.

**S13. «*The High House* iklimi yaşanan güne bağlıyor. Smith'in asimetrisi o hâlde bir tercih değil mi?»**
Carlill'in alıntıladığı parçalara göre *The High House* yaşanan sel ve mevsim bozulmasını açık bir kriz çerçevesine yerleştirir: “the seasons falling into one another”, “the birds still singing in December” (s. 12). Bunu, felaketin gerçekleştiği bir gelecekten geriye bakarak yapar (“future anterior”, ss. 6–7). Belirli bir günü “because” ile bağlayan bir cümle alıntılanmamıştır. Karşılaştırma «gerçekçilik bunu gerektirir» önermesini zayıflatır, ama bir tercihi kanıtlamaz. Canonical'ın ihtiyatı doğrudur: l. 93 («Tez yazarın tasarımı hakkında değil»). *The High House* NOT READ.

**S14. «Uyarılar çizginiz Berlant ya da Carlill'in ‘crisis ordinariness’ı değil mi?»**
Konu örtüşür, yöntem ayrılır. Carlill kişilere bir duygulanım tanısı koyar: felç, yas, suç ortaklığı. Tez ses, kanal ve karşılanış sırasını kaydeder ve bunu yorum olarak yazar: l. 825 («Bu dizinin gösterdiği şey dardır»). Bir fark daha var:
- *The High House*'un anlatıcıları geçiştirmeyi geriye dönüp kendileri sahiplenir: “We noticed the changes, but we dismissed them, or said that they were only a part of the inevitable” (s. 8).
- Dörtlemede geçiştirme sahnelerde olur. En yakın dönüş Grace'in varsayımıdır: l. 823 («But say, just say it was.»).

Roman sayfaları NOT VERIFIED.

**S15. «Avustralya yangınları iki romanda da var.»**
*Summer*'da yangınlar Sacha'ya fotoğrafla gelir: l. 689 («Belirli bir olayın gezegensel bir felakete bağlandığı tek yer ss. 25–26'dır.»). *The High House*'ta Sally onları haberlerde izler. Carlill olayı şöyle tanımlar:
> The crisis “in the southern hemisphere” that Sally here narrates is unmistakably the January 2020 Australian bushfires (s. 7)

Uzaktaki felaketin dolayımlı gelmesi iki romanda ortaktır. Fark bağın açıklığındadır, olayın yerinde değil.

**S16. «Sagers'ın 2023 sempozyumunda aktardığı Smith sözünü neden kullanmadınız?»**
Söz tırnaksızdır, sayfasızdır ve cümle bozuktur (s. 162). Kaynak sözlü ve yayımlanmamış bir konuşmadır (s. 176). Konusu tezin sorusu dışındadır: eleştirel olanla yaratıcı olanın ayrılmazlığı. Tez yazar sözünü yalnız üretim kaydı için kullanır.

---

## §6. Reddedilenler ve nedenleri

| # | Aday | Karşılamadığı koşul | Neden |
|---|---|---|---|
| R1 | Sagers s. 168'e l. 157'de («Bir cilt de tek bir mevsim değildir») atıf yapmak | 1, 2 | Canonical'da adı konmuş bir boşluk yok. Kaynak tezin iddiasını güçlendirmez; tez kaynağı düzeltir. Savunma notu S1 |
| R2 | Sagers'ın aktardığı 2023 Smith sözü | 2, 4 | Doğrulanamayan dolaylı bir aktarım; tezin sorusuyla ilgisiz; tasarım iddiasına kayma riski |
| R3 | Sagers'ın “serial and simultaneous” ifadesi | 1 | Aynı fikir Smith'in kendi sözüyle zaten var: l. 47 («Adı baştan konmuş bir mevsim») |
| R4 | Adamson'ı l. 359'da («Elisabeth'in annesi, Daniel'in yaz geceleri») anmak | 1, 2 | Boşluk yok. Ampirik bulgu bir karakter sözünü doğrulamaz. Savunma notu S6 |
| R5 | “weather-heritage”, “prototype”, “lieux de mémoire” | 3 | Kavram aktarımı |
| R6 | Adamson ve Rapson 2024 (WIREs) | 1, 2 | Kuramsal bir öneri. Smith ve Liu ile Bremer ve Schneider'in zaten yaptığı işi yineler |
| R7 | Hulme'u l. 739–743'e («İklim, bu devamların karşılaştığı şimdinin») ya da l. 1020–1022'ye («Bu sonuçların kuramdan yana bir karşılığı da vardır») eklemek | 1, 2, 4 | İş Dimick s. 3 ile (l. 737 «Dimick bir zamansal uyuşmazlığın»), Smith ve Liu ile ve Clark ile zaten yapılıyor. Yayımlanmış sayfa yok. Yanlış atıf riski (soyutluk formülü) |
| R8 | l. 727 («Dizinin ana bulgusu bu asimetridir») ve l. 1010'da («Dizinin ana bulgusu bu asimetridir») «iklimin adı» diye sözcük düzeltmesi | 1 | Canonical'da adı konmuş bir risk değil; bağlamda anlam açık. Savunma notu S9 |
| R9 | Carlill'i §4.4'e, yani l. 825'e («Bu dizinin gösterdiği şey dardır») eklemek | 1, 2, 3 | Kavram taşır; başka bir roman; boşluk yok |
| R10 | Carlill'i l. 741'e («Bu mesafe, gündelik deneyim ile») ya da l. 743'e («Asimetrinin bir riski de teslim edilmelidir.») eklemek | 1, 2 | Canonical'daki yorum zaten etiketli ve zorunluluk iddia etmiyor; risk cümlesi kaynaklı. Ghosh sorusu l. 727'de («Asimetri olayın yerinde değil») cevaplı |
| R11 | Adamson'ı l. 93'te («Romanların iklimi yaşanan bir güne») de anmak | — (gereksiz) | Giriş'teki kısa anma yeterli. Tek müdahale Sonuç'ta (M1) |
| R12 | §3.4'teki web adayları | 4 | Hiçbiri okunmadı (NOT READ) |

---

## §7. Belirsizlikler ve yapılamayan denetimler

1. **Romanlar yok.** `inputs/novels/` bulunmadığı için roman sayfalarına dayanan her iddia NOT VERIFIED'dır. Buna canonical'ın tarihli şimdileri, *Spring* s. 8 ve *Summer* ss. 25–26 da dahildir. Sagers'ın *Summer* sayfaları (Hamish Hamilton 2020) tezin Penguin 2021 nüshasıyla karşılaştırılmadı.
2. **Hulme.** Yayımlanmış sürüm açılamadı. Sayfalar 63–74 ve “in”/“of” başlık farkı doğrulanmadı. Gönderim sürümünün ifadesinin yayımlanmış metinle aynı olup olmadığı bilinmiyor.
3. **Carlill.** Sayı ataması (106(3), 373–394) doğrulanmadı. Bu raporda verilen sayfalar çevrimiçi ilk yayımın folyolarıdır. *The High House* açılmadı; romandan yapılan alıntılar yalnız Carlill'in alıntısı olarak doğrulandı.
4. **Adamson ve diğerleri.** Alıntılar yazarların seçtiği parçalardır. Çalışma tek olay atfını ayrıca sınamaz. Örneklem küçük ve özgüldür (s. 9).
5. **Web.** Crossref, OpenAlex, DOI çözümleyicisi ve yayıncı sayfaları kapalıydı (§3.1). Künyeler PDF'lerin kendisinden doğrulandı. Web bilgileri yalnız metadata'dır ve arama özetleri makine üretimidir.
6. **Promptta düzeltilmesi gereken noktalar.**
   - Rakip açıklamalar (olağan ihtiyat, siyasal odak, korunaklı yer) §2.5'te değil, l. 93'te («Romanların iklimi yaşanan bir güne») ve l. 1022'de («Tek bir olayı iklime bağlamaktan») geçer.
   - Canonical Sagers'ı promptta sayılan satırlar (143, 203, 240–241) dışında iki yerde daha kullanır: l. 885 («Sagers da *Summer*'ın iki kampı») ve l. 966 («Sagers, dizinin reparatif görünen dönüşünün»).
   - Sagers'ın kitabının iki editörü vardır; «D. Lloyd ve ark.» değil.
   - K4.2'deki «romanın realist kipi bunu gerektirir» rakip açıklaması canonical'da yoktur. En yakın yer l. 741'deki («Bu mesafe, gündelik deneyim ile») yorumdur.
   - Annenin sözü kaynakta büyük harfle başlar (“When we still had seasons”). Canonical'daki alıntı doğrudur.
   - Bilinen kaynaklar listesinde Andeweg ve Janković 2024, Baker 2022 ve Campos 2026 yoktur; üçü de canonical kaynakçasındadır.
7. **Çerçevedeki (05) bir nitelik.** 05 §4 ve §10 Sagers'ı «gerileme ve yenilenme döngüleri» ve «yenilenme okuması» olarak anar. Kaynak onarımı sonunda sorguladığı için (ss. 172, 174) bu nitelik tek yanlıdır. Canonical gövdesi dengeyi zaten kurar: l. 143 («Yakın Smith eleştirisinde») ve l. 966 («Sagers, dizinin reparatif görünen dönüşünün»). Çerçevede değişiklik önerilmez; yalnız bildirilir.
8. **05'teki satır numaraları.** 05 §12'deki numaralar eski canonical sürümüne (`dfe4ef24…`) aittir. Bu raporda yalnız güncel sürümün numaraları kullanıldı.
