# MASTER PROJECT REPORT

## Abstract
This study presents a full-stack stylometric pipeline for Digital Humanities that separates journal-driven structure from authorial style in a corpus of 125 academic articles. We first compress high-dimensional section data (7,621+ structural features) into five canonical categories (INTRO, BODY, DISCUSSION, CONCLUSION, OTHER), yielding substantial dimensionality reduction while preserving interpretable rhetorical scaffolding. Metadata enrichment combines TEI extraction with DOI/CrossRef fallback and a smart validation gate focused on analysis-critical fields, producing 98.4% journal coverage. At article level, we estimate journal effects with one-way ANOVA and variance partitioning (eta-squared). Results show strong venue effects for selected features, led by citations_per_1k (η² = 0.7195), followed by sent_lt_12_pct (η² = 0.1982) and median_sentence_len (η² = 0.1471). To recover author-level signal, we compute residual style space by subtracting journal means from each metric. We then test structure-style independence via PCA on canonical section features and residualized style metrics. The correlation between PC1_structure and PC1_style is weak and non-significant (r = -0.0523, p = 0.5654), supporting a decoupling hypothesis: journals impose macro-structural templates, but authorial voice persists in residual stylistic variation.

## (A) Article-Level Stylometry
Aşağıda 5 ana stilistik metrik için ortalama ve dağılım özeti verilmiştir.

| Metric | Mean | Std | P25 | Median | P75 |
| --- | --- | --- | --- | --- | --- |
| mtld | 100.917 | 20.2746 | 87.6669 | 99.2627 | 111.52 |
| avg_sentence_len | 18.1572 | 2.53417 | 16.5482 | 17.9514 | 19.75 |
| subordination_per_1k_words | 35.4874 | 5.2636 | 32.2421 | 34.9983 | 39.0529 |
| passive_sent_ratio | 0.149626 | 0.0429141 | 0.117988 | 0.145119 | 0.176796 |
| nominalisations_per_1k_words | 45.7562 | 10.9601 | 38.3437 | 44.9572 | 52.1135 |

## (B) Section Structure
7621+ geniş feature uzayından 5 canonical kategoriye geçiş özeti:

- Original wide columns: 7621
- Canonical wide columns: 11
- Compression (feature reduction): 99.87%
- Mapping success (section instances mapped to INTRO/BODY/DISCUSSION/CONCLUSION): 21.35%
- Mapping success (unique headings mapped to INTRO/BODY/DISCUSSION/CONCLUSION): 3.36%

Canonical kategori bazında ortalama kelime sayısı:

| canonical | avg_word_count |
| --- | --- |
| INTRO | 966.453 |
| BODY | 2086 |
| DISCUSSION | 1253.14 |
| CONCLUSION | 504 |
| OTHER | 1152.78 |

## (C) Metadata Enrichment
- Smart Gate Success: 98.4%
- DOI present: 125/125
- CrossRef enrichment success (journal or publisher filled): 123/125
- Journal Coverage (doi + journal available): 98.40%
- Smart Gate (%100 şartı değil, yeterli kapsam): GEÇTİ

## (D) Residual Space
Residual tanımı: `Residual = Value - GroupMean_journal`. Bu işlem dergi şablonu etkisini sıfırlar ve stilistik kalıntıyı ortaya çıkarır.

Özet karşılaştırma (orijinal vs residual):

| Metric | OriginalMean | ResidualMean | OriginalStd | ResidualStd |
| --- | --- | --- | --- | --- |
| citations_per_1k | 2.60373 | -7.22096e-18 | 2.64991 | 1.40347 |
| sent_lt_12_pct | 0.423533 | -3.38483e-18 | 0.0826175 | 0.073977 |
| median_sentence_len | 15.1789 | 3.61048e-16 | 3.76575 | 3.47773 |
| integral_ratio | 0.102175 | 2.82069e-18 | 0.171876 | 0.160096 |
| nominalisations_per_1k_words | 45.7014 | -1.24201e-15 | 11.0376 | 10.2964 |
| pos_adj_ratio | 0.0690028 | 3.15917e-18 | 0.012411 | 0.0116513 |

## (E) Variance Partition (η²)
**En yüksek η² (dergi etkili) - Top 3**
- Neye bakıldı: `citations_per_1k` | Ne bulundu: η²=0.7195, p=1.166e-31
- Neye bakıldı: `sent_lt_12_pct` | Ne bulundu: η²=0.1982, p=2.77e-05
- Neye bakıldı: `median_sentence_len` | Ne bulundu: η²=0.1471, p=0.0008096

**En düşük η² (yazar etkili) - Bottom 3**
- Neye bakıldı: `sent_gt_40_pct` | Ne bulundu: η²=0.0256, p=0.5436
- Neye bakıldı: `agentless_passive_ratio_of_passives` | Ne bulundu: η²=0.0285, p=0.4876
- Neye bakıldı: `sd_sentence_len` | Ne bulundu: η²=0.0329, p=0.4088

Note: Values are normalized (0-1) using $SS_{between}/SS_{total}$.

## (F) Decoupling
- Hesaplanan korelasyon: r=-0.0523, p=0.5654
- Yorum: Korelasyon sıfıra çok yakın ve istatistiksel olarak anlamsız (p>0.05).
- Sonuç: Yapısal organizasyon (section scaffolding) ile residual stil uzayı farklı eksenlerde çalışıyor; yani dergi formatı yapıyı kısıtlasa da yazarın stilistik sinyali residual uzayda bağımsız kalabiliyor.
