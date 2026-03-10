# ULTIMATE PROJECT AUDIT

This audit consolidates the full analytics lifecycle with methodological and inferential checkpoints.

### (A) Stylometry
**Neye Bakildi**
- MTLD, sentence-length, and syntax-oriented article-level metrics.

**Teknik Veri / Tablo**
| metric | mean | std | min | max |
| --- | --- | --- | --- | --- |
| mtld | 100.917 | 20.2746 | 58.1257 | 161.752 |
| avg_sentence_len | 18.1572 | 2.53417 | 10.906 | 25.6208 |
| median_sentence_len | 15.128 | 3.76524 | 4 | 23 |
| sd_sentence_len | 15.6475 | 2.35735 | 11.323 | 23.1659 |
| subordination_per_1k_words | 35.4874 | 5.2636 | 17.3457 | 48.7551 |
| nominalisations_per_1k_words | 45.7562 | 10.9601 | 21.7358 | 79.723 |
| passive_sent_ratio | 0.149626 | 0.0429141 | 0.0479042 | 0.27193 |

**Akademik Cikarim**
- Lexical diversity and syntactic density vary meaningfully across the corpus; this supports the need for both journal-level control and residual style modeling.

### (B) Structure
**Neye Bakildi**
- Missingness after reducing 7,621 structural columns into 5 canonical section categories.

**Teknik Veri / Tablo**
- Original structural feature count: 7621
- Canonical structural feature count: 10
| category | missingness_ratio | missingness_pct |
| --- | --- | --- |
| BODY | 0.976 | 97.6 |
| DISCUSSION | 0.96 | 96 |
| INTRO | 0.416 | 41.6 |
| CONCLUSION | 0.272 | 27.2 |
| OTHER | 0.008 | 0.8 |

**Akademik Cikarim**
- Highest uncertainty appears in `BODY`; this indicates that some rhetorical units are less consistently recoverable across articles (especially discussion-like segments).

### (C) Metadata
**Neye Bakildi**
- Journal coverage and DOI-to-year fallback performance.

**Teknik Veri / Tablo**
- DOI available: 125/125
- Journal coverage: 98.40% (123/125)
- DOI year fallback success: 100.00% (125/125 rows needing fallback)

**Akademik Cikarim**
- Metadata quality is high enough for journal-controlled analysis. DOI-derived year fallback reduces dependency on unstable external API responses.

### (D) Residuals
**Neye Bakildi**
- Difference between raw metrics and journal-demeaned residual metrics.

**Teknik Veri / Tablo**
Residual formula: `Residual = Value - GroupMean_journal`
| metric | orig_mean | resid_mean | orig_std | resid_std |
| --- | --- | --- | --- | --- |
| citations_per_1k | 2.60373 | -7.22096e-18 | 2.64991 | 1.40347 |
| sent_lt_12_pct | 0.423533 | -3.38483e-18 | 0.0826175 | 0.073977 |
| median_sentence_len | 15.1789 | 3.61048e-16 | 3.76575 | 3.47773 |
| integral_ratio | 0.102175 | 2.82069e-18 | 0.171876 | 0.160096 |
| nominalisations_per_1k_words | 45.7014 | -1.24201e-15 | 11.0376 | 10.2964 |
| pos_adj_ratio | 0.0690028 | 3.15917e-18 | 0.012411 | 0.0116513 |
| pos_verb_ratio | 0.103483 | -1.46676e-18 | 0.0131318 | 0.0124033 |
| mtld | 100.779 | -2.65731e-15 | 20.4104 | 19.3262 |

**Akademik Cikarim**
- Residualization removes venue template effects and isolates cleaner author-level stylistic signal (a closer proxy for 'authorial voice').

### (E) Variance Partition (eta-squared)
**Neye Bakildi**
- Journal effect strength by metric using eta-squared from ANOVA.

**Teknik Veri / Tablo**
| class | metric | eta_sq | p_value |
| --- | --- | --- | --- |
| Journal-Driven | citations_per_1k | 0.719492 | 1.16609e-31 |
| Journal-Driven | sent_lt_12_pct | 0.198231 | 2.77049e-05 |
| Journal-Driven | median_sentence_len | 0.14712 | 0.0008096 |
| Author-Driven | sent_gt_40_pct | 0.0255955 | 0.543643 |
| Author-Driven | agentless_passive_ratio_of_passives | 0.0284679 | 0.487576 |
| Author-Driven | sd_sentence_len | 0.032888 | 0.408823 |
Note: Values are normalized in [0,1] via SS_between / SS_total.

**Akademik Cikarim**
- High eta-squared metrics are strongly venue-shaped; low eta-squared metrics preserve more writer-specific variance.

### (F) Decoupling
**Neye Bakildi**
- Pearson correlation between structure PC1 and residual style PC1.

**Teknik Veri / Tablo**
| metric | value |
| --- | ---: |
| Pearson r | -0.0523 |
| p-value | 0.5654 |

**Akademik Cikarim**
- "Iskelet (Structure) neden Ruhu (Style) yonetemiyor?": Because the observed coupling is near-zero and statistically non-significant, structural templates operate as editorial scaffolds without tightly determining residual stylistic expression.
