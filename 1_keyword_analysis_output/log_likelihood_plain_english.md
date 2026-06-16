# Log-Likelihood Analysis – Plain English

## What is log-likelihood?

Frequency counts alone can be misleading: a word that appears 10 times in a
50-syllabus year looks different from the same count in a 5-syllabus year.
**Log-likelihood (G²)** is a statistical test that asks: *is this word appearing
more (or less) often than we would expect by chance, given the size of each
year’s corpus?*

The test compares each year against the pooled rest of the corpus.
A higher G² score means a stronger, more reliable signal.

| G² threshold | Meaning |
|---|---|
| ≥ 3.84 | Significant at p < 0.05 (1-in-20 chance this is random) |
| ≥ 6.63 | Significant at p < 0.01 (1-in-100) |
| ≥ 10.83 | Significant at p < 0.001 (1-in-1 000) |
| ≥ 15.13 | Significant at p < 0.0001 (1-in-10 000) |

A **`+`** direction means the word is *over-represented* in that year – used more
than the overall corpus rate would predict.  A **`-`** means *under-represented*.

---

## Key findings by year

### 2020

*68 syllabi • 77,164 tokens • 51 words significantly over/under-represented*

**Significantly over-represented** (words used more than expected in this year):

| Word | G² | Significance | Freq this year | Freq elsewhere |
|---|---|---|---|---|
| **change** | 187.4 | p < 0.0001 | 9.29/1k | 4.85/1k |
| **center** | 89.7 | p < 0.0001 | 3.80/1k | 1.86/1k |
| **climate** | 47.9 | p < 0.0001 | 12.31/1k | 9.45/1k |
| **architecture** | 44.7 | p < 0.0001 | 19.63/1k | 16.07/1k |
| **history** | 40.4 | p < 0.0001 | 2.49/1k | 1.40/1k |
| **infrastructure** | 40.1 | p < 0.0001 | 2.85/1k | 1.68/1k |
| **issues** | 39.1 | p < 0.0001 | 2.31/1k | 1.28/1k |
| **nature** | 23.5 | p < 0.0001 | 1.57/1k | 0.91/1k |
| **level** | 22.4 | p < 0.0001 | 1.57/1k | 0.92/1k |
| **cities** | 20.3 | p < 0.0001 | 2.13/1k | 1.38/1k |

**Significantly under-represented** (words used less than expected):

| Word | G² | Significance | Freq this year | Freq elsewhere |
|---|---|---|---|---|
| site | 39.8 | p < 0.0001 | 1.36/1k | 2.50/1k |
| justice | 37.3 | p < 0.0001 | 0.89/1k | 1.82/1k |
| community | 32.4 | p < 0.0001 | 2.40/1k | 3.68/1k |
| performance | 29.9 | p < 0.0001 | 0.70/1k | 1.44/1k |
| analysis | 29.1 | p < 0.0001 | 0.95/1k | 1.76/1k |

---

### 2021

*62 syllabi • 67,645 tokens • 47 words significantly over/under-represented*

**Significantly over-represented** (words used more than expected in this year):

| Word | G² | Significance | Freq this year | Freq elsewhere |
|---|---|---|---|---|
| **energy** | 72.1 | p < 0.0001 | 4.86/1k | 2.73/1k |
| **health** | 31.8 | p < 0.0001 | 1.91/1k | 1.03/1k |
| **sustainable** | 28.5 | p < 0.0001 | 3.39/1k | 2.22/1k |
| **building** | 28.3 | p < 0.0001 | 6.02/1k | 4.42/1k |
| **issues** | 25.5 | p < 0.0001 | 2.20/1k | 1.33/1k |
| **environment** | 21.0 | p < 0.0001 | 3.89/1k | 2.79/1k |
| **current** | 16.9 | p < 0.0001 | 1.52/1k | 0.93/1k |
| **buildings** | 14.0 | p < 0.001 | 2.03/1k | 1.39/1k |
| **performance** | 12.2 | p < 0.001 | 1.74/1k | 1.19/1k |
| **design** | 11.4 | p < 0.001 | 19.32/1k | 17.40/1k |

**Significantly under-represented** (words used less than expected):

| Word | G² | Significance | Freq this year | Freq elsewhere |
|---|---|---|---|---|
| climate | 39.4 | p < 0.0001 | 7.89/1k | 10.48/1k |
| material | 34.4 | p < 0.0001 | 1.97/1k | 3.26/1k |
| infrastructure | 23.6 | p < 0.0001 | 1.21/1k | 2.06/1k |
| public | 20.5 | p < 0.0001 | 1.55/1k | 2.42/1k |
| texas | 20.0 | p < 0.0001 | 0.61/1k | 1.19/1k |

---

### 2022

*31 syllabi • 38,522 tokens • 32 words significantly over/under-represented*

**Significantly over-represented** (words used more than expected in this year):

| Word | G² | Significance | Freq this year | Freq elsewhere |
|---|---|---|---|---|
| **city** | 25.8 | p < 0.0001 | 4.13/1k | 2.60/1k |
| **site** | 24.0 | p < 0.0001 | 3.48/1k | 2.14/1k |
| **nature** | 12.7 | p < 0.001 | 1.64/1k | 0.97/1k |
| **planning** | 11.7 | p < 0.001 | 2.75/1k | 1.89/1k |
| **projects** | 10.2 | p < 0.01 | 2.93/1k | 2.09/1k |
| **strategies** | 8.8 | p < 0.01 | 2.34/1k | 1.64/1k |
| **water** | 7.5 | p < 0.01 | 2.13/1k | 1.52/1k |
| **architect** | 7.2 | p < 0.01 | 1.84/1k | 1.29/1k |
| **scale** | 7.2 | p < 0.01 | 1.92/1k | 1.35/1k |
| **issues** | 6.9 | p < 0.01 | 2.00/1k | 1.43/1k |

**Significantly under-represented** (words used less than expected):

| Word | G² | Significance | Freq this year | Freq elsewhere |
|---|---|---|---|---|
| design | 42.2 | p < 0.0001 | 13.71/1k | 18.19/1k |
| climate | 32.1 | p < 0.0001 | 7.40/1k | 10.31/1k |
| resilience | 27.4 | p < 0.0001 | 0.42/1k | 1.26/1k |
| module | 19.6 | p < 0.0001 | 0.47/1k | 1.17/1k |
| texas | 18.8 | p < 0.0001 | 0.47/1k | 1.15/1k |

---

### 2023

*41 syllabi • 43,289 tokens • 22 words significantly over/under-represented*

**Significantly over-represented** (words used more than expected in this year):

| Word | G² | Significance | Freq this year | Freq elsewhere |
|---|---|---|---|---|
| **module** | 24.7 | p < 0.0001 | 1.92/1k | 1.00/1k |
| **texas** | 13.2 | p < 0.001 | 1.66/1k | 1.01/1k |
| **american** | 11.7 | p < 0.001 | 2.19/1k | 1.47/1k |
| **local** | 8.7 | p < 0.01 | 1.99/1k | 1.38/1k |
| **data** | 6.9 | p < 0.01 | 1.62/1k | 1.13/1k |
| **world** | 6.8 | p < 0.01 | 2.26/1k | 1.68/1k |
| **methods** | 6.7 | p < 0.01 | 1.57/1k | 1.10/1k |
| **study** | 6.0 | p < 0.05 | 2.03/1k | 1.52/1k |

**Significantly under-represented** (words used less than expected):

| Word | G² | Significance | Freq this year | Freq elsewhere |
|---|---|---|---|---|
| architectural | 34.8 | p < 0.0001 | 3.21/1k | 5.21/1k |
| challenges | 28.8 | p < 0.0001 | 0.60/1k | 1.53/1k |
| strategies | 18.3 | p < 0.0001 | 0.97/1k | 1.81/1k |
| carbon | 18.2 | p < 0.0001 | 0.99/1k | 1.84/1k |
| climate | 16.7 | p < 0.0001 | 8.22/1k | 10.25/1k |

---

### 2024

*43 syllabi • 47,798 tokens • 46 words significantly over/under-represented*

**Significantly over-represented** (words used more than expected in this year):

| Word | G² | Significance | Freq this year | Freq elsewhere |
|---|---|---|---|---|
| **space** | 36.1 | p < 0.0001 | 2.45/1k | 1.25/1k |
| **future** | 26.0 | p < 0.0001 | 3.35/1k | 2.09/1k |
| **landscape** | 25.3 | p < 0.0001 | 3.01/1k | 1.85/1k |
| **module** | 14.2 | p < 0.001 | 1.67/1k | 1.02/1k |
| **world** | 12.3 | p < 0.001 | 2.41/1k | 1.66/1k |
| **justice** | 11.9 | p < 0.001 | 2.26/1k | 1.55/1k |
| **context** | 10.8 | p < 0.001 | 1.80/1k | 1.19/1k |
| **state** | 8.4 | p < 0.01 | 2.07/1k | 1.49/1k |
| **approach** | 7.0 | p < 0.01 | 1.59/1k | 1.13/1k |
| **cultural** | 6.5 | p < 0.05 | 2.28/1k | 1.73/1k |

**Significantly under-represented** (words used less than expected):

| Word | G² | Significance | Freq this year | Freq elsewhere |
|---|---|---|---|---|
| performance | 39.7 | p < 0.0001 | 0.44/1k | 1.41/1k |
| carbon | 34.3 | p < 0.0001 | 0.80/1k | 1.88/1k |
| sustainability | 22.7 | p < 0.0001 | 1.13/1k | 2.09/1k |
| data | 15.6 | p < 0.0001 | 0.65/1k | 1.26/1k |
| communities | 12.2 | p < 0.001 | 1.00/1k | 1.64/1k |

---

### 2025

*36 syllabi • 37,383 tokens • 35 words significantly over/under-represented*

**Significantly over-represented** (words used more than expected in this year):

| Word | G² | Significance | Freq this year | Freq elsewhere |
|---|---|---|---|---|
| **resilience** | 37.4 | p < 0.0001 | 2.33/1k | 1.05/1k |
| **urban** | 36.7 | p < 0.0001 | 7.30/1k | 4.82/1k |
| **justice** | 23.5 | p < 0.0001 | 2.67/1k | 1.52/1k |
| **texas** | 17.4 | p < 0.0001 | 1.82/1k | 1.01/1k |
| **social** | 14.3 | p < 0.001 | 4.52/1k | 3.27/1k |
| **challenges** | 14.2 | p < 0.001 | 2.17/1k | 1.34/1k |
| **community** | 13.3 | p < 0.001 | 4.52/1k | 3.31/1k |
| **sustainable** | 10.9 | p < 0.001 | 3.26/1k | 2.34/1k |
| **sustainability** | 10.9 | p < 0.001 | 2.73/1k | 1.89/1k |
| **approach** | 9.5 | p < 0.01 | 1.74/1k | 1.12/1k |

**Significantly under-represented** (words used less than expected):

| Word | G² | Significance | Freq this year | Freq elsewhere |
|---|---|---|---|---|
| center | 28.2 | p < 0.0001 | 1.12/1k | 2.37/1k |
| building | 28.0 | p < 0.0001 | 3.02/1k | 4.88/1k |
| american | 24.1 | p < 0.0001 | 0.70/1k | 1.64/1k |
| architecture | 23.7 | p < 0.0001 | 13.78/1k | 17.11/1k |
| systems | 20.9 | p < 0.0001 | 1.95/1k | 3.26/1k |

---

### 2026

*66 syllabi • 72,440 tokens • 63 words significantly over/under-represented*

**Significantly over-represented** (words used more than expected in this year):

| Word | G² | Significance | Freq this year | Freq elsewhere |
|---|---|---|---|---|
| **architectural** | 203.8 | p < 0.0001 | 8.61/1k | 4.14/1k |
| **material** | 163.5 | p < 0.0001 | 5.59/1k | 2.44/1k |
| **performance** | 98.3 | p < 0.0001 | 2.60/1k | 0.98/1k |
| **systems** | 71.3 | p < 0.0001 | 4.80/1k | 2.74/1k |
| **environmental** | 54.9 | p < 0.0001 | 7.48/1k | 5.11/1k |
| **design** | 53.8 | p < 0.0001 | 21.08/1k | 16.96/1k |
| **spatial** | 47.7 | p < 0.0001 | 2.13/1k | 1.05/1k |
| **public** | 43.8 | p < 0.0001 | 3.38/1k | 2.01/1k |
| **practice** | 40.5 | p < 0.0001 | 3.30/1k | 1.99/1k |
| **community** | 33.9 | p < 0.0001 | 4.61/1k | 3.15/1k |

**Significantly under-represented** (words used less than expected):

| Word | G² | Significance | Freq this year | Freq elsewhere |
|---|---|---|---|---|
| issues | 91.0 | p < 0.0001 | 0.41/1k | 1.74/1k |
| change | 54.8 | p < 0.0001 | 3.95/1k | 6.16/1k |
| future | 44.9 | p < 0.0001 | 1.26/1k | 2.48/1k |
| american | 41.1 | p < 0.0001 | 0.77/1k | 1.73/1k |
| nature | 39.3 | p < 0.0001 | 0.43/1k | 1.18/1k |

---

## Summary: what changed between 2020 and 2026?

Words that became **significantly over-represented** in 2026 but were not in 2020:

- **architectural** — G² = 203.8 (p < 0.0001), 8.61 vs 4.14 per 1k words (+108% relative to rest of corpus)
- **material** — G² = 163.5 (p < 0.0001), 5.59 vs 2.44 per 1k words (+129% relative to rest of corpus)
- **performance** — G² = 98.3 (p < 0.0001), 2.60 vs 0.98 per 1k words (+164% relative to rest of corpus)
- **systems** — G² = 71.3 (p < 0.0001), 4.80 vs 2.74 per 1k words (+75% relative to rest of corpus)
- **environmental** — G² = 54.9 (p < 0.0001), 7.48 vs 5.11 per 1k words (+46% relative to rest of corpus)
- **design** — G² = 53.8 (p < 0.0001), 21.08 vs 16.96 per 1k words (+24% relative to rest of corpus)
- **spatial** — G² = 47.7 (p < 0.0001), 2.13 vs 1.05 per 1k words (+103% relative to rest of corpus)
- **public** — G² = 43.8 (p < 0.0001), 3.38 vs 2.01 per 1k words (+68% relative to rest of corpus)
- **practice** — G² = 40.5 (p < 0.0001), 3.30 vs 1.99 per 1k words (+66% relative to rest of corpus)
- **community** — G² = 33.9 (p < 0.0001), 4.61 vs 3.15 per 1k words (+46% relative to rest of corpus)
- **conditions** — G² = 28.6 (p < 0.0001), 1.79 vs 1.00 per 1k words (+79% relative to rest of corpus)
- **ecological** — G² = 25.6 (p < 0.0001), 2.33 vs 1.45 per 1k words (+61% relative to rest of corpus)
- **analysis** — G² = 21.7 (p < 0.0001), 2.25 vs 1.45 per 1k words (+56% relative to rest of corpus)
- **strategies** — G² = 20.1 (p < 0.0001), 2.36 vs 1.56 per 1k words (+51% relative to rest of corpus)
- **cultural** — G² = 19.7 (p < 0.0001), 2.46 vs 1.65 per 1k words (+49% relative to rest of corpus)

Words that were **significantly over-represented** in 2020 but are no longer in 2026:

- **change** — G² = 187.4 (p < 0.0001), 9.29 per 1k in 2020
- **center** — G² = 89.7 (p < 0.0001), 3.80 per 1k in 2020
- **architecture** — G² = 44.7 (p < 0.0001), 19.63 per 1k in 2020
- **history** — G² = 40.4 (p < 0.0001), 2.49 per 1k in 2020
- **issues** — G² = 39.1 (p < 0.0001), 2.31 per 1k in 2020
- **nature** — G² = 23.5 (p < 0.0001), 1.57 per 1k in 2020
- **level** — G² = 22.4 (p < 0.0001), 1.57 per 1k in 2020
- **cities** — G² = 20.3 (p < 0.0001), 2.13 per 1k in 2020
- **global** — G² = 19.2 (p < 0.0001), 2.63 per 1k in 2020
- **landscape** — G² = 18.5 (p < 0.0001), 2.63 per 1k in 2020
- **society** — G² = 15.4 (p < 0.0001), 2.38 per 1k in 2020
- **art** — G² = 14.7 (p < 0.001), 1.49 per 1k in 2020
- **american** — G² = 12.9 (p < 0.001), 2.02 per 1k in 2020
- **city** — G² = 12.5 (p < 0.001), 3.37 per 1k in 2020
- **environment** — G² = 11.4 (p < 0.001), 3.59 per 1k in 2020

---

## How to read this

- **G²** is the raw score. Bigger = more surprising departure from the expected rate.
- **Freq this year / Freq elsewhere** shows the actual usage rates so you can see
  the direction and magnitude.
- Words that appear consistently across all years will have low G² even if common,
  because they are not *distinctive* to any year.
- Words appearing in very few syllabi can score high G² by coincidence; always
  cross-check with raw counts.
- The test is run word-by-word (no multiple-comparison correction); treat borderline
  p < 0.05 results with appropriate caution.