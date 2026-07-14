# Keyword Analysis

## Keyword frequency change

*(frequency per 1 000 words; slope = OLS trend per year across 2020–2026)*

| Keyword | Aliases | Freq 2020 | Freq 2026 | Δ freq/1k | Slope/yr |
|---------|---------|------|------|------|------|
| climate | — | 13.58 | 12.29 | -1.29 | +0.214 |
| change | — | 10.25 | 4.32 | -5.93 | -0.696 |
| environmental | environment | 9.53 | 10.34 | +0.81 | +0.025 |
| systems | system | 3.89 | 6.35 | +2.47 | +0.189 |
| energy | energetic | 3.34 | 2.90 | -0.45 | -0.216 |
| material | materials | 4.59 | 8.83 | +4.24 | +0.480 |
| sustainable | sustainability | 5.06 | 3.83 | -1.23 | -0.089 |
| future | — | 2.13 | 1.37 | -0.76 | -0.039 |
| global | — | 2.90 | 1.60 | -1.30 | -0.060 |
| infrastructure | infrastructural | 4.12 | 3.02 | -1.10 | -0.069 |
| society | societal | 2.89 | 1.75 | -1.14 | -0.059 |
| world | — | 2.34 | 1.09 | -1.26 | -0.122 |
| carbon | — | 2.24 | 2.29 | +0.05 | -0.017 |
| critical | — | 2.22 | 2.20 | -0.01 | +0.048 |
| human | — | 1.79 | 1.45 | -0.34 | -0.061 |
| justice | — | 0.99 | 1.71 | +0.72 | +0.184 |
| ecological | ecology | 2.47 | 3.28 | +0.80 | +0.158 |
| community | communities | 4.34 | 6.78 | +2.43 | +0.380 |
| issue | issues | 3.03 | 0.57 | -2.46 | -0.412 |
| impact | — | 1.84 | 1.42 | -0.42 | -0.028 |
| experience | — | 1.21 | 2.05 | +0.84 | +0.141 |
| science | sciences | 2.43 | 1.45 | -0.98 | -0.088 |
| resource | resources | 2.03 | 1.55 | -0.47 | -0.050 |
| data | — | 1.16 | 1.58 | +0.43 | +0.042 |
| health | — | 0.84 | 1.43 | +0.59 | -0.004 |
| resilience | resiliency | 1.56 | 1.78 | +0.22 | +0.191 |
| economic | economy | 1.84 | 1.13 | -0.71 | -0.057 |
| texas | — | 1.06 | 1.25 | +0.20 | +0.148 |
| nature | natural | 3.30 | 1.13 | -2.17 | -0.371 |
| current | — | 1.24 | 0.72 | -0.52 | -0.094 |

## Keyword-2 frequency change

*(frequency per 1 000 words; slope = OLS trend per year across 2020–2026)*

| Keyword | Aliases | Freq 2020 | Freq 2026 | Δ freq/1k | Slope/yr |
|---------|---------|------|------|------|------|
| indigenous | — | 0.37 | 1.07 | +0.70 | +0.078 |
| vernacular | — | 0.14 | 1.43 | +1.29 | +0.203 |
| material | materials | 4.59 | 8.83 | +4.24 | +0.480 |
| critical | — | 2.22 | 2.20 | -0.01 | +0.048 |
| system | systems | 3.89 | 6.35 | +2.47 | +0.189 |
| space | spatial, spatializing, spatialize | 2.42 | 3.77 | +1.36 | +0.237 |
| theory | — | 1.03 | 1.33 | +0.30 | +0.023 |
| representation | — | 0.60 | 0.74 | +0.14 | +0.046 |
| human | — | 1.79 | 1.45 | -0.34 | -0.061 |
| non-human | — | 0.00 | 0.00 | +0.00 | +0.000 |
| ethics | ethical | 0.64 | 1.58 | +0.94 | +0.094 |
| technology | technical, tech, technological | 3.52 | 4.18 | +0.67 | +0.021 |
| culture | cultural | 2.66 | 3.15 | +0.50 | +0.077 |
| local | — | 1.40 | 1.72 | +0.32 | +0.074 |
| global | — | 2.90 | 1.60 | -1.30 | -0.060 |
| knowledge | — | 1.89 | 1.60 | -0.29 | -0.017 |

## Output files

| File | Description |
|------|-------------|
| `keywords-1/keywords_heatmap.png` | Keyword frequency per 1,000 words, by year |
| `keywords-1/keywords_heatmap_normalised.png` | Row-normalised heatmap (relative change) |
| `keywords-1/keywords_ridgeline.png` | Ridgeline chart of keyword frequency over time |
| `keywords-1/keywords_bar_chart.png` | Grouped bar chart of keyword frequency |
| `keywords-2/keywords_heatmap.png` | Keyword-2 frequency per 1,000 words, by year |
| `keywords-2/keywords_heatmap_normalised.png` | Row-normalised heatmap for keyword-2 |
| `keywords-2/keywords_ridgeline.png` | Ridgeline chart of keyword-2 frequency over time |
| `keywords-2/keywords_bar_chart.png` | Grouped bar chart of keyword-2 frequency |

## Notes

- Both stopword lists applied in step 0 (PDF preprocessing).
- Frequencies normalised per 1 000 words to account for varying corpus sizes per year.
- Keyword groups loaded from `data/keywords.txt` and `data/keywords-2.txt`; comma-separated aliases on each line
  are counted together and displayed under the first alias.
- Set `DO_TOP_N_ANALYSIS = True` in the script to also produce top-100 word charts
  and log-likelihood analysis.