# LaTeX Context: main.tex & bibliography.tex

## main.tex (68 lines)
**Document class**: report, 12pt, a4paper
**Geometry**: top=3cm, bottom=3cm, left=3.5cm, right=2.5cm

### Packages
- **Text**: inputenc (utf8), fontenc (T1), lmodern, microtype (no expansion), babel (american)
- **Math**: amsmath, amsthm, amssymb, amsfonts, mathtools
- **Algorithms**: algorithm, algorithmic
- **Colors**: xcolor
- **Graphics/Tables**: graphicx, subcaption, float, booktabs, tabularx, multirow, caption (font=small)
- **Links**: hyperref (colorlinks, linkcolor=black, urlcolor=blue)

### Theorem Environments
- `remark` — numbered per section
- `theorem` — numbered per chapter
- `proposition` — shares theorem counter
- `lemma` — shares theorem counter
- `definition` — shares theorem counter

### Hyphenation
Custom: gra-dual-ly, reaching, Investment, financial, announcement, reframed, appro-priate, including, volatili-ty, dependent, rever-ting, re-pre-sent, associated, entered, glo-bal-ly, ex-chan-ge, immedia-tely, reflec-ting, assumptions, markets, Heston, sy-ste-ma-ti-cally, deribit

### Document Structure
```
\tableofcontents
\include{Chapter_1}
\include{Chapter_2}
\include{Chapter_3}
%\include{Conclusions}  ← commented out
\include{bibliography}
```

### Notes
- pdftitle currently "Boh" — placeholder
- Commented methodological note about BTC-PERPETUAL on Deribit for internal consistency

---

## bibliography.tex (84 lines, 30 references)

### Reference List (bibkey → short description)
| Key | Author(s) | Year | Topic |
|-----|-----------|------|-------|
| bis2018 | BIS | 2018 | Crypto overview, Annual Report Ch5 |
| elad2025 | Elad, Kinder | 2025 | Crypto derivatives market stats |
| mandelbrot1968 | Mandelbrot, Van Ness | 1968 | fBm definition, SIAM Review |
| alexander2023 | Alexander, Chen, Imeraj | 2023 | Crypto quanto & inverse options, Math Finance |
| bayer2016 | Bayer, Friz, Gatheral | 2016 | Pricing under rough vol, Quant Finance |
| coinglass2025 | Coinglass | 2025 | Annual report (ETF data) |
| coinmarketcap2026 | CoinMarketCap | 2026 | Market data |
| nualart2006 | Nualart | 2006 | Malliavin Calculus, Springer |
| yermack2014 | Yermack | 2014 | Is Bitcoin real currency?, NYU |
| deribit_mark | Deribit | — | Mark price documentation |
| black1973 | Black, Scholes | 1973 | Option pricing, JPE |
| biagini2008 | Biagini, Hu, Øksendal, Zhang | 2008 | Stochastic calc for fBm, Springer |
| decreusefond1999 | Decreusefond, Üstünel | 1999 | Stochastic analysis of fBm |
| gatheral2018 | Gatheral, Jaisson, Rosenbaum | 2018 | Volatility is rough, Quant Finance |
| gatheral2011 | Gatheral | 2011 | Volatility Surface book, Wiley |
| forde2009 | Forde, Jacquier | 2009 | Small-time asymptotics Heston |
| bennedsen2017 | Bennedsen, Lunde, Pakkanen | 2017 | Hybrid scheme BSS processes, Finance & Stochastics |
| borri2025 | Borri, Liu, Tsyvinski, Wu | 2025 | Crypto as investable asset class |
| mccrickerd2018 | McCrickerd, Pakkanen | 2018 | Turbocharging MC rBergomi, Quant Finance |
| merton1973 | Merton | 1973 | Rational option pricing, Bell Journal |
| nakamoto2008 | Nakamoto | 2008 | Bitcoin whitepaper |
| heston1993 | Heston | 1993 | Closed-form stoch vol, RFS |
| takaishi2020 | Takaishi | 2020 | Rough vol of Bitcoin, Finance Res Letters |
| takaishi2021 | Takaishi | 2021 | Time-varying asymmetric vol BTC, PLoS ONE |
| takaishi2025 | Takaishi | 2025 | Multifractality BTC vol, Finance Res Letters |
| lucic2024 | Lucic, Sepp | 2024 | Valuation/hedging crypto inverse options, Quant Finance |
| wainwright2024 | Wainwright | 2024 | BTC volatility, Fidelity Digital Assets |

### Missing from bibliography (referenced in Ch3 but not in .bib)
- `bennedsen2022` — cited in Ch3 §3.3.1 and §3.4.3
- `cao2023` — cited in Ch3 §3.4.1
- `hoang2022` — cited in Ch3 §3.4.1
