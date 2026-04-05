# src/config/dates.py
"""
Shared date definitions for the thesis pipeline.

NOTE: populate_tables.py has its OWN date groupings (EVENT_GROUPS,
BASELINE_GROUPS) with different structure and labels for LaTeX output.
Those are intentionally separate and must NOT be replaced by these.
"""

# 7 major crypto events, each with day-before / day-of / day-after
EVENT_DATES = {
    # -- LUNA/UST collapse (May 9, 2022) --
    "20220508": "luna_pre",
    "20220509": "luna_event",
    "20220510": "luna_post",
    # -- FTX bankruptcy (Nov 8, 2022) --
    "20221107": "ftx_pre",
    "20221108": "ftx_event",
    "20221109": "ftx_post",
    # -- SVB / banking crisis (Mar 10, 2023) --
    "20230309": "svb_pre",
    "20230310": "svb_event",
    "20230311": "svb_post",
    # -- Bitcoin Spot ETF approval (Jan 10, 2024) --
    "20240109": "etf_pre",
    "20240110": "etf_event",
    "20240111": "etf_post",
    # -- Fourth Bitcoin halving (Apr 20, 2024) --
    "20240419": "halving_pre",
    "20240420": "halving_event",
    "20240421": "halving_post",
    # -- BTC breaks $100k (Dec 5, 2024) --
    "20241204": "btc100k_pre",
    "20241205": "btc100k_event",
    "20241206": "btc100k_post",
    # -- Trump inauguration / crypto EO (Jan 20, 2025) --
    "20250119": "trump_pre",
    "20250120": "trump_event",
    "20250121": "trump_post",
}

# Low-IV baseline (ATM IV < 45%): calm summer 2023, post-ETF, mid-2024
LOW_IV_DATES = {
    "20230812": "low_iv_summer_2023",    # calm summer 2023 (ATM ~26%)
    "20230930": "low_iv_pre_rally",      # pre Q4 2023 rally (ATM ~27%)
    "20240210": "low_iv_post_etf",       # 1 month after ETF approval (ATM ~41%)
}

# Medium-IV baseline (ATM IV 45-70%): crypto winter, autumn 2024, 2025
MEDIUM_IV_DATES = {
    "20221217": "mid_iv_crypto_winter",  # deep crypto winter, BTC ~$17k (ATM ~51%)
    "20240914": "mid_iv_autumn_2024",    # autumn 2024 cooldown (ATM ~48%)
    "20250203": "mid_iv_bull_run_2025",  # Feb 2025 bull run (ATM ~58%)
}

# High-IV baseline (ATM IV > 70%): post-LUNA, bear market, post-FTX
HIGH_IV_DATES = {
    "20220618": "high_iv_post_luna",     # 5 weeks after Luna collapse (ATM ~117%)
    "20220702": "high_iv_bear_2022",     # bear market summer 2022 (ATM ~81%)
    "20221116": "high_iv_post_ftx",      # 1 week after FTX collapse (ATM ~80%)
}

# Combined baseline (all IV regimes)
BASELINE_DATES = {**LOW_IV_DATES, **MEDIUM_IV_DATES, **HIGH_IV_DATES}

# Combined (deduplicated, sorted)
ALL_DATES = {**EVENT_DATES, **BASELINE_DATES}
SNAPSHOT_DATE_STRINGS = sorted(ALL_DATES.keys())
