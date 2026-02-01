# Asset Universe Rationale

## Purpose of the Project

The purpose of this project is to study how different macro risk channels behave across market regimes, drawdowns, and periods of correlation breakdown.  
Rather than optimising a single strategy, the goal is to understand **risk transmission mechanisms** across asset classes and regions, and how diversification behaves when it is most needed.

This universe is therefore designed to be:
- macro-complete rather than exhaustive
- interpretable rather than optimised
- representative of real-world portfolio risk

---

## Design Principles

The asset universe was constructed using the following principles:

1. **Each asset should represent a distinct risk transmission channel**  
   Assets were selected to capture equity duration, cyclicality, credit stress, volatility, FX-driven risk-off dynamics, and inflation shocks.

2. **Some overlap is intentional**  
   Real portfolios are exposed to multiple channels simultaneously. The goal is not pure factor isolation, but understanding how channels interact.

3. **Liquidity and history matter**  
   Preference is given to liquid indices or ETFs with long histories, ensuring robustness across multiple regimes.

4. **Pragmatic use of proxies**  
   Where direct indices are unavailable or impractical, liquid ETF proxies are used to represent the underlying risk cleanly.

5. **Explainability over precision**  
   Every asset in the universe must have a clear economic role that can be articulated and defended.

---

## Asset Groups and Rationale

### Equities

**SPX Index – US Large Cap Equities**  
Serves as the core global risk asset and benchmark anchor. The S&P 500 is a composite exposure that reflects equity duration, cyclicality, credit conditions, and inflation sensitivity. Its behaviour varies across regimes, making it a useful reference point rather than a pure factor proxy.

**NDX Index – US Growth / Equity Duration**  
Represents long-duration equity exposure. The Nasdaq is dominated by high-margin, asset-light businesses whose valuations are particularly sensitive to real interest rates and discounting of future cash flows.

**RTY Index – US Small Cap / Domestic Cycle & Financing Risk**  
Captures sensitivity to domestic economic conditions, credit availability, and financing constraints. Small caps are often affected by tighter financial conditions even when long-term rates are stable, reflecting a different transmission channel than growth equities.

**Nikkei 225 Index – Japan Equities**  
Provides non-US equity exposure with a distinct monetary and policy backdrop. Japan’s equity market reflects different sector composition, currency dynamics, and historical policy regimes, making it useful for regional diversification analysis.

**FTSE 100 Index – UK Equities**  
Represents a value-tilted, globally exposed equity market with high sensitivity to commodities, financials, and currency movements. Useful for examining equity behaviour under inflationary and rates-driven regimes.

---

### Rates / Duration

**USGG2YR Index – US 2-Year Treasury Yield**  
Represents front-end rates and monetary policy expectations. Useful for distinguishing policy-driven regimes from growth-driven rate moves.

**USGG10YR Index – US 10-Year Treasury Yield**  
Represents long-duration discounting and growth expectations. Central to understanding equity duration sensitivity and equity–rates correlation regimes.

**UK 10-Year Gilt Index**  
Provides non-US duration exposure, allowing analysis of geographic differences in rate sensitivity and diversification benefits.

**German 10-Year Bund (DE10 Index)**  
Represents the Eurozone risk-free rate and a key flight-to-quality asset. German Bunds provide a contrast to US Treasuries and UK Gilts due to different monetary policy dynamics, lower term premia, and a stronger historical role in risk-off environments. Including Bunds allows analysis of geographic differences in duration behaviour and safe-haven dynamics across regimes.

---

### Credit

**LQD US Equity – Investment Grade Credit**  
Represents higher-quality corporate credit and spread behaviour in moderate stress environments. Useful for identifying early signs of tightening financial conditions.

**HYG US Equity – High Yield Credit**  
Captures lower-quality credit risk and risk appetite. High yield credit often deteriorates before or alongside equity drawdowns, making it a key stress transmission indicator.

---

### Volatility

**VIX Index – Equity Volatility**  
Acts as a primary regime indicator. Elevated volatility regimes often coincide with correlation breakdowns, deleveraging, and shifts in asset behaviour.

---

### Foreign Exchange

**DXY Index – US Dollar**  
Represents global liquidity conditions and risk-off dynamics. USD strength is often associated with tightening financial conditions and stress in global assets.

**GBP/USD – Sterling Exchange Rate**  
Reflects domestic UK risk, rate differentials, and currency exposure relevant to a GBP-based investor.

**EUR/USD – Euro Exchange Rate**  
Provides a second major developed-market FX perspective, helping distinguish USD-driven moves from regional effects.

**USD/JPY**  
Represents a classic risk-off and funding-currency dynamic. Yen strength during volatility spikes and deleveraging episodes makes USD/JPY a useful indicator of global risk sentiment and carry unwind.

**AUD/USD**  
Represents pro-cyclical, commodity-linked FX exposure. AUD/USD is sensitive to global growth expectations and risk appetite, often weakening ahead of broader risk-off episodes.

---

### Commodities / Inflation

**CL1 Comdty – Crude Oil**  
Represents energy prices, inflation shocks, and geopolitical risk. Oil plays a central role in inflationary regimes and supply-driven stress events.

**Gold (XAU Curncy or GC1 Comdty)**  
Acts as a defensive asset and store-of-value proxy. Useful for examining behaviour in risk-off, inflationary, and monetary uncertainty regimes.

---

### Defensive Anchor

**IEF US Equity – Intermediate US Treasuries**  
Provides a clean defensive asset proxy and a benchmark for diversification effectiveness. Useful for identifying periods where traditional equity–bond diversification breaks down.

Note us10y - “What discount rate is the market applying to a 10-year cashflow right now?”
IEF - “What happens if I continuously hold intermediate (7-10y) Treasuries and roll them forward through time?” (return on intermediates)
US 10Y yield tells you equity duration sensitivity (a driver) - how sensitive equity valuations are to discount rates
IEF tells you bond duration P&L (an outcome) - price sensitivity of a bond portfolio to yield changes

Bond returns come from:
    Return≈−(Duration×Δy)+Carry+Roll-down+Convexity
The yield curve only gives you Δy. So adding this gives bond duration sensitivity

Should offset losses of equity during equity drawdowns
---

## Explicit Trade-offs and Limitations

- This universe does not attempt to isolate pure academic factors.
- Some assets load on multiple risk channels by design.
- Regional equity indices are used instead of a single global ex-US index to make geographic and policy differences explicit.
- ETF proxies are used where they provide cleaner, more continuous exposure.

---

## What This Universe Is Not

- Not a claim of investability or transaction-level realism
- Not a performance-optimised strategy
- Not a complete factor model

It is a **research framework** for understanding macro risk behaviour across regimes.

---

## Intended Use

This asset universe supports:
- regime identification
- drawdown analysis
- rolling correlation studies
- diversification failure analysis
- narrative-driven interpretation of market behaviour

It provides a stable foundation for subsequent quantitative analysis.

---

## Sample Period and Lookback Choice

The historical sample for this project begins in **2007** and runs to the present.

This start date was chosen deliberately to balance two considerations:

1. **Regime coverage**  
   The primary goal of the project is to study asset behaviour across distinct macro and market regimes, rather than to maximise sample length. Starting in 2007 ensures coverage of:
   - the Global Financial Crisis
   - the post-GFC QE era
   - the taper tantrum and subsequent policy normalisation
   - the COVID shock
   - the post-pandemic inflation and rates regime

   These episodes provide a sufficiently rich set of drawdowns, stress events, and correlation shifts to analyse diversification and regime-dependent behaviour.

2. **Data consistency across asset classes**  
   Several key assets in the universe (credit ETFs, volatility indices, and defensive duration proxies) have reliable and continuous data histories beginning in the mid-to-late 2000s. Starting the sample in 2007 ensures that equity, rates, credit, FX, volatility, and commodity series are available concurrently, avoiding artificial truncation or bias toward more recent periods.

This lookback period is therefore intended to be **economically representative rather than exhaustive**. It prioritises consistency and regime relevance over maximising the number of observations, which is appropriate given the diagnostic and exploratory nature of the analysis.
