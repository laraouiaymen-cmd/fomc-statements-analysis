### Title  
**FOMC Statement Reaction: Do Post-Meeting Statements Predict Next-Day S&P 500 Direction?**

### Motivation  
FOMC post-meeting statements are among the most closely watched macroeconomic communications. Markets react within minutes, yet the text may still contain residual information about near-term direction if language systematically differs before “up” versus “down” days. This project tests a narrow, falsifiable question: **do FOMC statements contain any signal about the next trading day direction of the S&P 500?** The goal is not to build a trading strategy, but to evaluate whether modern text representations can extract a weak directional pattern beyond randomness.

### Research Question  
Given the text of an FOMC post-meeting statement at date *t*, can we predict whether the S&P 500 closes up or down on the next trading day (*t+1*)?

### Data  
- FOMC post-meeting statements (1994–2025) downloaded from Federal Reserve webpages; missing statements are detected and fetched automatically.  
- Market data from `yfinance` to compute next-day close-to-close returns and directional labels.  
The dataset is split chronologically (80/20) to avoid look-ahead bias.

### Methodology  
Each statement is converted into a fixed-length vector using four embedders: **TF-IDF**, **BERT**, **FinBERT**, and **GTE**. These vectors are fed into multiple model families (logistic regression / elastic net, random forest, and XGBoost). Performance is evaluated primarily with **ROC-AUC** on direction. To avoid overfitting in a small sample, hyperparameters are not heavily tuned; instead, each model is run under three preset configurations (conservative/moderate/aggressive), enabling a **sensitivity analysis** that labels each (embedding, model) pair as robust or fragile. Results are summarized using a diagnostic framework (Reliable / Uninformative / Dangerous) to distinguish stable signal from configuration-dependent noise.

### Expected Outcome  
A compact comparison of representations and models, plus an end-to-end reproducible pipeline that outputs tables/figures and a single “Grand Champion” inference specification for demonstration.

