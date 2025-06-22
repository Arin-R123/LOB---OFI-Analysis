# Limit Order Book Analysis


Project Overview
This project implements and extends the methodology from the 2023 paper "Cross-impact of Order Flow Imbalance in Equity Markets" by Cont et al. Using 2,500 1-minute AAPL Limit Order Book snapshots, we construct normalized multi-level OFI vectors and apply PCA to create an integrated OFI feature for short-horizon return forecasting.

Methodology
Data: 1-minute snapshots of AAPL’s top 10 LOB levels

Step 1 – OFI Calculation: For each level, compute:

Bid flow: positive if bid price increases or bid size increases

Ask flow: negative if ask price increases or ask size decreases

Step 2 – Normalization: Divide OFI at each level by its average depth over the time window

Step 3 – PCA Integration:

Construct 10-dimensional OFI vector

Apply PCA

Use first principal component (captures >89% variance)

Step 4 – Forecasting:

Use the PCA-integrated OFI to predict 2-minute log returns

Evaluate using linear regression and R²

Reference:
Cont, R., Cucuringu, M., & Zhang, C. (2023). Cross-impact of order flow imbalance in equity markets. Quantitative Finance

