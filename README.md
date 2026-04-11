# Inventory Capital Optimizer

#### ML-Driven Demand Forecasting & Prescriptive Liquidation Engine | $10.43M Trapped Capital Released

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/ML-LightGBM-green)](https://lightgbm.readthedocs.io/)

> [!IMPORTANT]
> **Executive Summary:** This project replaces static "Min/Max" reorder rules with a dynamic, machine learning-driven optimization engine. By engineering a custom SPEC Scorer with asymmetric loss functions and Pareto-based service level tiers, the system identified **\$10.43M in trapped working capital** concentrated in slow-moving inventory and protected **$2.44B in potential revenue exposure** through volatility-weighted safety stock buffers — all driven by an XGBoost/LightGBM forecasting pipeline on 14,000+ SKUs.

---

> [!NOTE]
> **Supply Chain Analytics Connection:** This project demonstrates the full lifecycle of ML-driven inventory optimization — from Pareto ABC classification and demand signal engineering to prescriptive liquidation triggers. The SPEC Scorer and Star Schema architecture are directly applicable to pharmaceutical inventory management, medical device supply planning, and healthcare distribution network optimization where stockouts carry patient safety implications beyond financial exposure.

> [!NOTE]
> **Analytics Engineering Connection:** The pipeline demonstrates production-grade data engineering patterns alongside the ML layer: a Star Schema migration from flat-file to dimensional model (`inventory_fact`, `product_dim`, `store_dim`), asymmetric custom loss function design in Python, and a Tableau decision support layer backed by pre-aggregated exports. The architecture maps directly to AE roles requiring both modeling depth and data infrastructure ownership.

---

## Project Overview

Legacy inventory systems in complex retail and manufacturing environments rely on static "Min/Max" thresholds — leading to two asymmetric financial risks that compound over time:

1. **Stockouts:** High-velocity SKUs running dry, costing an estimated **$2.44B in annual revenue risk**.
2. **Trapped Capital:** Slow-moving items tying up **$10.43M in cash flow** and accumulating holding costs.

**The Solution:** A prescriptive analytics pipeline that automates safety stock calculation and prioritizes replenishment strictly based on **Gross Margin Impact** rather than simple unit volume.

**Description:** A dynamic ML pipeline (XGBoost/LightGBM/Random Forest) integrated with a Star Schema data architecture and a Tableau analytical dashboard enabling real-time "What-If" simulations for supply chain leadership.

**Objective:** Achieve Continuity of Supply for critical revenue drivers while maximizing Inventory Turnover and releasing trapped working capital.

## Data Sources

1. **Primary Datasets:** 14,000+ SKU-level daily snapshots including historical transactions, store IDs, and inventory levels.
2. **Additional Data:** External market signals (Competitor Pricing, Weather, Holiday Promotions) and operational metadata (Category-specific Lead Times and Max Shelf Life).

## Process

- Identified the "Vital Few" using Pareto ABC Classification (Top 20% Revenue = Class A) to mathematically enforce tiered service levels — 99% for Class A, flexible for Class C.
- Quantified baseline demand volatility using Category-specific RMSE and identified the "Capital Gap" between current stock levels and model-predicted need.
- Developed a custom **SPEC Scorer** to evaluate models based on the financial cost of errors (Stockout Penalty vs. Overstock Penalty) rather than symmetric statistical accuracy.
- Engineered Smart Markdown Logic that dynamically triggers liquidation prices when "Days of Supply" exceeds "Max Shelf Life."
- Deployed a Tableau Decision Support System allowing managers to adjust Service Level Z-scores live and access a prioritized "Restock Radar."

## Technical Pivot

**From Statistical Accuracy (RMSE) to Dollar Impact (SPEC)**

Initial models were evaluated using RMSE — a symmetric metric that treats a $500 stockout identically to a $5 overstock error.

- **The Change:** Pivoted to a custom SPEC (Stock-keeping-oriented Prediction Error Costs) Scorer, penalizing stockouts at 0.75 and overstocks at 0.25.
- **The Result:** The model actively prioritizes high-margin revenue protection, ensuring Class A items are never stocked out — even at the cost of slight over-prediction.

**From Flat-File to Star Schema Architecture**

Early iterations used a single wide CSV for Tableau, causing "Aggregated Measure Inflation" where Category RMSE was incorrectly summed across dimensions.

- **The Change:** Migrated to a Star Schema with Fact (`inventory_fact`) and Dimension (`product_dim`, `store_dim`) tables.
- **The Result:** Ensured data integrity and enabled sub-second dashboard performance even with complex multi-dimensional "What-If" parameters.

## Key Insights

- **The Symmetry Trap:** Standard ERP systems fail because they optimize for volume, not margin. Dollar-weighting the prediction errors revealed that 80% of revenue risk was concentrated in just 20% of SKUs — exactly the Class A items the SPEC Scorer was tuned to protect.
- **Market Price Sensitivity:** Overstocked items were frequently priced more than 5% above market parity, proving that inventory stagnation was a pricing strategy failure — not a demand forecasting failure.
- **Lead-Time Leverage:** By dynamically linking safety stock buffers to lead-time volatility, the business can calculate the exact capital released when negotiating faster vendor delivery SLAs.

## Recommendations

- **Automate Class A Replenishment:** Deploy the "Restock Radar" thresholds for all Class A items to maintain a 99% service level mandate without manual buyer intervention.
- **Market-Linked Markdowns:** Utilize the Price Index tool to trigger liquidations specifically when pricing exceeds competitive market parity.
- **SLA Renegotiation:** Use the Lead Time Sensitivity model to negotiate faster delivery with high-holding-cost categories to free trapped capital at the source.

## Next Steps & Action Plan

- **Model Retraining:** Schedule quarterly automated retraining loops to adapt to seasonal shifts and macroeconomic demand volatility.
- **API Integration:** Connect the "Suggested Order Qty" logic directly to procurement systems for automated Purchase Order (PO) generation.
- **Performance Scaling:** Refactor the SPEC Scorer into a vectorized NumPy operation to handle the projected scale of 1M+ transaction rows.
