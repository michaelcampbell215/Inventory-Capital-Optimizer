# Inventory Capital Optimizer
#### ML-Driven Demand Forecasting & Prescriptive Liquidation Engine
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/ML-LightGBM-green)](https://lightgbm.readthedocs.io/)

> [!IMPORTANT]
> **Executive Summary:** This project replaces static "Min/Max" reorder rules with a dynamic, machine learning-driven optimization engine. By implementing asymmetric loss functions and Pareto-based service levels, the system effectively identified and protected \$2.44B in potential revenue leakage while releasing $10.43M in trapped capital through intelligent, volatility-weighted safety stock buffering.

---

## Project Overview

Legacy inventory systems in complex retail and manufacturing enviornments rely on static "Min/Max" thresholds leading to two asymmetric financial risks:

1. **Stockouts:** High-velocity SKUs running out, costing an estimated $2.44B in annual revenue risk.

2. **Trapped Capital:** Slow-moving items tying up $10.43M in cash flow and increasing holding costs.

**The Solution:**  A prescriptive analytics pipeline that automates safety stock calculation and prioritizes replenishment strictly based on Gross Margin Impact rather than simple unit volume.

**Description:** A dynamic ML pipeline (XGBoost/Random Forest) integrated with a Star Schema data architecture and a Tableau analytical dashboard to enable real-time "What-If" simulations for supply chain leadership.

**Objective:** Achieve Continuity of Supply for critical revenue drivers while maximizing Inventory Turnover and releasing trapped working capital.

## Data Sources

1. **Primary Datasets:** 14,000+ SKU-level daily snapshots including historical transactions, store IDs, and inventory levels.
2. **Additional Data:** External market signals (Competitor Pricing, Weather, Holiday Promotions) and operational metadata (Category-specific Lead Times and Max Shelf Life).

## Process

*    Identified the "Vital Few" using Pareto ABC Classification (Top 20% Revenue = Class A) to mathematically enforce tiered service levels.

*    Quantified baseline volatility using Category-specific RMSE and identified the "Capital Gap" between current stock and model-predicted needs.

*    Developed a custom SPEC Scorer to evaluate models based on the financial cost of errors (Stockout vs. Overstock) rather than symmetric statistical averages.

*    Engineered a Smart Markdown Logic that dynamically triggers liquidation prices when "Days of Supply" exceeds "Max Shelf Life."

*    Deployed a Tableau Decision Support System that allows managers to adjust Service Level Z-scores live and access a prioritized "Restock Radar."

## Technical Pivot

**From Statistical Accuracy (RMSE) to Dollar Impact (SPEC)** 
Initially, the models were evaluated using RMSE (Root Mean Square Error). However, RMSE is a symmetric metric. It treats a \$500 stockout identically to a $5 overstock error.
*   **The Change:** We pivoted to a custom SPEC (Stock-keeping-oriented Prediction Error Costs) Scorer.
*   **The Result:** By penalizing stockouts at 0.75 and overstocks at 0.25, the model actively prioritizes high-margin revenue protection, ensuring we never run out of "Class A" items even at the cost of slight over-prediction.

**From Flat-File to Star Schema Architecture**
Early iterations used a single, wide CSV for Tableau. This led to "Aggregated Measure Inflation" (where Category RMSE was incorrectly summed).

*   **The Change:** Migrated to a Star Schema with Fact (inventory_fact) and Dimension (product_dim, store_dim) tables.
*   **The Result:** Ensured data integrity and allowed for sub-second dashboard performance even with complex "What-If" parameters.
## Key Insights

*   **The Symmetry Trap:** Standard ERP systems fail because they optimize for volume. Dollar-weighting the errors revealed that 80% of our revenue risk was concentrated in just 20% of the SKUs.
*   **Market Price Sensitivity:** Overstocked items were frequently found to be priced >5% above market parity, proving that inventory stagnation was a pricing strategy failure, not a demand forecasting failure.
*   **Lead-Time Leverage:** By dynamically linking safety stock buffers to lead-time volatility, the business can explicitly calculate the exact capital released when negotiating faster vendor delivery SLAs.

## Recommendations

*   **Automate Class A Replenishment:** Deploy the "Restock Radar" thresholds for all Class A items to maintain a 99% service level mandate.
*   **Market-Linked Markdowns:** Utilize the Price Index tool to trigger liquidations specifically when we are priced higher than competitors.
*   **SLA Renegotiation:** Use the Lead Time Sensitivity model to negotiate faster delivery with Furniture and Electronics vendors to free up the most trapped capital.

## Next Steps & Action Plan

*   **Model Retraining:** Schedule quarterly automated retraining loops to adapt to seasonal shifts and macroeconomic volatility.

*   **API Integration:** Connect the "Suggested Order Qty" logic directly to procurement systems for automated Purchase Order (PO) generation.

*   **Performance Scaling:** Refactor the SPEC Scorer into a vectorized NumPy operation to handle the projected scale of 1M+ transaction rows.