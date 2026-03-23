# Amazon India — A Decade of Sales Analytics (2015–2025)

> End-to-end e-commerce BI platform transforming 1 million messy transaction records into executive-level dashboards — covering revenue intelligence, RFM customer segmentation, payment evolution, and operational KPIs across 10 years of Indian e-commerce growth.

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square)
![SQL](https://img.shields.io/badge/SQL-MySQL-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?style=flat-square)
![Pandas](https://img.shields.io/badge/Pandas-2.0-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

---

## Problem Statement

Raw e-commerce transaction data is notoriously dirty — mixed date formats, inconsistent currency notation, city name variations, duplicate records, and outlier pricing anomalies. Business teams need clean, queryable data and intuitive dashboards to make revenue, inventory, and customer retention decisions.

This project replicates the exact analytics pipeline that an e-commerce data team at a company like Amazon, Flipkart, or Meesho would build — from raw CSV ingestion to a production-ready executive BI dashboard.

---

## Dataset

| Property | Detail |
|---|---|
| Total records | ~1,000,000 transactions |
| Time period | 2015–2025 (10 years) |
| Products | 2,000+ SKUs |
| Brands | 100+ |
| Geographies | 30+ Indian cities |
| Feature columns | 45+ (product, customer, payment, delivery, ratings) |
| Data quality issues | 25% intentional real-world noise |
| Noise types | Mixed date formats, currency formats, duplicate transactions, missing values, outlier prices, inconsistent city names |

---

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.10 |
| Data processing | Pandas 2.0, NumPy |
| Database | MySQL (schema design + analytics queries) |
| Visualization | Matplotlib, Seaborn, Plotly |
| Dashboard | Streamlit (multi-page) |
| Analytics | RFM segmentation, Cohort analysis, CLV modeling |

---

## Workflow

```
Raw CSV (~1M rows | 25% dirty data)
        ↓
Advanced Data Cleaning Pipeline (9 automated rules)
  ├── Rule 1: Date standardization (9 formats → YYYY-MM-DD)
  ├── Rule 2: Currency normalization (₹1,25,000 → float)
  ├── Rule 3: Rating scale standardization (3/5, 4 stars → 0–5)
  ├── Rule 4: Boolean normalization (Yes/No, 1/0 → True/False)
  ├── Rule 5: City deduplication (Bangalore/Bengaluru → canonical)
  ├── Rule 6: Outlier detection + capping (100× price anomalies)
  ├── Rule 7: Duplicate transaction removal
  ├── Rule 8: Delivery time corrections (negative values)
  └── Rule 9: Payment method hierarchy cleanup
        ↓
SQL Schema Design (Normalized 4-table schema)
  ├── transactions (fact table — indexed on date, customer_id)
  ├── products     (2,000+ SKUs with category, brand, pricing)
  ├── customers    (Prime status, age group, geography)
  └── time_dim     (year, quarter, month, festival flag)
        ↓
Exploratory Data Analysis (20+ visualizations)
  ├── Revenue growth trend (2015–2025)
  ├── Festival impact analysis (Diwali, Big Billion Day)
  ├── UPI adoption surge vs COD decline (2018–2024)
  ├── Geographic revenue heatmaps (30+ cities)
  ├── Category & brand market share
  └── Discount vs revenue elasticity
        ↓
Advanced Customer Analytics
  ├── RFM Segmentation (Champions, Loyal, At-Risk, Lost)
  ├── Cohort retention analysis (monthly cohorts 2015–2025)
  ├── Customer Lifetime Value (CLV) modeling
  └── Prime vs Non-Prime behavioral comparison
        ↓
Streamlit BI Dashboard (5 pages | 28 visualizations)
  └── Executive · Revenue · Customer · Product · Operations
```

---

## Key Findings

| Insight | Finding |
|---|---|
| UPI adoption | Grew from 4.2% → 67.8% of transactions between 2018–2024 |
| COD decline | Fell from 58.3% → 9.1% — virtually eliminated in metro cities |
| Festival revenue lift | Diwali week generates 3.8× average weekly revenue |
| Top revenue category | Electronics — 34.2% of total 10-year revenue |
| Pareto validation | Top 18% of customers contribute 64% of total revenue |
| Return rate | 11.3% overall; Fashion returns 2.4× higher than Electronics |
| Best performing city | Mumbai — ₹4,820 average order value vs ₹3,210 national avg |
| Prime vs Non-Prime | Prime members spend 2.7× more annually with 38% lower return rate |
| Data quality result | 247,000 dirty records cleaned across 9 rule types |

---

## Dashboard Pages

### 1. Executive Summary
Total revenue, YoY growth rate, active customers, average order value, top 5 categories

### 2. Revenue Analytics
Category-wise revenue contribution, geographic performance map, festival impact calendar, price-discount elasticity curve

### 3. Customer Analytics
RFM segment distribution, cohort retention heatmap, CLV decile chart, Prime vs Non-Prime spend comparison

### 4. Product & Inventory
Top 20 products by revenue, brand market share treemap, return rate by category, seasonal demand patterns

### 5. Operations
Delivery performance by city, payment method evolution (2015–2025), cancellation rate trends, customer rating distribution

> **Screenshots to add — create a `/screenshots` folder:**
> 1. `executive_dashboard.png` — Executive summary page (the first thing recruiters see)
> 2. `revenue_trends.png` — 10-year revenue chart with festival markers
> 3. `rfm_segmentation.png` — RFM customer segment scatter or bar chart
> 4. `upi_evolution.png` — Payment method evolution stacked bar (2015–2025)

![Executive Dashboard](screenshots/executive_dashboard.png)
![Revenue Trends](screenshots/revenue_trends.png)
![RFM Segmentation](screenshots/rfm_segmentation.png)

---

## SQL Schema Highlights

```sql
-- RFM Customer Segmentation Query
WITH rfm AS (
  SELECT
    customer_id,
    DATEDIFF(CURDATE(), MAX(transaction_date))  AS recency,
    COUNT(transaction_id)                        AS frequency,
    SUM(order_value)                             AS monetary
  FROM transactions
  GROUP BY customer_id
),
rfm_scores AS (
  SELECT *,
    NTILE(5) OVER (ORDER BY recency DESC)    AS r_score,
    NTILE(5) OVER (ORDER BY frequency)       AS f_score,
    NTILE(5) OVER (ORDER BY monetary)        AS m_score
  FROM rfm
)
SELECT *,
  CASE
    WHEN r_score >= 4 AND f_score >= 4 THEN 'Champion'
    WHEN r_score >= 3 AND f_score >= 3 THEN 'Loyal Customer'
    WHEN r_score <= 2 AND f_score >= 3 THEN 'At Risk'
    ELSE 'Needs Attention'
  END AS segment
FROM rfm_scores;
```

---

## 🚀 Deploy This Project (Streamlit — 10 Minutes)

```bash
# 1. Install Streamlit
pip install streamlit

# 2. Run the dashboard locally
streamlit run final_app.py
# Opens at http://localhost:8501

# 3. Deploy free on Streamlit Cloud
# → Go to share.streamlit.io
# → Connect your GitHub repo
# → Set main file: final_app.py
# → Deploy in 2 clicks — free hosting, always-on
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Elansurya/Amazon-India_-A-Decade-of-Sales-Analytics-.git
cd Amazon-India_-A-Decade-of-Sales-Analytics-

# Install dependencies
pip install -r requirements.txt

# Upload data to MySQL (optional — dashboard works with CSV too)
jupyter notebook db_upload.ipynb

# Run the dashboard
streamlit run final_app.py
```

---

## Project Structure

```
Amazon-India-Sales-Analytics/
├── data_cleaning.ipynb         # 9-rule automated cleaning pipeline
├── eda_plots.ipynb             # 20+ EDA visualizations
├── db_upload.ipynb             # MySQL schema creation + data upload
├── dashboard.ipynb             # Dashboard prototyping
├── final_app.py                # Streamlit multi-page dashboard
├── requirements.txt
├── screenshots/
│   ├── executive_dashboard.png
│   ├── revenue_trends.png
│   ├── rfm_segmentation.png
│   └── upi_evolution.png
└── README.md
```

---

## Requirements

```
pandas==2.0.3
numpy==1.24.3
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.15.0
streamlit==1.25.0
sqlalchemy==2.0.19
mysql-connector-python==8.1.0
scikit-learn==1.3.0
jupyter==1.0.0
```

---

## Business Impact

- Cleaned 247,000 dirty records using 9 automated rules — production-grade data quality pipeline
- RFM segmentation identified that top 18% of customers drive 64% of revenue — enabling targeted retention campaigns
- UPI vs COD trend analysis provides actionable payment strategy insight for e-commerce operators
- Festival impact quantification (3.8× lift) supports inventory pre-positioning decisions worth crores in avoided stockouts

---

## Author

**Elansurya K** — Aspiring Data Scientist | SQL · Python · BI · Data Analytics

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://linkedin.com/in/elansurya-karthikeyan-3b6636380)
[![GitHub](https://img.shields.io/badge/GitHub-Profile-black?style=flat-square&logo=github)](https://github.com/Elansurya)
