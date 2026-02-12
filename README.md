# 🛒 Amazon India – A Decade of Sales Analytics (2015–2025)

## 🚀 Project Overview
An end-to-end E-commerce Analytics & Business Intelligence platform built on nearly **1 million real-world transaction records** spanning 10 years (2015–2025).

This project transforms messy raw data (with 25% data quality issues) into production-ready analytical datasets, SQL-optimized storage, and executive-level dashboards for strategic decision-making.

---

## 🎯 Business Problem

E-commerce platforms generate massive transaction data daily.  
However, raw data is often messy, inconsistent, and difficult to interpret.

The objective of this project was to:

- 🧹 Clean and standardize large-scale messy transactional data
- 📊 Perform advanced Exploratory Data Analysis (EDA)
- 🗄️ Design optimized SQL database schema
- 📈 Build interactive dashboards (PowerBI / Streamlit)
- 💡 Deliver strategic business insights for revenue growth

---

## 📊 Dataset Overview

- ~1,000,000 transaction records
- 2000+ products
- 100+ brands
- 30+ Indian cities
- 45+ transaction columns
- 25% intentional real-world data quality issues

### Key Data Categories
- Product details (category, brand, pricing)
- Customer segmentation (Prime status, age group, geography)
- Payment methods (UPI evolution, COD decline)
- Delivery performance
- Festival sales indicators
- Return & rating analysis

---

## 🧹 Advanced Data Cleaning Pipeline

Implemented robust preprocessing pipeline to handle:

- Multiple date formats → standardized to YYYY-MM-DD
- Mixed currency formats (₹1,25,000 → numeric conversion)
- Rating inconsistencies (3/5, 4 stars → standardized scale)
- Boolean normalization (Yes/No, 1/0 → True/False)
- City name standardization (Bangalore/Bengaluru)
- Outlier detection (100x price anomalies)
- Duplicate transaction identification
- Delivery time corrections
- Payment method hierarchy cleanup

Applied statistical validation & domain-driven cleaning logic.

---

## 📈 Exploratory Data Analysis (EDA)

Developed 20+ comprehensive analytical visualizations including:

- 📊 Revenue growth trend (2015–2025)
- 🔥 Seasonal & festival impact analysis
- 🧮 RFM Customer Segmentation
- 💳 Payment method evolution (UPI growth analysis)
- 🌍 Geographic performance heatmaps
- 🏷 Category & brand market share analysis
- 💰 Discount vs revenue elasticity analysis
- 📦 Return rate & satisfaction correlation
- 📉 Customer churn & retention cohort analysis
- 📊 Customer Lifetime Value (CLV) modeling

---

## 🗄️ SQL Database Architecture

Designed optimized relational schema:

### Tables:
- transactions
- products
- customers
- time_dimension

### SQL Features:
- Complex joins across multi-table datasets
- Aggregation queries for KPIs
- Indexing for performance optimization
- Dashboard connectivity integration
- Data validation procedures

---

## 📊 Dashboard Development (25–30 BI Visualizations)

Built multi-page Business Intelligence dashboards including:

### Executive Dashboard
- Total Revenue
- Growth Rate
- Active Customers
- Average Order Value
- Top Categories

### Revenue Analytics
- Category-wise contribution
- Geographic performance
- Festival impact tracking
- Price optimization insights

### Customer Analytics
- RFM segmentation
- Cohort retention curves
- Prime vs Non-Prime behavior
- Demographic spending analysis

### Product & Inventory Analytics
- Product lifecycle tracking
- Brand market positioning
- Demand forecasting insights

### Operations & Logistics
- Delivery performance analysis
- Payment success rates
- Return & cancellation analytics

---

## ⚙️ Tech Stack

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- SQL (MySQL/PostgreSQL)
- PowerBI / Streamlit
- Data Cleaning & Statistical Analysis

---

## 📊 Business Impact & Insights

- Identified long-term revenue growth trajectory
- Discovered UPI adoption surge & COD decline trends
- Revealed high-value customer segments
- Detected seasonal revenue spikes during festivals
- Optimized pricing & discount strategy analysis
- Improved operational efficiency through delivery performance insights

---

## 🏗️ System Workflow

Raw CSV Data  
↓  
Advanced Data Cleaning  
↓  
EDA & Statistical Analysis  
↓  
SQL Database Storage  
↓  
BI Dashboard Visualization  
↓  
Executive Decision Support  

---

## 📌 Key Learnings

- Handling large-scale messy datasets (~1M records)
- Building production-level cleaning pipelines
- Advanced SQL optimization for analytics
- BI dashboard storytelling techniques
- Translating raw data into strategic business decisions

---

## 🔮 Future Enhancements

- Predictive sales forecasting using ML models
- Customer churn prediction
- Automated data refresh pipeline
- Real-time dashboard integration
- Cloud-based data warehouse deployment

---

## 👨‍💻 Author
Elansurya K  
Data Scientist | Machine Learning | NLP | SQL
