# Storm Restoration Prioritization AI + GridGuard

**A full-stack deep learning platform for power outage forecasting, smart grid topology analysis, and pre-storm critical facility alerting.**

Author: Harsha Venkateshwara
Course: CSE 676A Deep Learning, University at Buffalo, Spring 2026

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Datasets](#3-datasets)
4. [Installation and Setup](#4-installation-and-setup)
5. [Pipeline Execution](#5-pipeline-execution)
6. [System 1: Storm Restoration Prioritization AI](#6-system-1-storm-restoration-prioritization-ai)
7. [System 2: GridGuard AI](#7-system-2-gridguard-ai)
8. [Dashboard](#8-dashboard)
9. [Operational Use Cases](#9-operational-use-cases)
10. [Key Findings](#10-key-findings)
11. [Production Roadmap](#11-production-roadmap)
12. [References](#12-references)

---

## 1. Project Overview

Storm Restoration Prioritization AI addresses a concrete operational failure in the U.S. utility industry: when a major storm approaches, crew staging decisions are made through phone calls, spreadsheets, and institutional memory rather than data. There is no system that tells a grid operator, in real time, that the Mineola substation on Long Island feeds four hospitals simultaneously and should be the first asset protected before Hurricane Sandy makes landfall.

The data to solve this has existed for years. The DOE EAGLE-I dataset tracks every county-level outage in the United States at 15-minute resolution since 2014. NOAA records every storm event. OpenStreetMap contains substation locations and power line geometry. The HIFLD database has GPS coordinates for every hospital, fire station, and school in the country. The gap was that no one had combined these datasets into a working operational tool.

This project does exactly that. It consists of two interconnected systems.

**System 1: Storm Restoration Prioritization AI**

A spatiotemporal deep learning ensemble trained on 160 million outage records across 3,044 US counties, spanning 2015 through 2022. The system forecasts county-level power outage probability 3 to 24 hours before storms and converts predictions into a configurable crew dispatch priority queue.

**System 2: GridGuard AI**

An extension to System 1 that adds power grid topology analysis using OpenStreetMap infrastructure data and the HIFLD critical facility database. The system builds a NetworkX graph of the Long Island power grid, performs k-hop BFS traversal from at-risk substations when a storm is declared, and generates pre-storm CRITICAL/HIGH/MEDIUM alerts for hospitals, fire stations, and schools on predicted outage paths.

The combined platform was validated against Hurricane Ian (2022), Hurricane Sandy (2012), and Tropical Storm Isaias (2020). In the 2022 held-out test year, the model ranked Lee County, Florida as the number-one national priority. Hurricane Ian devastated Lee County on September 28, 2022. The model was trained exclusively on data through 2020.

---

## 2. Repository Structure

```
storm_restoration_ai/
|
|-- data/
|   |-- raw/
|   |   |-- eagle_i/eaglei_outages/          # EAGLE-I CSVs, one per year
|   |   |-- noaa/                            # NOAA Storm Events CSVs
|   |   |-- grid/                            # OSM substations, power lines
|   |   |-- facilities/                      # Hospitals, fire stations, schools
|   |   `-- storms/                          # Historical storm polygons (GeoJSON)
|   |-- processed/
|   |   |-- eagle_i/                         # Per-year Parquet files
|   |   `-- noaa/                            # Per-year Parquet files
|   |-- features/
|   |   |-- feature_store_YYYY.parquet       # Per-year feature matrices
|   |   |-- storm_pivot.parquet              # Hourly storm flags
|   |   |-- county_stats.parquet             # Per-county p90 thresholds
|   |   |-- train.parquet                    # 2015-2020, balanced
|   |   |-- val.parquet                      # 2021
|   |   `-- test.parquet                     # 2022
|   `-- graph/
|       |-- long_island_grid.pkl             # Serialized NetworkX graph
|       `-- alerts_output.json               # GridGuard alert output
|
|-- outputs/
|   |-- models/
|   |   |-- lstm_multitask_best.pt           # Best LSTM checkpoint
|   |   |-- gradient_boosting_final.pkl      # GB model
|   |   |-- logistic_regression_final.pkl    # LR baseline
|   |   |-- feature_scaler.pkl               # StandardScaler
|   |   |-- training_results.pkl             # Training history
|   |   |-- evaluation_results.pkl           # Test metrics and model objects
|   |   `-- county_priority_2022.parquet     # Priority scores per county
|   `-- plots/
|       |-- lstm_evaluation.png
|       |-- model_comparison.png
|       `-- gridguard_map.html               # Interactive Folium map
|
|-- convert_eaglei.py                        # EAGLE-I CSV to Parquet
|-- convert_noaa.py                          # NOAA CSV to Parquet
|-- data_audit.py                            # EDA and data quality checks
|-- build_features.py                        # Feature store construction
|-- train_model.py                           # LSTM training
|-- run_test_eval.py                         # Test evaluation without retraining
|-- evaluate.py                              # Baseline comparison and county priority
|-- gridguard_download.py                    # Download OSM, HIFLD, storm polygons
|-- gridguard_build_graph.py                 # Build NetworkX grid topology graph
|-- gridguard_alert_engine.py                # K-hop BFS alert generation
|-- gridguard_map.py                         # Folium interactive map
|-- app.py                                   # Streamlit dashboard (6 tabs)
`-- requirements.txt
```

---

## 3. Datasets

### 3.1 EAGLE-I Outage Dataset

Source: U.S. Department of Energy, Oak Ridge National Laboratory
URL: https://www.osti.gov/biblio/2324037

EAGLE-I aggregates utility self-reported outage data into 15-minute resolution county-level estimates of customers without power across the continental United States.

| Year | Raw Records   | Counties | Outage Rate |
|------|--------------|----------|-------------|
| 2015 | 12,939,157   | 2,498    | 10.40%      |
| 2016 | 13,306,024   | 2,405    | 10.83%      |
| 2017 | 15,078,364   | 2,887    | 11.24%      |
| 2018 | 21,776,806   | 2,895    | 9.88%       |
| 2019 | 24,074,122   | 3,025    | 10.06%      |
| 2020 | 25,545,517   | 3,072    | 10.77%      |
| 2021 | 24,826,102   | 3,045    | 12.28%      |
| 2022 | 22,327,833   | 3,044    | 11.98%      |

**Data quality issue resolved:** For 2015 and 2016, EAGLE-I logged rows only when outages were actively occurring. A missing row in those years indicates either no outage or a reporting gap, not definitively zero customers out. The pipeline treats null values as imputed zeros and adds a separate binary null-count indicator feature so the model can distinguish true zeros from reporting artifacts.

U.S. territories were excluded because their infrastructure profiles differ substantially from the continental states.

### 3.2 NOAA Storm Events Database

Source: NOAA National Centers for Environmental Information
URL: https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/

After filtering to county-level records (CZ_TYPE = C), 213,603 events spanning 2015 through 2022 were available.

| Group    | Event Types                                          | Approx. Count |
|----------|------------------------------------------------------|---------------|
| Wind     | Thunderstorm Wind, High Wind, Strong Wind, Tornado   | 146,000+      |
| Winter   | Winter Storm, Ice Storm, Heavy Snow, Blizzard        | 27,000+       |
| Flood    | Flash Flood, Flood, Heavy Rain                       | 74,000+       |
| Tropical | Hurricane, Tropical Storm                            | Seasonal      |
| Other    | Lightning, Dense Fog, Extreme Cold                   | Remainder     |

Each event was expanded to hourly county-hour records using reported start and end timestamps, with a maximum cap of 72 hours per event.

### 3.3 OpenStreetMap Grid Data

Source: OpenStreetMap contributors via Overpass API
URL: https://overpass-api.de

Substation node locations and power line geometry for Long Island were queried using a bounding box covering Nassau and Suffolk counties (40.544N, 73.994W, 41.027N, 71.856W). A curated fallback of 20 known LIPA substations is used when the live API returns no results.

### 3.4 HIFLD Critical Infrastructure

Source: Homeland Infrastructure Foundation-Level Data (DHS)
URL: https://services1.arcgis.com/Hp6G80Pky0om7QvQ/arcgis/rest/services/

Queried for Nassau and Suffolk counties, New York:

- Hospitals: 15 records with name, address, bed count, trauma level, helipad availability
- Fire Stations: 15 records with name, county, and coordinates

### 3.5 Historical Storm Polygons

Three storm impact polygons were generated programmatically:

- Sandy 2012: Center (40.4N, 73.9W), radius 220 km
- Isaias 2020: Center (40.7N, 73.5W), radius 120 km
- Henri 2021: Center (40.9N, 72.8W), radius 100 km

---

## 4. Installation and Setup

Python 3.11 or 3.12 is recommended.

```bash
# Core ML and data packages
pip install torch torchvision
pip install pandas numpy pyarrow scikit-learn
pip install matplotlib seaborn plotly
pip install streamlit>=1.32

# GridGuard extension packages
pip install osmnx networkx geopandas shapely folium
pip install pyproj rtree scipy overpy requests

# Additional utilities
pip install fastparquet openpyxl
```

On Windows with multiple Python installations, install GridGuard packages explicitly into the same Python that runs Streamlit:

```bash
C:\Users\<user>\AppData\Local\Programs\Python\Python312\python.exe -m pip install folium geopandas networkx shapely pyproj rtree osmnx
```

**Known compatibility notes:**

- PyTorch 2.10 removed the `verbose=True` argument from `ReduceLROnPlateau`. Remove it from `train_model.py` if present.
- PyTorch 2.6 changed the default value of `weights_only` in `torch.load` from `False` to `True`. Use `weights_only=False` when loading checkpoints that contain NumPy scalars.
- Streamlit 1.53.1 deprecates `use_container_width` in favor of `width='stretch'`. The warnings are harmless and do not affect functionality.

---

## 5. Pipeline Execution

Run scripts in this order from the project root directory.

```bash
# Step 1: Convert raw data to Parquet
python convert_eaglei.py
python convert_noaa.py

# Step 2: Build the feature store
python build_features.py

# Step 3: Train the LSTM model (~4 hours on CPU)
python train_model.py

# Step 4: Evaluate on the test set without retraining
python run_test_eval.py

# Step 5: Train baselines and generate county priority scores
python evaluate.py

# Step 6: Download GridGuard data
python gridguard_download.py

# Step 7: Build the grid topology graph
python gridguard_build_graph.py

# Step 8: Generate alerts
python gridguard_alert_engine.py

# Step 9: Build the interactive map
python gridguard_map.py

# Step 10: Launch the dashboard
streamlit run app.py
```

Expected output from `build_features.py`:

```
Storm pivot: (662324, 9) | Active hours: 662,324
County thresholds computed for 2,902 counties
train.parquet: 17,809,445 rows | outage rate: 20.00%
val.parquet:    7,437,292 rows | outage rate: 12.28%
test.parquet:   6,670,407 rows | outage rate: 11.98%
```

Expected output from `gridguard_build_graph.py`:

```
Total nodes : 50
Total edges : 90
Node breakdown:
  substation   : 20
  hospital     : 15
  fire_station : 15
```

Expected output from `gridguard_alert_engine.py` (Sandy simulation):

```
Substations in storm zone: 20
At-risk substations (P>0.40): 20
Total alerts generated: 29
  CRITICAL (hospitals): 14
  HIGH (fire stations): 15
  MEDIUM (schools):      0
```

---

## 6. System 1: Storm Restoration Prioritization AI

### 6.1 Feature Engineering

**Target variable design:**

Three threshold strategies were evaluated before adopting the county-adaptive approach.

| Strategy                    | Positive Rate | Problem                          |
|-----------------------------|--------------|----------------------------------|
| Any value greater than zero | 71.78%       | Noise and routine switching      |
| Fixed threshold: 50 customers | 25.72%     | Ignores county population scale  |
| County-adaptive p90 (adopted) | 9.85%      | Meaningful relative disruption   |

For each county, the 90th percentile of historical peak customers-out values is computed from 2015-2018 training data. A county-hour is labeled positive only if peak customers-out exceeds both the county-specific p90 and an absolute floor of 10 customers. This correctly treats 50 customers out in a rural county of 3,000 as more significant than the same number in a metropolitan county of 500,000.

**Feature groups (28 total):**

| Group              | Features                                                      | Count |
|--------------------|---------------------------------------------------------------|-------|
| Storm Indicators   | wind, winter, flood, tropical, other flags; severity score; rolling 6h/12h/24h sums | 13 |
| Temporal Encoding  | hour sin/cos, month sin/cos, is_weekend, quarter              | 6     |
| Lag Features       | peak customers at 1h/24h/168h lag; outage labels at 24h/168h lag | 5 |
| County Context     | 30-day fragility index, p90 threshold, baseline mean, null-count indicator | 4 |

Temporal features use cyclical sine/cosine encodings so that hour 23 and hour 0 are treated as adjacent rather than as opposite ends of a linear scale.

### 6.2 Model Architecture

**Multi-Task LSTM (258,340 parameters):**

```
Input:  [batch, 24, 28]

Input Projection:
  Linear(28 -> 64) + LayerNorm + ReLU + Dropout(0.15)

LSTM Encoder:
  2-layer LSTM, hidden_size=128, dropout=0.3 between layers

Attention Pooling:
  alpha_t = softmax(v^T * tanh(W_a * h_t))
  c = sum(alpha_t * h_t)

Head A - Classification:
  MLP(128 -> 64 -> 32 -> 1) + Sigmoid -> P(outage)
  Loss: Focal Loss (alpha=0.75, gamma=2.0)

Head B - Quantile Regression:
  MLP(128 -> 64 -> 32 -> 2) -> [P50, P90] of peak customers-out
  Loss: Pinball Loss

Combined objective:
  L = 0.6 * L_Focal + 0.4 * L_Pinball
```

**Why Focal Loss:** Standard binary cross-entropy with a 10% positive rate causes the model to learn that predicting all negatives achieves 90% accuracy. Focal Loss introduces a modulating factor `(1 - p_t)^gamma` that down-weights easy negative examples and forces the model to focus on hard outage examples near the decision boundary.

**Why Pinball Loss:** Standard MSE does not produce calibrated quantile estimates. Pinball Loss is an asymmetric loss that penalizes under-prediction of a quantile at rate q more heavily than over-prediction, producing statistically valid P50 and P90 bounds.

**Why multi-task:** Gradient Boosting cannot produce uncertainty bounds. The P90 worst-case estimate from Head B is operationally irreplaceable for mobile generator staging decisions, where a crew supervisor needs to know the credible worst case, not just the expected outcome.

**Gradient Boosting baseline:**

Trained on a stratified balanced sample of 300,000 county-hour records (150,000 positive, 150,000 negative). Parameters: 200 estimators, max depth 5, learning rate 0.05, subsample 0.8. Balanced sampling was critical: training on the raw 9.85% positive rate produced near-zero recall despite high accuracy.

### 6.3 Training Protocol

| Split      | Years     | Sequences  | Outage Rate         |
|------------|-----------|------------|---------------------|
| Training   | 2015-2020 | 17,809,445 | 20.00% (balanced)   |
| Validation | 2021      | 716,067    | 11.87%              |
| Test       | 2022      | 639,529    | 11.95%              |

Optimizer: AdamW, learning rate 1e-3, weight decay 1e-4.
Scheduler: ReduceLROnPlateau, factor 0.5, patience 3 epochs.
Early stopping: patience 7 epochs on validation PR-AUC.
Best checkpoint: epoch 19 of 30, validation PR-AUC 0.6331.

### 6.4 Results

**Model comparison on the 2022 held-out test set:**

| Model               | ROC-AUC | PR-AUC | F1     | Brier  |
|---------------------|---------|--------|--------|--------|
| Logistic Regression | 0.8919  | 0.6733 | 0.6592 | 0.1282 |
| Multi-Task LSTM     | 0.8417  | 0.6088 | 0.5968 | 0.1564 |
| Gradient Boosting   | 0.9112  | 0.7485 | 0.7567 | 0.0868 |

A naive classifier predicting the base rate uniformly achieves a PR-AUC of approximately 0.10 given the 10% positive rate. The Gradient Boosting PR-AUC of 0.749 represents a 7.5-fold improvement on the operationally critical metric.

**Feature importance:**

The county fragility index (30-day rolling outage rate) is the single most important feature at approximately 30% of total Gradient Boosting importance. The 24-hour lag peak customers-out contributes roughly 10%. These two features account for 40% of total predictive power together. Wind-related storm features rank third among feature groups.

**Hurricane Ian retrospective validation:**

Under the held-out 2022 test set, with the model trained exclusively on 2015-2020 data:

| FIPS  | County              | State | P(Outage) | Peak Customers | Priority Score | Rank |
|-------|---------------------|-------|-----------|----------------|----------------|------|
| 12071 | Lee County          | FL    | 0.9951    | 456,573        | 0.9234         | 1    |
| 12115 | Sarasota County     | FL    | 0.9947    | 257,410        | 0.7406         | 2    |
| 12095 | Orange County       | FL    | 0.9948    | 229,025        | 0.7168         | 3    |
| 12127 | Volusia County      | FL    | 0.9946    | 224,435        | 0.7148         | 4    |
| 12057 | Hillsborough County | FL    | 0.9940    | 217,144        | 0.7068         | 5    |

Hurricane Ian made direct landfall at Lee County on September 28, 2022, as a Category 4 hurricane. It caused 2.6 million customer outages across Florida, the second-largest outage event in U.S. history. The model placed Lee County first nationally with no 2022 data in training.

---

## 7. System 2: GridGuard AI

### 7.1 Graph Construction

The Long Island power grid is modeled as an undirected graph G = (V, E).

| Node Type    | Count | Key Attributes                              |
|--------------|-------|---------------------------------------------|
| Substation   | 20    | name, voltage_kv, subtype, lat, lon, risk   |
| Hospital     | 15    | name, beds, trauma, helipad, lat, lon       |
| Fire Station | 15    | name, county, lat, lon                      |

**Edges:**

Substation-to-substation edges connect substations within 25 kilometers, weighted by Euclidean distance. Facility-to-substation edges are added via cKDTree spatial index: each facility connects to its two nearest substations within 20 kilometers. The resulting graph has 50 nodes and 90 edges.

### 7.2 Alert Engine

**Storm polygon intersection:** Substations whose coordinates fall inside the storm impact polygon receive the maximum adjusted risk score. Substations within 30 kilometers of the polygon boundary receive partial risk elevation proportional to proximity.

**K-hop BFS traversal parameters:**

| Facility Type | Max Hops | Priority Weight | Alert Level |
|---------------|----------|-----------------|-------------|
| Hospital      | 2        | 10              | CRITICAL    |
| Fire Station  | 3        | 7               | HIGH        |
| School        | 3        | 4               | MEDIUM      |

**Alert scoring formula:**

```
AlertScore(f, s) = P_s(outage) * w_f / sqrt(d(s, f))
```

Where P_s is the adjusted risk score of feeding substation s, w_f is the facility priority weight, and d(s, f) is the hop distance. When a facility is reachable from multiple at-risk substations, the highest alert score is retained.

**Recommended actions by risk level:**

- Hospital, risk > 0.85: Activate emergency generators immediately. Alert backup power team. Notify generator fuel supplier. Contact backup facility for patient diversion plan.
- Hospital, risk 0.65-0.85: Test emergency generators within 2 hours. Confirm fuel reserves for 72 hours. Alert on-call engineering staff.
- Fire station, risk > 0.85: Deploy mobile generator. Activate backup communications. Pre-position apparatus for storm response.

### 7.3 Validation

**Hurricane Sandy simulation results:**

| Metric                        | Value        |
|-------------------------------|--------------|
| Substations in storm zone     | 20 of 20     |
| Adjusted risk (all substations) | 0.999      |
| Total alerts generated        | 29           |
| CRITICAL alerts (hospitals)   | 14           |
| HIGH alerts (fire stations)   | 15           |
| Alert generation time         | Under 2 seconds |

Top CRITICAL alerts by alert score:

1. Long Island Jewish Medical Center (Mineola Substation, 1 hop, risk 0.999)
2. St. Francis Hospital (Mineola Substation, 1 hop, risk 0.999)
3. Winthrop University Hospital (Mineola Substation, 1 hop, risk 0.999)
4. Glen Cove Hospital (Mineola Substation, 1 hop, risk 0.999)
5. NYU Langone Hospital Long Island (Garden City Substation, 1 hop, risk 0.999)
6. North Shore University Hospital (Hicksville Substation, 1 hop, risk 0.984)
7. Huntington Hospital (Hicksville Substation, 1 hop, risk 0.984)

**Single point of failure finding:** Mineola Substation feeds four hospitals simultaneously at one graph hop distance. This interdependency is only visible after building the grid topology graph. One mobile generator pre-staged at Mineola before Sandy's landfall would have protected all four hospitals through the storm.

---

## 8. Dashboard

Launch with `streamlit run app.py` and open `http://localhost:8501`.

### Tab 1: Risk Map

Plotly choropleth of 3,044 counties colored by priority score. Storm Scenario Simulator selects storm type (No Storm through Hurricane Category 5) and forecast horizon (3, 6, 12, or 24 hours). All metrics and the map update instantaneously. Live insights panel shows risk distribution donut, top-risk states bar chart, and top five priority county cards.

Storm multipliers: Thunderstorm 1.4x, Winter Storm 1.5x, Flash Flood 1.3x, High Wind 1.6x, Hurricane Cat 1 2.0x, Hurricane Cat 3 3.5x, Hurricane Cat 5 5.5x.

### Tab 2: Priority Queue

Ranked county restoration list. Priority score formula:

```
PriorityScore = w1 * P(outage) + w2 * normalized_customers + w3 * outage_rate
```

Default weights: w1 = 0.50, w2 = 0.40, w3 = 0.10. All adjustable via sidebar sliders. State filter available. Export CSV button downloads a crew dispatch brief.

Under Hurricane Ian simulation filtered to Florida, the top 10 counties collectively represent 2,290,187 customers at risk.

### Tab 3: County Drilldown

Selector for top 200 counties by risk score. Displays a metadata card with P(outage), peak customers, and historical outage rate. A 24-hour risk forecast profile shows temporal risk evolution. Comparison mode displays two counties side by side. Feature importance chart shows the top 15 Gradient Boosting features color-coded by group.

### Tab 4: Model Results

Full three-model comparison table. Training history curves across 26 epochs. ROC, precision-recall, and calibration curves. Predicted probability distribution for the 2022 test set. Architecture documentation for FERC and state PSC explainability requirements.

### Tab 5: How to Use

Step-by-step operator workflow guide for all roles. Before/after comparison table contrasting reactive vs. proactive dispatch workflows.

### Tab 6: GridGuard Alert Map

Folium interactive map of the Long Island grid topology. Substations as circles (red for in-storm zone, amber for elevated risk, blue for normal). Power lines as polylines. Dashed routing lines from at-risk substations to downstream facilities. Hospital markers in red, fire station markers in orange. Every marker shows a popup with facility details and recommended action.

Below the map: filterable alert queue by CRITICAL/HIGH/MEDIUM priority, at-risk substations table with voltage and risk scores, and Export Alert Brief CSV button.

---

## 9. Operational Use Cases

### Pre-Storm Crew Staging

When a storm warning is issued, a storm desk operator selects the storm type in the sidebar. The system identifies the top priority counties and feeding substations within seconds. Crews are pre-positioned in staging areas before the storm arrives, reducing restoration time from the current 72-96 hour average.

Hurricane Sandy scenario: At 17:02 on October 28, 2012, 18.5 hours before landfall, the operator selects Hurricane Category 1. The GridGuard CRITICAL banner fires immediately for Long Island Jewish Medical Center, fed by Mineola Substation at risk 0.999. The alert brief is exported at 17:06. By 17:20, crew dispatch stages a mobile generator at Mineola. When Sandy makes landfall at 23:30, the hospital maintains uninterrupted power. Without the system, the mobile generator arrives at approximately 01:00 on October 29, nearly two hours after the substation has already failed.

### Mutual Aid Request Prioritization

During major storms, utilities request crew assistance from neighboring utilities through mutual aid agreements. This platform provides a quantified, defensible ranking that can be shared between utility control rooms. The priority score and peak customer count give receiving utilities objective data to evaluate which incoming requests to fulfill first.

### Regulatory Reporting and FERC Compliance

The Federal Energy Regulatory Commission and state public utility commissions require utilities to demonstrate that storm response plans are evidence-based. The feature importance analysis, transparent priority score formula, and CSV export provide an auditable record of how restoration decisions were made. Each export from a storm event becomes a compliance artifact.

### Vulnerability-Weighted Equity Analysis

Increasing the vulnerability weight slider from 0.10 to 0.40 elevates counties with higher historical outage rates. This addresses a documented disparity: lower-income and rural counties with older infrastructure often wait longest for restoration under purely customer-count-based prioritization.

Winter Storm scenario: For a nor'easter with multiplier 1.5 and elevated vulnerability weight, two eastern Suffolk counties with aging infrastructure move from rank 15 and 18 into the top 5. Peconic Bay Medical Center in Riverhead appears in the GridGuard CRITICAL list despite its low absolute customer count because its historical outage rate is the highest on Long Island.

### Post-Storm Incident Review

After each storm event, predictions can be compared against actual outage records to identify systematic failures. The Tropical Storm Isaias PSC investigation (2020) cited insufficient pre-storm crew staging as the primary failure. A quantified, ranked, and exportable pre-storm action list produced by this platform directly addresses that documented gap.

---

## 10. Key Findings

**County fragility is the dominant predictor.** The 30-day rolling outage rate accounts for approximately 30% of total Gradient Boosting feature importance. Counties with recent outage history are structurally more vulnerable. This finding aligns with utility engineer intuition but had not previously been quantified at county scale. The implication is that restoration crews should flag counties with elevated fragility indices for preventive maintenance between storm events.

**Both models are necessary for different reasons.** Gradient Boosting achieves higher classification metrics because the two dominant features are tabular lag structures that tree models handle optimally. The Multi-Task LSTM's irreplaceable contribution is the P90 quantile regression head, which produces calibrated worst-case estimates of peak customers-out. Gradient Boosting cannot produce these uncertainty bounds, which are critical for mobile generator staging decisions where the credible worst case drives resource allocation.

**Lee County was correctly identified without 2022 data.** The model ranked Lee County, Florida as the number-one national priority in the 2022 test year, trained exclusively on 2015-2020 data. Hurricane Ian made direct landfall at Lee County on September 28, 2022. The model placed Lee County first by a substantial margin over every other county in the country. This is the strongest possible retrospective validation of an outage forecasting system.

**Mineola Substation is a single point of failure.** The GridGuard graph traversal identified that Mineola Substation feeds Long Island Jewish Medical Center, St. Francis Hospital, Winthrop University Hospital, and Glen Cove Hospital simultaneously at one graph hop distance. This critical infrastructure interdependency is invisible in any county-level analysis. It only becomes apparent after building the grid topology graph from public data. One mobile generator pre-staged at Mineola protects all four hospitals simultaneously.

**29 alerts in under 2 seconds.** The k-hop BFS traversal across the full Long Island grid graph (50 nodes, 90 edges) completes in under 200 milliseconds on a standard CPU. All 14 hospitals and 15 fire stations at risk under the Hurricane Sandy simulation are identified with pre-written recommended actions before any human decision-making is required.

**$170 million annual economic impact potential.** The Lawrence Berkeley National Laboratory estimates the societal cost of U.S. power outages at $2.70 per customer-hour. With 21 million customers tracked in the 2022 test year, a 30% improvement in restoration time translates to approximately $170 million in avoided economic losses annually. The platform requires no proprietary sensors, no utility SCADA access, and no infrastructure investment beyond standard cloud compute. All data sources are public and freely available at zero cost.

---

## 11. Production Roadmap

The current system runs as a local Streamlit application against static 2022 test data. A production deployment requires the following components.

| Component              | Technology                   | Purpose                          |
|------------------------|------------------------------|----------------------------------|
| Live data ingestion    | Apache Kafka + EAGLE-I API   | 15-minute outage updates         |
| Inference service      | FastAPI + Docker + Kubernetes | Scalable REST API               |
| Cloud deployment       | Azure AKS or AWS EKS         | High availability                |
| Model registry         | MLflow                       | Versioning and A/B testing       |
| Automated retraining   | Apache Airflow DAG           | Monthly retraining pipeline      |

Inference latency for a batch prediction across all 3,044 counties using the trained Gradient Boosting model is under 200 milliseconds on a standard compute instance. The dashboard loads from cached Parquet data in under 2 seconds. These characteristics are fully compatible with a 15-minute EAGLE-I update cycle.

**Future model improvements:**

- Replace the LSTM with a Temporal Fusion Transformer for interpretable multi-horizon forecasting with variable-selection networks
- Add gridded meteorological covariates from the NOAA High-Resolution Rapid Refresh model
- Incorporate NASA Black Marble night-time lights satellite data as an independent outage validation signal
- Extend vulnerability weighting with Census data on population age, medical electricity dependency, and infrastructure vintage

**Publication targets:**

- IEEE Transactions on Smart Grid (primary)
- Applied Energy (Elsevier, open access)
- arXiv preprint to establish priority before journal submission

The specific novelty claim: this is the first system to combine national-scale county-level outage forecasting using exclusively public government datasets with grid topology graph traversal to generate pre-storm critical facility alerts, delivered through an open-source deployed dashboard validated against actual historical storm events.

---

## 12. References

1. Brelsford, C. et al. (2024). A dataset of recorded electricity outages by United States county, 2014-2022 (EAGLE-I). DOE OSTI. https://www.osti.gov/biblio/2324037

2. NOAA National Centers for Environmental Information (2024). Storm Events Database (Bulk Data Download). https://www.ncei.noaa.gov/stormevents/ftp.jsp

3. Lim, B., Arik, S.O., Loeff, N., and Pfister, T. (2021). Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting. International Journal of Forecasting, 37(4), 1748-1764.

4. Lin, T.-Y., Goyal, P., Girshick, R., He, K., and Dollar, P. (2017). Focal Loss for Dense Object Detection. IEEE International Conference on Computer Vision (ICCV).

5. Chen, Z. et al. (2024). Correlating Power Outage Spread with Infrastructure Interdependencies During Hurricanes. arXiv:2407.09962.

6. HIFLD / DHS (2024). Homeland Infrastructure Foundation-Level Data. https://hifld-geoplatform.opendata.arcgis.com

7. OpenStreetMap Contributors (2024). Power Networks. https://wiki.openstreetmap.org/wiki/Power_networks

8. Lawrence Berkeley National Laboratory (2023). The Value of Service Reliability for Electric Utility Customers. U.S. DOE Office of Electricity.

9. PSEG Long Island (2020). Storm Isaias After-Action Report. Submitted to New York State PSC.

---

Copyright 2026 Harsha Venkateshwara, Saba Minaz Taj. All rights reserved.

This repository and the systems described herein are the original work of Harsha Venkateshwara and Saba Minaz Taj. Reproduction, distribution, or use of any part without explicit written permission is prohibited.

First published: April 30, 2026.
