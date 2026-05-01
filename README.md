# final-project-team_bravo6
final-project-team_bravo6 created by GitHub Classroom
# Storm Restoration Prioritization AI + GridGuard

A full-stack machine learning platform for power outage forecasting, restoration prioritization, and critical facility alerting using publicly available datasets.

---

## Problem Statement

Utility companies must decide how to allocate restoration crews during storms. Current approaches rely on manual coordination and reactive decisions, which can delay response and increase outage impact.

This project addresses this gap by building a data-driven decision support system that predicts outage risk, ranks restoration priorities, and identifies critical infrastructure at risk before storms.

---

## Datasets Used

- **EAGLE-I Dataset**  
  ~160 million county-level outage records covering 3,044 counties across the U.S.

- **NOAA Storm Events Dataset**  
  213,000+ storm records categorized into wind, winter, flood, tropical, and other events.

- **OpenStreetMap (OSM)**  
  Power grid infrastructure including substations and transmission lines.

- **HIFLD Dataset**  
  Critical facilities such as hospitals, fire stations, and schools.

---

## System Overview

The platform consists of two main components:

### Storm Restoration Prioritization AI
- Predicts outage probability at the county level  
- Generates a ranked restoration priority queue  

### GridGuard AI
- Builds a grid topology graph  
- Performs k-hop traversal  
- Generates alerts for critical facilities  

---

## Methodology

- Feature Engineering:
  - Temporal features (hour, month)
  - Storm indicators
  - Lag features (1hr, 24hr, 168hr)
  - County vulnerability metrics

- Models:
  - Multi-Task LSTM  
  - Gradient Boosting (best performer)

- Graph Analysis:
  - NetworkX-based grid modeling  
  - BFS traversal for alert generation  

---

## Results

- Best Model: Gradient Boosting  
- ROC-AUC: 0.911  
- PR-AUC: 0.748 (~7.5x improvement over baseline)

Validated on:
- Hurricane Sandy (2012)  
- Tropical Storm Isaias (2020)  

---

## Dashboard Features

- Risk Map (choropleth visualization)
- Priority Queue (ranked restoration list)
- County Drilldown
- Model Results (evaluation metrics)
- GridGuard Alert Map

---

## How to Run

### Clone the repository
git clone <your-repo-link>  
cd storm_restoration_ai  

### Install dependencies
pip install -r requirements.txt  

### Run the application
streamlit run app.py  

---

## Project Structure

storm_restoration_ai/  
│  
├── app.py  
├── requirements.txt  
├── src/  
│   ├── models/  
│   ├── features/  
│   ├── scoring/  
│   └── evaluation/  
├── scripts/  
├── outputs/  
├── notebooks/  
└── data/  

---

## Key Contributions

- Developed an outage prediction system using large-scale public datasets  
- Designed a hybrid ML approach (LSTM + Gradient Boosting)  
- Built a graph-based alert system for critical infrastructure  
- Created an interactive dashboard for real-time decision support  

---

## Limitations

- Grid topology is approximated using public data  
- No real-time weather integration  
- Limited validation across regions  

---

## Future Work

- Integrate real-time weather APIs  
- Use utility-grade grid topology data  
- Expand critical facility coverage  
- Improve model generalization  

---

## Authors

- Harsha Venkateshwara — 50%  
- Saba Minaz Taj — 50%  

---

## License

This project is for academic and research purposes.
