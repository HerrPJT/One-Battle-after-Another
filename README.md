# One-Battle-after-Another

# [cite_start]Military Conflict Analysis through Machine Learning [cite: 1, 2]

[cite_start]This project applies Machine Learning techniques to identify factors influencing the outcome of historical land battles[cite: 6]. [cite_start]By analyzing a dataset of confrontations spanning nearly four centuries, the study explores how variables like air superiority, surprise, and military strength correlate with victory, defeat, or draws[cite: 6, 46].

---

## 📊 Project Overview
* [cite_start]**Objective:** Identify key historical variables and patterns that explain the success or failure of military operations[cite: 5, 6].
* [cite_start]**Scope:** 660 land battles occurring between 1600 and 1973[cite: 9].
* [cite_start]**Context:** While modern warfare has evolved, this study focuses on historical context and the influence of adverse conditions on military adaptation[cite: 7].

## 🗄️ Dataset: CDB90
[cite_start]The analysis uses a refined version of the **CDB90**, originally created by the US Army Concepts Analysis Agency in 1990[cite: 9].

* [cite_start]**Total Variables:** 169 factors, ranging from weather and terrain to troop morale and leadership[cite: 9].
* [cite_start]**Data Cleaning:** Used the `CDB90-patched` file to fill gaps in the original dataset[cite: 12, 51].
* **Key Files Used:**
    * [cite_start]`battles.csv`: General strategy and final results[cite: 13].
    * [cite_start]`belligerents.csv`: Troop numbers, objectives, and casualties[cite: 14].
    * [cite_start]`terrain.csv` & `weather.csv`: Environmental conditions[cite: 15].
    * [cite_start]`front_widths.csv`: Combat line widths[cite: 16].

## ⚙️ Methodology & Pre-processing
* [cite_start]**Integration:** Merged tables using `isqno` as the unique ID and `pandas.update` for missing values[cite: 50, 52].
* [cite_start]**Feature Engineering:** * Applied **One-Hot Encoding** to categorical variables[cite: 64].
    * [cite_start]Military forces were summed to estimate total strength for attackers and defenders[cite: 54].
* [cite_start]**Data Split:** 80% training and 20% testing[cite: 68].
* [cite_start]**Stratification:** Used `stratify=y` to preserve the distribution of the "Draw" class, which only had 43 samples[cite: 69].
* [cite_start]**Normalization:** `StandardScaler` was applied exclusively for the KNN model[cite: 70].

## 🤖 Machine Learning Models
[cite_start]Three models were implemented and evaluated[cite: 72]:

| Model | [cite_start]Accuracy [cite: 84] | Key Parameters |
| :--- | :--- | :--- |
| **Decision Trees** | 67.4% | [cite_start]Entropy criterion, `max_depth=5`, `min_samples_leaf=5`[cite: 74, 75]. |
| **K-Nearest Neighbors** | 68.9% | [cite_start]`K=18`, Manhattan distance, uniform weights[cite: 77, 78]. |
| **Random Forest** | 70.4% | [cite_start]`n_estimators=300`, `max_depth=15`[cite: 81]. |

## 📈 Key Insights
* [cite_start]**Feature Importance (Decision Tree):** Air superiority (`aeroa`) and the element of surprise (`surpaa`) were the primary predictors of victory[cite: 91].
* [cite_start]**Feature Importance (Random Forest):** This model highlighted the defender's human strength (`str_def`), defensive posture (`wofd`), and artillery (`arty_def`) as the most relevant features[cite: 98].
* [cite_start]**Challenges:** The "Draw" (Empate) class remains difficult to predict due to the low number of samples in the dataset[cite: 85, 95].

---

[cite_start]**Authors:** Pedro Tomás & Samuel Figueira [cite: 2]
[cite_start]**Department:** Electrical and Computer Engineering (DEEC) [cite: 2]
