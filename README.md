# One-Battle-after-Another

# Military Conflict Analysis through Machine Learning 

This project applies Machine Learning techniques to identify factors influencing the outcome of historical land battles. By analyzing a dataset of confrontations spanning nearly four centuries, the study explores how variables like air superiority, surprise, and military strength correlate with victory, defeat, or draws.

---

## 📊 Project Overview
* **Objective:** Identify key historical variables and patterns that explain the success or failure of military operations.
* **Scope:** 660 land battles occurring between 1600 and 1973.
* **Context:** While modern warfare has evolved, this study focuses on historical context and the influence of adverse conditions on military adaptation.

## 🗄️ Dataset: CDB90
The analysis uses a refined version of the **CDB90**, originally created by the US Army Concepts Analysis Agency in 1990.

* **Total Variables:** 169 factors, ranging from weather and terrain to troop morale and leadership.
* **Data Cleaning:** Used the `CDB90-patched` file to fill gaps in the original dataset.
* **Key Files Used:**
    * `battles.csv`: General strategy and final results.
    * `belligerents.csv`: Troop numbers, objectives, and casualties.
    * `terrain.csv` & `weather.csv`: Environmental conditions.
    * `front_widths.csv`: Combat line widths.

## ⚙️ Methodology & Pre-processing
* **Integration:** Merged tables using `isqno` as the unique ID and `pandas.update` for missing values.
* **Feature Engineering:** * Applied **One-Hot Encoding** to categorical variables.
* Military forces were summed to estimate total strength for attackers and defenders.
* **Data Split:** 80% training and 20% testing.
* **Stratification:** Used `stratify=y` to preserve the distribution of the "Draw" class, which only had 43 samples.
* **Normalization:** `StandardScaler` was applied exclusively for the KNN model.

## 🤖 Machine Learning Models
Three models were implemented and evaluated:

| Model | Accuracy  | Key Parameters |
| :--- | :--- | :--- |
| **Decision Trees** | 67.4% | Entropy criterion, `max_depth=5`, `min_samples_leaf=5`. |
| **K-Nearest Neighbors** | 68.9% | `K=18`, Manhattan distance, uniform weights. |
| **Random Forest** | 70.4% | `n_estimators=300`, `max_depth=15`. |

## 📈 Key Insights
* **Feature Importance (Decision Tree):** Air superiority (`aeroa`) and the element of surprise (`surpaa`) were the primary predictors of victory[cite: 91].
* **Feature Importance (Random Forest):** This model highlighted the defender's human strength (`str_def`), defensive posture (`wofd`), and artillery (`arty_def`) as the most relevant features.
* **Challenges:** The "Draw" (Empate) class remains difficult to predict due to the low number of samples in the dataset.

---

* **Authors:** Pedro Tomás & Samuel Figueira 
* **Department:** Electrical and Computer Engineering (DEEC) 
