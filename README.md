# Credit Card Fraud Detection

## Abstract
This project explores fraud detection on the **Kaggle Credit Card Fraud Detection dataset**.  
The dataset is highly imbalanced, with frauds representing only **0.172%** of transactions.  
The goal was to design and compare machine learning pipelines able to **handle severe class imbalance** while achieving robust generalization.  

Two main approaches were implemented:  
- A **single CatBoost classifier** with optimized sampling strategies.  
- An **ensemble model** combining SVM, CatBoost, and Random Forest, with a margin-based specialization strategy.  

On the test set, the ensemble model reached **Recall ≈ 0.816**, **Precision ≈ 0.869**, **AP ≈ 0.828** and **AUC ≈ 0.984**, demonstrating effective handling of rare-event classification.  

---

## Project Overview
The workflow followed these steps:  

1. **Data preparation**  
   - Stratified train/test split (80/20).    
   - Additional validation split from the training set (80/20).
   - Exploratory analysis: feature distributions and correlation heatmaps.
   - Oversampling the minority class only in the training set with **SMOTE** (10–30% of negatives). 
   - Custom negative sampling: random undersampling vs. bin-based undersampling.    
   - Feature scaling with MinMaxScaler.    

2. **Feature selection**  
   - Focus on the most informative features (identified through correlation analysis and Random Forest feature importance) + `Time` and `Amount`.

3. **Models**  
   - **Single model**: CatBoost(iterations=800, learning_rate=0.1, depth=6, eval_metric="Logloss") 
   - **Ensemble model**:  
     - SVM (rbf kernel) used to separate samples inside/outside the margin.  
     - CatBoost, SVM (sigmoid kernel), and Random Forest trained separately on subsets.  
     - Final prediction by averaging model probabilities and applying class-specific thresholds.  

4. **Hyperparameter tuning**  
   - Performed on the **validation set** (single fold).  
   - **No k-fold cross-validation** was used due to computational constraints, but this could improve robustness of hyperparameter tuning in future work.  

5. **Evaluation**  
   - Optimized thresholds via **F2-score** on validation.  
   - Metrics: Confusion Matrix, Precision, Recall, Accuracy, Average Precision (AP), AUC.  

---

## Results

### Single Model (CatBoost, SMOTE 0.3)

- **Validation** → Precision 0.93 | Recall 0.83 | AUC 0.97 | AP 0.84  

|               | Predicted 0 | Predicted 1 |
|---------------|-------------|-------------|
| **True 0**    | 45485       | 5           |
| **True 1**    | 13          | 66          |

- **Test**       → Precision 0.93 | Recall 0.79

|               | Predicted 0 | Predicted 1 |
|---------------|-------------|-------------|
| **True 0**    | 56858       | 6           |
| **True 1**    | 21          | 77          |


#### Precision-Recall Curve - Single Model
![Precision Recall Curve](Credit%20card%20fraud%20detection%20models/Generated%20plots/Precision-Recall%20Curve_single.png)

#### ROC Curve - Single Model
![ROC Curve](Credit%20card%20fraud%20detection%20models/Generated%20plots/ROC%20curve_single.png)

### Ensemble Model (SVM + CatBoost + Random Forest, SMOTE 0.1)

- **Validation** → Precision 0.87 | Recall 0.85 | AUC 0.99 | AP 0.84 

|               | Predicted 0 | Predicted 1 |
|---------------|-------------|-------------|
| **True 0**    | 45480       | 10          |
| **True 1**    | 12          | 67          |

- **Test**       → Precision 0.87 | Recall 0.82

|               | Predicted 0 | Predicted 1 |
|---------------|-------------|-------------|
| **True 0**    | 56852       | 12          |
| **True 1**    | 18          | 80          |

#### Precision-Recall Curve - Ensemble Model
![Precision Recall Curve](Credit%20card%20fraud%20detection%20models/Generated%20plots/Precision-Recall%20Curve_ensemble.png)

#### ROC Curve - Ensemble Model
![ROC Curve](Credit%20card%20fraud%20detection%20models/Generated%20plots/ROC%20curve_ensemble.png)

**Trade-off**: the ensemble improved recall but required significantly higher computational cost, while CatBoost provided the best balance between performance and efficiency.  

---

## Repository Contents
- **functions.py** – helper functions for balancing, inside/outside split, evaluation, and plotting  
- **single_model.py** – training and evaluation of the CatBoost classifier  
- **ensemble_model.py** – training and evaluation of the SVM+CatBoost+Random Forest ensemble  
- **generate_plots** – distributions, and correlation matrix generation  
- **Generated plots** – feature distributions, heatmaps, PR and ROC curves  

---

## Usage

1. Download the dataset `creditcard.csv` from Kaggle:  
   [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/)  
   Place the file in the **root** of the repository.  

2. Create train and test sets by "running train_test_creation.py": This generates "credit_card_train.csv" and "credit_card_test.csv". The split is stratified: 80% training/validation and 20% test.

3. Run one of the model scripts:
   - "single_model.py" → trains and evaluates the CatBoost classifier.
   - "ensemble_model.py" → trains and evaluates the SVM + CatBoost + Random Forest ensemble.
Both scripts print metrics and generate evaluation plots (Confusion Matrix, ROC curve, Precision-Recall curve).

---

## Dependencies
The project was implemented in **Python**, using:  
- **numpy / pandas** – data manipulation and numerical computation  
- **matplotlib / seaborn** – visualizations (distributions, correlations, PR/ROC curves)  
- **scikit-learn** – preprocessing, metrics, Random Forest, SVM, model selection  
- **imblearn (SMOTE)** – synthetic minority oversampling  
- **catboost** – gradient boosting classifier  
- **PyTorch** – used for reproducibility setup and potential extensions  

---

## License
This repository is intended for academic and educational purposes.  
All materials are the intellectual property of the author and may not be copied or redistributed without permission.  