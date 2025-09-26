from functions import *
seme=20
np.random.seed(seme)
torch.manual_seed(seme)
random.seed(seme)

########## IPERPARAMETRI ############

binned_feature = "V11" #feature su cui effettuare il binning per il bilanciamento

model = CatBoostClassifier(
        iterations=800,
        learning_rate=0.1,
        depth=6,
        eval_metric="Logloss",
        random_state=seme,
        verbose=0
    )

beta = 2
############# CARICAMENTO DATASET ##############

target_column = 'Class'
FEATURES = ["V14","V10","V12","V4","V11","V17","V3","V16","V7","V2","V21","V9","V8","V18","V19","Amount","Time"]
#FEATURES = ["Time","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount"]
train_df_, val_df = stratified_split("credit_card_train.csv", seme, test_size=0.2)

X_train = train_df_[FEATURES]
y_train = train_df_[target_column]

smote = SMOTE(random_state=seme, sampling_strategy=0.3)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
train_df = pd.DataFrame(X_resampled, columns=FEATURES)
train_df[target_column] = y_resampled

print("Distribuzione originale:", dict(zip(*np.unique(y_train, return_counts=True))))
print("Distribuzione dopo oversampling:", dict(zip(*np.unique(y_resampled, return_counts=True))))

X_original = train_df[FEATURES].to_numpy()
X_bin = train_df[binned_feature].to_numpy()
y_original = train_df[target_column].to_numpy() 
X_val = val_df[FEATURES].to_numpy()
y_val = val_df[target_column].to_numpy()

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_original.copy()) 
train1_idx = get_balanced_indices_bin(X_bin, y_original, np.arange(len(y_original)), seme) # <-- bilanciamento training set con bin
#train1_idx = get_balanced_indices(y_original, np.arange(len(y_original)), seme) # <-- bilanciamento con selezione casuale negativi
X_train, y_train = X_original[train1_idx], y_original[train1_idx]

test_df = pd.read_csv("credit_card_test.csv")
X_test_original=test_df[FEATURES].to_numpy()
true_y_test = test_df[target_column].to_numpy()
X_train = scaler.transform(X_train.copy()) 

model.fit(X_train, y_train)
X_val = scaler.transform(X_val.copy())
X_test = scaler.transform(X_test_original.copy())
final_predictions_train = model.predict_proba(X_train)[:, 1]
final_predictions_val =  model.predict_proba(X_val)[:, 1]
final_predictions_test =  model.predict_proba(X_test)[:, 1]

# Ottimizzazione soglia su validation
precision_vals_in, recall_vals_in, thresholds_in = precision_recall_curve(y_val, final_predictions_val)
fbeta_scores_in = (1 + beta**2) * (precision_vals_in * recall_vals_in) / (beta**2 * precision_vals_in + recall_vals_in + 1e-8)
best_idx_in = np.argmax(fbeta_scores_in)
FILTRO_prob = thresholds_in[best_idx_in]


models_predictions_train= final_predictions_train.copy()
models_predictions_train = np.where(final_predictions_train >= FILTRO_prob, 1, 0)

models_predictions_val= final_predictions_val.copy()
models_predictions_val = np.where(final_predictions_val >= FILTRO_prob, 1, 0)

models_predictions_test= final_predictions_test.copy()
models_predictions_test = np.where(final_predictions_test >= FILTRO_prob, 1, 0)

#evaluate_model_performance(y_original, models_predictions_train, final_predictions_train, set_name="Training")
evaluate_model_performance(y_val, models_predictions_val, final_predictions_val, set_name="Validation")
evaluate_model_performance(true_y_test, models_predictions_test, final_predictions_test, set_name="Test")
