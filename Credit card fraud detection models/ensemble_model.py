from functions import *
seme=20
np.random.seed(seme)
torch.manual_seed(seme)
random.seed(seme)

########## IPERPARAMETRI ############

binned_feature = "V11" #feature su cui effettuare il binning per il bilanciamento

model1 = SVC(
    kernel='rbf', 
    probability=True, 
    C=1,  
    gamma=0.01, 
    random_state=seme)

models = [
    ("svm_sigmoid", SVC(
        probability=True, 
        kernel="sigmoid", 
        C=1, 
        gamma=0.05, 
        random_state=seme
    )),
    ("catboost", CatBoostClassifier(
        iterations=500,
        learning_rate=0.1,
        depth=6,
        eval_metric="Logloss",
        random_state=seme,
        verbose=0
    )),   
    ("randomforest", RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=seme, n_jobs=-1))
]

beta = 2
############# CARICAMENTO DATASET ##############

target_column = 'Class'
FEATURES = ["V14","V10","V12","V4","V11","V17","V3","V16","V7","V2","V21","V9","V8","V18","V19","Amount","Time"]
#FEATURES = ["Time","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount"]
train_df_, val_df = stratified_split("credit_card_train.csv", seme, test_size=0.2)

X_train = train_df_[FEATURES].to_numpy()
y_train = train_df_[target_column].to_numpy()

# Applico SMOTE
smote = SMOTE(random_state=seme, sampling_strategy=0.1)
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
train1_idx = get_balanced_indices_bin(X_bin, y_original, np.arange(len(y_original)), seme) # bilanciamento training set
X_train, y_train = X_scaled[train1_idx], y_original[train1_idx]

test_df = pd.read_csv("credit_card_test.csv")
X_test_original=test_df[FEATURES].to_numpy()
true_y_test = test_df[target_column].to_numpy()

################ IDENTIFICO EVENTI DENTRO E FUORI DAL MARGINE DELLA SVM PER ESEGUIRE TRAINING MIRATI
print ("Separazione inside/outside")
model1.fit(X_train, y_train)
inside_margin_train, outside_margin_train = margin_split(model1, X_scaled)
inside_margin_val, outside_margin_val = margin_split(model1, X_val, scaler=scaler)
inside_margin_test, outside_margin_test = margin_split(model1, X_test_original, scaler=scaler)

########################################################
scaler_inside = MinMaxScaler()
scaler_outside = MinMaxScaler()

n_models = len(models)

predictions_all_train = np.zeros((len(X_original), n_models))
predictions_val = np.zeros((len(X_val), n_models))
predictions_test = np.zeros((len(X_test_original), n_models))
X_scaled[inside_margin_train] = scaler_inside.fit_transform(X_original[inside_margin_train].copy())
X_scaled[outside_margin_train] = scaler_outside.fit_transform(X_original[outside_margin_train].copy())

for j, (name, model) in enumerate(models):

    print("training", j+1,"di", n_models, "---", name)

    seme=seme+1
    balanced_inside_idx = get_balanced_indices_bin(X_bin, y_original, inside_margin_train, seme)
    balanced_outside_idx = get_balanced_indices_bin(X_bin, y_original, outside_margin_train, seme)


    #INSIDE MARGIN
    X_train_inside = X_scaled[balanced_inside_idx].copy()
    y_train_inside = y_original[balanced_inside_idx].copy()
    model.fit(X_train_inside, y_train_inside)
    #predictions_all_train[inside_margin_train, j] = model.predict_proba(X_scaled[inside_margin_train] )[:, 1]

    X_val_inside = scaler_inside.transform(X_val[inside_margin_val].copy())
    predictions_val[inside_margin_val, j] = model.predict_proba(X_val_inside)[:, 1]

    X_test_inside = scaler_inside.transform(X_test_original[inside_margin_test].copy())
    predictions_test[inside_margin_test, j] = model.predict_proba(X_test_inside)[:, 1]

    # OUTSIDE MARGIN
    X_train_outside = scaler_outside.transform(X_original[balanced_outside_idx].copy()) 
    y_train_outside = y_original[balanced_outside_idx].copy()
    model.fit(X_train_outside, y_train_outside)
    #predictions_all_train[outside_margin_train, j] = model.predict_proba(X_scaled[outside_margin_train] )[:, 1]

    X_val_outside = scaler_outside.transform(X_val[outside_margin_val].copy())
    predictions_val[outside_margin_val, j] = model.predict_proba(X_val_outside)[:, 1]

    X_test_outside = scaler_outside.transform(X_test_original[outside_margin_test].copy())
    predictions_test[outside_margin_test, j] = model.predict_proba(X_test_outside)[:, 1]


#final_predictions_train = predictions_all_train.mean(axis=1)
final_predictions_val = predictions_val.mean(axis=1)
final_predictions_test = predictions_test.mean(axis=1)

# Ottimizzazione soglia su validation
precision_vals_in, recall_vals_in, thresholds_in = precision_recall_curve(y_val[inside_margin_val], final_predictions_val[inside_margin_val])
fbeta_scores_in = (1 + beta**2) * (precision_vals_in * recall_vals_in) / (beta**2 * precision_vals_in + recall_vals_in + 1e-8)
best_idx_in = np.argmax(fbeta_scores_in)
FILTRO_inside = thresholds_in[best_idx_in]

precision_vals_out, recall_vals_out, thresholds_out = precision_recall_curve(y_val[outside_margin_val], final_predictions_val[outside_margin_val])
fbeta_scores_out = (1 + beta**2) * (precision_vals_out * recall_vals_out) / (beta**2 * precision_vals_out + recall_vals_out + 1e-8)
best_idx_out = np.argmax(fbeta_scores_out)
FILTRO_outside = thresholds_out[best_idx_out]

#models_predictions_train= final_predictions_train.copy()
#models_predictions_train[inside_margin_train] = np.where(final_predictions_train[inside_margin_train] >= FILTRO_inside, 1, 0)
#models_predictions_train[outside_margin_train] = np.where(final_predictions_train[outside_margin_train] >= FILTRO_outside, 1, 0)

models_predictions_val = final_predictions_val.copy()
models_predictions_val[inside_margin_val] = np.where(final_predictions_val[inside_margin_val] >= FILTRO_inside, 1, 0)
models_predictions_val[outside_margin_val] = np.where(final_predictions_val[outside_margin_val] >= FILTRO_outside, 1, 0)

models_predictions = final_predictions_test.copy()
models_predictions[inside_margin_test] = np.where(final_predictions_test[inside_margin_test] >= FILTRO_inside, 1, 0)
models_predictions[outside_margin_test] = np.where(final_predictions_test[outside_margin_test] >= FILTRO_outside, 1, 0)

#evaluate_model_performance(y_original, models_predictions_train, final_predictions_train, set_name="Training")
evaluate_model_performance(y_val, models_predictions_val, final_predictions_val, set_name="Validation")
evaluate_model_performance(true_y_test, models_predictions, final_predictions_test, set_name="Test")
