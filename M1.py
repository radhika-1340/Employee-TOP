# Enhanced Survival Analysis Pipeline with XGB AFT, IBS (with IPCW), C-index, Calibration, and Visuals

import pandas as pd
import numpy as np
import xgboost as xgb
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import seaborn as sns

# 1. Preprocessing
def preprocess_data(df):
    df = df.copy()
    for col in df.select_dtypes(include='object').columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    return df

# 2. Winsorization
def winsorize_columns(df, columns):
    for col in columns:
        lower = df[col].quantile(0.01)
        upper = df[col].quantile(0.99)
        df[col] = df[col].clip(lower=lower, upper=upper)
    return df

# 3. Compute bounds for AFT
def compute_bounds(df, max_duration=365):
    df = df.copy()
    df['lower_bound'] = np.where(df['event'] == 1, df['duration'], max_duration)
    df['upper_bound'] = np.where(df['event'] == 1, df['duration'], max_duration)
    return df

# 4. Split train and OOT
def split_data(df, oot_flag_col='oot_flag'):
    train_df = df[df[oot_flag_col] == 0]
    oot_df = df[df[oot_flag_col] == 1]
    return train_df, oot_df

# 5. Train XGB AFT

def train_xgb_aft(X, y_lower, y_upper, params):
    dtrain = xgb.DMatrix(X)
    dtrain.set_float_info('label_lower_bound', y_lower)
    dtrain.set_float_info('label_upper_bound', y_upper)
    model = xgb.train(params, dtrain, num_boost_round=100)
    return model

# 6. Predict mu (log survival time)
def predict_mu(model, X):
    dtest = xgb.DMatrix(X)
    mu = model.predict(dtest)
    return mu

# 7. IBS with IPCW (Kaplan-Meier based weighting)
def calculate_ibs_ipcw(y_time, y_event, mu_pred, time_grid):
    kmf = KaplanMeierFitter()
    kmf.fit(y_time, event_observed=y_event)
    
    weights = np.interp(time_grid, kmf.survival_function_.index, kmf.survival_function_['KM_estimate'])
    weights = np.clip(weights, 1e-6, 1)

    pred_surv = np.exp(-np.outer(np.exp(mu_pred), time_grid))
    km_curve = kmf.survival_function_at_times(time_grid).values

    sq_diff = np.square(pred_surv.mean(axis=0) - km_curve)
    ibs_ipcw = np.sum(sq_diff / weights) / len(time_grid)
    return ibs_ipcw

# 8. C-index

def calculate_cindex(y_time, y_event, pred_mu):
    return concordance_index(y_time, -pred_mu, y_event)

# 9. Calibration Plot at specific horizon
def plot_calibration(y_time, y_event, mu_pred, horizon=180):
    probs = np.exp(-np.exp(mu_pred) * horizon)
    observed = (y_time <= horizon) & (y_event == 1)
    pred_prob, true_prob = calibration_curve(observed, probs, n_bins=10)
    plt.figure(figsize=(6, 5))
    plt.plot(true_prob, pred_prob, marker='o')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title(f"Calibration Curve at {horizon} days")
    plt.xlabel("Observed Proportion")
    plt.ylabel("Predicted Probability")
    plt.grid(True)
    plt.show()

# 10. Survival Curve Comparison

def plot_km_vs_model(y_time, y_event, mu_pred):
    kmf = KaplanMeierFitter()
    kmf.fit(y_time, y_event)
    time_grid = np.arange(1, 366)
    model_curve = np.exp(-np.exp(mu_pred).mean() * time_grid)

    plt.figure(figsize=(8, 5))
    plt.step(kmf.survival_function_.index, kmf.survival_function_['KM_estimate'], label='Kaplan-Meier', color='blue')
    plt.plot(time_grid, model_curve, label='XGB AFT (avg)', color='red')
    plt.title("Survival Curve Comparison")
    plt.xlabel("Days")
    plt.ylabel("Survival Probability")
    plt.legend()
    plt.grid(True)
    plt.show()

# 11. Full Pipeline

def run_pipeline(df):
    df = preprocess_data(df)
    df = winsorize_columns(df, df.select_dtypes(include='number').columns.difference(['duration', 'event']))
    df = compute_bounds(df)

    train_df, oot_df = split_data(df)
    features = df.columns.difference(['duration', 'event', 'lower_bound', 'upper_bound', 'oot_flag'])
    X_train = train_df[features]
    y_lower = train_df['lower_bound']
    y_upper = train_df['upper_bound']

    X_oot = oot_df[features]
    y_oot_time = oot_df['duration']
    y_oot_event = oot_df['event']

    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 4, 5],
        'aft_loss_distribution_scale': [1.0, 5.0, 10.0],
        'aft_loss_distribution': ['logistic', 'extreme']
    }

    best_model = None
    best_ibs = np.inf

    for d in param_grid['aft_loss_distribution']:
        for s in param_grid['aft_loss_distribution_scale']:
            for lr in param_grid['learning_rate']:
                for md in param_grid['max_depth']:
                    params = {
                        'objective': 'survival:aft',
                        'eval_metric': 'aft-nloglik',
                        'tree_method': 'hist',
                        'aft_loss_distribution': d,
                        'aft_loss_distribution_scale': s,
                        'learning_rate': lr,
                        'max_depth': md,
                        'verbosity': 0
                    }
                    model = train_xgb_aft(X_train, y_lower, y_upper, params)
                    mu_pred = predict_mu(model, X_oot)
                    ibs = calculate_ibs_ipcw(y_oot_time, y_oot_event, mu_pred, np.arange(1, 366, 5))
                    if ibs < best_ibs:
                        best_ibs = ibs
                        best_model = model
                        best_mu = mu_pred
                        best_params = params

    print(f"Best Params: {best_params}")
    print(f"Best IBS (IPCW): {best_ibs:.4f}")
    print(f"C-Index: {calculate_cindex(y_oot_time, y_oot_event, best_mu):.4f}")

    plot_km_vs_model(y_oot_time, y_oot_event, best_mu)
    plot_calibration(y_oot_time, y_oot_event, best_mu, horizon=180)

    return best_model, best_params, best_ibs

# Example usage:
# df = pd.read_csv("your_data.csv")
# model, params, ibs = run_pipeline(df)
