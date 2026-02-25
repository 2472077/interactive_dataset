import os
import os
import io
import json
import time
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder

# Configuration (expects Backend/config.py to expose STORAGE_DIR)
from config import STORAGE_DIR


class LocalStorage:
    """Simple local storage abstraction for uploaded files.

    Methods:
    - save(fileobj, filename) -> returns storage_path (absolute)
    - list() -> list of files
    - load(path) -> bytes
    """

    def __init__(self, storage_dir: str = None):
        self.storage_dir = storage_dir or STORAGE_DIR
        os.makedirs(self.storage_dir, exist_ok=True)

    def _abs_path(self, name: str) -> str:
        return os.path.abspath(os.path.join(self.storage_dir, name))

    def save(self, fileobj, filename: str) -> str:
        ts = int(time.time())
        safe_name = f"{ts}_{os.path.basename(filename)}"
        path = self._abs_path(safe_name)
        mode = "wb"
        # fileobj may be bytes, BytesIO, or an open file-like
        if isinstance(fileobj, (bytes, bytearray)):
            data = fileobj
        else:
            fileobj.seek(0)
            data = fileobj.read()
        with open(path, mode) as fh:
            if isinstance(data, str):
                data = data.encode("utf-8")
            fh.write(data)
        return path

    def load(self, path: str) -> bytes:
        with open(path, "rb") as fh:
            return fh.read()

    def list(self):
        return [os.path.join(self.storage_dir, f) for f in os.listdir(self.storage_dir)]


# instantiate a module-level storage used by the API (main.py expects `storage` variable)
storage = LocalStorage()


# -----------------------------
# Data cleaning helpers
# -----------------------------

def is_numeric_like(series: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(series):
        return True
    s_num = pd.to_numeric(series, errors="coerce")
    return bool(s_num.notna().mean() >= 0.7)


def infer_policies_for_df(df: pd.DataFrame) -> dict:
    policies = {}
    for col in df.columns:
        s = df[col]
        numeric_like = bool(is_numeric_like(s))  # Convert to Python bool
        pol = {"numeric_like": numeric_like}
        if numeric_like:
            s_num = pd.to_numeric(s, errors="coerce").dropna()
            pol["zero_as_missing"] = False
            pol["nonnegative_only"] = False
            if not s_num.empty:
                # Explicitly convert numpy bool to Python bool
                pol["zero_as_missing"] = bool((s_num == 0).mean() > 0.5)
                pol["nonnegative_only"] = bool((s_num >= 0).mean() >= 0.95)
        policies[col] = pol
    return policies


# --- Missing value handlers ---
def apply_zero_as_missing(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    try:
        df[col] = df[col].replace(0, pd.NA)
    except Exception:
        pass
    return df


def enforce_nonnegative_as_nan(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    try:
        s = pd.to_numeric(df[col], errors="coerce")
        df.loc[s < 0, col] = pd.NA
    except Exception:
        pass
    return df


def mean(df: pd.DataFrame, col: str):
    return df[col].mean()


def median(df: pd.DataFrame, col: str):
    return df[col].median()


def mode(df: pd.DataFrame, col: str):
    try:
        return df[col].mode(dropna=True)[0]
    except Exception:
        return df[col].dropna().iloc[0] if df[col].dropna().shape[0] else pd.NA


def custom(df: pd.DataFrame, col: str, custom_value):
    df = df.copy()
    df[col] = df[col].fillna(custom_value)
    return df[col]


def drop_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna()


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates()


# --- Text standardization ---
def lower(df: pd.DataFrame, col: str):
    return df[col].astype(str).str.lower()


def upper(df: pd.DataFrame, col: str):
    return df[col].astype(str).str.upper()


def title(df: pd.DataFrame, col: str):
    return df[col].astype(str).str.title()


def date_time(df_cleaned, col):
    def parse_date(val):
        if pd.isna(val):
            return None
        for fmt in ('%Y-%m-%d', '%Y-%d-%m','%d-%m-%Y', '%m-%d-%Y', '%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d'):
            try:
                return pd.to_datetime(val, format=fmt).strftime('%Y-%m-%d')
            except:
                continue
        return None
    
    return df_cleaned[col].apply(parse_date)


# --- One-hot encoding ---
def One_hot_encoding(df_cleaned: pd.DataFrame, onehot_columns: list) -> pd.DataFrame:
    try:
        encoder = OneHotEncoder(sparse_output=False, drop='first')
    except TypeError:
        encoder = OneHotEncoder(sparse=False, drop='first')
    encoded_data = encoder.fit_transform(df_cleaned[onehot_columns])
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(onehot_columns))
    df_cleaned = df_cleaned.drop(onehot_columns, axis=1)
    df_cleaned = pd.concat([df_cleaned.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
    return df_cleaned


def normalize(df_cleaned: pd.DataFrame, col: str):
    s = pd.to_numeric(df_cleaned[col], errors="coerce")
    min_val = s.min()
    max_val = s.max()
    if pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
        df_cleaned[col] = 0.0
    else:
        df_cleaned[col] = (s - min_val) / (max_val - min_val)
    return df_cleaned[col]


def outliers(df: pd.DataFrame, col: str) -> pd.DataFrame:
    s = pd.to_numeric(df[col], errors="coerce")
    s_nonnull = s.dropna()
    if s_nonnull.empty:
        return df
    q_low, q_high = s_nonnull.quantile([0.01, 0.99])
    trimmed = s_nonnull[(s_nonnull >= q_low) & (s_nonnull <= q_high)]
    if trimmed.empty:
        trimmed = s_nonnull
    Q1 = trimmed.quantile(0.25)
    Q3 = trimmed.quantile(0.75)
    IQR = Q3 - Q1
    if pd.isna(Q1) or pd.isna(Q3) or IQR == 0:
        lower, upper = q_low, q_high
    else:
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
    mask = s.between(lower, upper)
    nonneg_share = (s_nonnull >= 0).mean()
    if nonneg_share >= 0.95:
        mask = mask & (s >= 0)
    return df[mask.fillna(False)]


# --- Modeling helpers ---
def fit_linear_regression(df_cleaned: pd.DataFrame, reg_cols: list):
    reg = linear_model.LinearRegression()
    X = df_cleaned[[reg_cols[0], reg_cols[1]]]
    y = df_cleaned[reg_cols[2]]
    model = reg.fit(X, y)
    return model


def lin_regressor(df_cleaned: pd.DataFrame, reg_cols: list, values: list):
    model = fit_linear_regression(df_cleaned, reg_cols)
    return model.predict([values])


def fit_poly_regression(df_cleaned: pd.DataFrame, reg_cols: list, degree: int):
    poly = PolynomialFeatures(degree)
    X = df_cleaned[[reg_cols[0], reg_cols[1]]]
    X_poly = poly.fit_transform(X)
    model = linear_model.LinearRegression().fit(X_poly, df_cleaned[[reg_cols[2]]])
    return model, poly, X_poly


def poly_regressor(df_cleaned: pd.DataFrame, reg_cols: list, degree: int, values: list):
    model, poly, _ = fit_poly_regression(df_cleaned, reg_cols, degree)
    return model.predict(poly.transform([values]))


def fit_knn(df_cleaned: pd.DataFrame, features: list, target: str, k: int):
    X = df_cleaned[features].values
    y = df_cleaned[target].values
    knn = KNeighborsClassifier(n_neighbors=k)
    model = knn.fit(X, y)
    return model, X, y

def drop_missing_values(df_cleaned):
    return df_cleaned.dropna()

# def mean(df_cleaned, col):
#     return df_cleaned[col].fillna(df_cleaned[col].mean())

# def median(df_cleaned, col):
#     return df_cleaned[col].fillna(df_cleaned[col].median())

# def mode(df_cleaned, col):
#     return df_cleaned[col].fillna(df_cleaned[col].mode()[0])


def mean(df_cleaned, col):
    s = pd.to_numeric(df_cleaned[col], errors="coerce")
    return s.fillna(s.mean())

def median(df_cleaned, col):
    s = pd.to_numeric(df_cleaned[col], errors="coerce")
    return s.fillna(s.median())

def mode(df_cleaned, col):
    s = df_cleaned[col]
    m = s.mode(dropna=True)
    if m.empty:
        # nothing to fill with; return unchanged
        return s
    return s.fillna(m.iloc[0])


def custom(df_cleaned, col, custom_value):
    return df_cleaned[col].fillna(custom_value)

def drop_duplicates(df_cleaned):
    return df_cleaned.drop_duplicates()

### One Hot Encoding ###
# def One_hot_encoding(df_cleaned, onehot_columns):
#     encoder = OneHotEncoder(sparse_output=False, drop='first')
#     encoded_data = encoder.fit_transform(df_cleaned[onehot_columns])
#     encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(onehot_columns))
#     df_cleaned = df_cleaned.drop(onehot_columns, axis=1)
#     df_cleaned = pd.concat([df_cleaned.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
#     return df_cleaned


def One_hot_encoding(df_cleaned, onehot_columns):
    try:
        encoder = OneHotEncoder(sparse_output=False, drop='first')
    except TypeError:
        # Older sklearn versions use 'sparse' param name
        encoder = OneHotEncoder(sparse=False, drop='first')
    encoded_data = encoder.fit_transform(df_cleaned[onehot_columns])
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(onehot_columns))
    df_cleaned = df_cleaned.drop(onehot_columns, axis=1)
    df_cleaned = pd.concat([df_cleaned.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
    return df_cleaned


# def normalize(df_cleaned, col):
#     min_val = df_cleaned[col].min()
#     max_val = df_cleaned[col].max()
#     df_cleaned[col] = (df_cleaned[col] - min_val) / (max_val - min_val)
#     return df_cleaned[col]


def normalize(df_cleaned, col):
    s = pd.to_numeric(df_cleaned[col], errors="coerce")
    min_val = s.min()
    max_val = s.max()
    if pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
        df_cleaned[col] = 0.0  # avoid divide-by-zero; choose a stable default
    else:
        df_cleaned[col] = (s - min_val) / (max_val - min_val)
    return df_cleaned[col]



def outliers(df, col):
    # Coerce to numeric
    s = pd.to_numeric(df[col], errors="coerce")

    s_nonnull = s.dropna()
    if s_nonnull.empty:
        return df  # nothing to filter

    # Trim extremes to reduce leverage of absurd values (e.g., 9,999,999)
    q_low, q_high = s_nonnull.quantile([0.01, 0.99])
    trimmed = s_nonnull[(s_nonnull >= q_low) & (s_nonnull <= q_high)]
    if trimmed.empty:
        trimmed = s_nonnull

    Q1 = trimmed.quantile(0.25)
    Q3 = trimmed.quantile(0.75)
    IQR = Q3 - Q1

    if pd.isna(Q1) or pd.isna(Q3) or IQR == 0:
        # Fall back to quantile bounds
        lower, upper = q_low, q_high
    else:
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

    mask = s.between(lower, upper, inclusive="both")

    # Heuristic: if column is mostly non-negative, drop negatives
    nonneg_share = (s_nonnull >= 0).mean()
    if nonneg_share >= 0.95:
        mask = mask & (s >= 0)

    # Return filtered rows; keep original dtypes and cols
    return df[mask.fillna(False)]


def fit_linear_regression(df_cleaned, reg_cols):
    reg = linear_model.LinearRegression()
    var1 = df_cleaned[[reg_cols[0], reg_cols[1]]]
    model = reg.fit(var1, df_cleaned[reg_cols[2]])
    return model

def lin_regressor(df_cleaned, reg_cols, values):
    model = fit_linear_regression(df_cleaned, reg_cols)
    return model.predict(values)

def fit_poly_regression(df_cleaned, reg_cols, degree):
    poly = PolynomialFeatures(degree)
    var1 = df_cleaned[[reg_cols[0], reg_cols[1]]]
    var1_poly = poly.fit_transform(var1)
    model = linear_model.LinearRegression().fit(var1_poly, df_cleaned[[reg_cols[2]]])
    return model, poly, var1_poly

def poly_regressor(df_cleaned, reg_cols, degree, values):
    model, poly, _ = fit_poly_regression(df_cleaned, reg_cols, degree)
    return model.predict(poly.fit_transform(values))

def fit_knn(df_cleaned, features, target, k):
    X = df_cleaned[features].values
    y = df_cleaned[target].values
    knn = KNeighborsClassifier(n_neighbors=k)
    model = knn.fit(X, y)
    return model, X, y

