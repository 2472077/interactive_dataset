import io
import uuid
import time
from contextlib import asynccontextmanager
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Header
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import functions
from functions import *
import db_ops
from db_conn import init_db
from nlp_query import classify_intent_with_columns_and_values, execute_intent

# In-memory session store: { session_id: {"df": dataframe, "created_at": timestamp, "policies": dict} }
sessions = {}
SESSION_TIMEOUT = 600  # 10 minutes in seconds


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Initializing database...")
    init_db()
    print("Database initialized. API ready.")
    yield
    # Shutdown
    print("Application shutting down...")


app = FastAPI(title="Dataset Cleaner API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5000", "http://127.0.0.1:5000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple token-based auth store for demonstration: { token: {username, user_id, role, created_at} }
user_sessions = {}
USER_SESSION_TTL = 60 * 60 * 24  # 1 day


def _create_token_for_user(user: dict) -> str:
    token = str(uuid.uuid4())
    user_sessions[token] = {
        "username": user["username"],
        "user_id": user["id"],
        "role": user.get("role", "user"),
        "created_at": time.time(),
    }
    return token


def _get_user_from_token(token: str):
    rec = user_sessions.get(token)
    if not rec:
        return None
    if time.time() - rec["created_at"] > USER_SESSION_TTL:
        del user_sessions[token]
        return None
    return rec


def get_user_from_request(request: Request):
    auth = request.headers.get("Authorization")
    if not auth:
        return None
    if auth.startswith("Bearer "):
        token = auth.split(" ", 1)[1]
        return _get_user_from_token(token)
    return None


def get_df(session_id: str) -> pd.DataFrame:
    if session_id not in sessions:
        raise HTTPException(
            status_code=404, detail="Session not found. Please upload a file first."
        )
    # Check if session has timed out
    elapsed = time.time() - sessions[session_id]["created_at"]
    if elapsed > SESSION_TIMEOUT:
        del sessions[session_id]
        raise HTTPException(
            status_code=410, detail="Session timed out. Please upload your file again."
        )
    return sessions[session_id]["df"].copy()


def save_df(session_id: str, df: pd.DataFrame):
    if session_id in sessions:
        sessions[session_id]["df"] = df
    else:
        sessions[session_id] = {"df": df, "created_at": time.time()}


def df_to_json(df: pd.DataFrame) -> dict:
    import json

    df = df.replace([float("inf"), float("-inf")], None)
    records = json.loads(df.to_json(orient="records"))
    return {"columns": df.columns.tolist(), "data": records, "shape": list(df.shape)}


# ─────────────────────────────────────────────
# Helpers: numeric-like detection & histogram
# ─────────────────────────────────────────────


def is_numeric_like(series: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(series):
        return True
    s_num = pd.to_numeric(series, errors="coerce")
    return bool(s_num.notna().mean() >= 0.7)


def column_histogram(df: pd.DataFrame, col: str, bins: int = 20) -> dict:
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        return {"counts": [], "bins": []}
    counts, bin_edges = np.histogram(s, bins=bins)
    return {"counts": counts.tolist(), "bins": bin_edges.tolist()}


# ─────────────────────────────────────────────
# Upload
# ─────────────────────────────────────────────
@app.post("/upload")
async def upload_file(request: Request, file: UploadFile = File(...)):
    # Require authentication: users and admins can upload
    user = get_user_from_request(request)
    if not user:
        raise HTTPException(
            status_code=401, detail="Authentication required to upload files."
        )

    contents = await file.read()
    try:
        if file.filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        elif file.filename.endswith(".xlsx"):
            df = pd.read_excel(io.BytesIO(contents))
        elif file.filename.endswith(".json"):
            df = pd.read_json(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {e}")

    # Normalize common missing tokens and drop empty rows
    df = df.replace(r"^\s*$", pd.NA, regex=True)
    df = df.replace(["NaN", "N/A", "NULL", "null", "None"], pd.NA)
    df = df.dropna(how="all")

    # 1) Infer policies from the data itself
    policies = infer_policies_for_df(df)

    # 2) Apply the policies generically
    for col, pol in policies.items():
        if pol.get("numeric_like"):
            if pol.get("zero_as_missing"):
                df = apply_zero_as_missing(df, col)
            if pol.get("nonnegative_only"):
                df = enforce_nonnegative_as_nan(df, col)

    # 3) Save session with df and policies
    session_id = str(uuid.uuid4())
    sessions[session_id] = {"df": df, "created_at": time.time(), "policies": policies}

    # Save the original file to storage (local for now)
    try:
        storage_path = storage.save(io.BytesIO(contents), file.filename)
    except Exception as e:
        storage_path = None

    # Create dataset record and attach to session
    ds_id = None
    try:
        if storage_path:
            ds_id = db_ops.create_dataset_record(
                name=file.filename,
                storage_path=storage_path,
                uploaded_by=user.get("user_id"),
                session_id=session_id,
            )
            sessions[session_id]["dataset_id"] = ds_id
            sessions[session_id]["uploaded_by"] = user.get("username")
            # Log the upload action
            db_ops.log_action(
                user.get("user_id"),
                "upload",
                dataset_id=ds_id,
                details=f"Uploaded file: {file.filename}",
            )
    except Exception:
        pass

    return {"session_id": session_id, **df_to_json(df)}


# ─────────────────────────────────────────────
# Auth endpoints + admin views
# ─────────────────────────────────────────────
class AuthRequest(BaseModel):
    username: str
    password: str
    role: Optional[str] = "user"


@app.post("/auth/register")
def auth_register(req: AuthRequest):
    result = db_ops.create_user(req.username, req.password, role=req.role or "user")
    if not result:
        raise HTTPException(status_code=400, detail="Username already exists")
    return {"status": "ok", "message": "User registered"}


@app.post("/auth/login")
def auth_login(req: AuthRequest):
    user = db_ops.verify_user(req.username, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = _create_token_for_user(user)
    return {
        "token": token,
        "username": user["username"],
        "role": user.get("role", "user"),
    }


@app.post("/auth/logout")
def auth_logout(request: Request):
    auth = request.headers.get("Authorization") or ""
    if auth.startswith("Bearer "):
        token = auth.split(" ", 1)[1]
        if token in user_sessions:
            del user_sessions[token]
    return {"status": "ok"}


# Admin-only endpoints for audit and dataset inspection
@app.get("/admin/audit-logs")
def admin_audit_logs(request: Request):
    user = get_user_from_request(request)
    if not user or user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin role required")
    logs = db_ops.list_audit_logs(500)
    out = []
    for l in logs:
        out.append(
            {
                "id": l["id"],
                "user_id": l["user_id"],
                "dataset_id": l["dataset_id"],
                "action": l["action"],
                "details": l["details"],
                "timestamp": l["timestamp"],
            }
        )
    return {"logs": out}


@app.get("/admin/datasets")
def admin_datasets(request: Request):
    user = get_user_from_request(request)
    if not user or user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin role required")
    dss = db_ops.list_datasets()
    out = []
    for d in dss:
        out.append(
            {
                "id": d["id"],
                "name": d["name"],
                "storage_path": d["storage_path"],
                "uploaded_by": d["uploaded_by"],
                "session_id": d["session_id"],
                "uploaded_at": d["uploaded_at"],
            }
        )
    return {"datasets": out}


@app.get("/admin/users")
def admin_users(request: Request):
    user = get_user_from_request(request)
    if not user or user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin role required")
    us = db_ops.list_users()
    out = []
    for u in us:
        out.append(
            {
                "id": u["id"],
                "username": u["username"],
                "role": u["role"],
                "created_at": u["created_at"],
            }
        )
    return {"users": out}


# ─────────────────────────────────────────────
# Column Summary (stats + histogram for numeric-like)
# ─────────────────────────────────────────────
class ColumnSummaryRequest(BaseModel):
    session_id: str
    column: str
    bins: Optional[int] = 20


@app.post("/column-summary")
def column_summary(req: ColumnSummaryRequest):
    df = get_df(req.session_id)
    if req.column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{req.column}' not found.")

    numeric_like = is_numeric_like(df[req.column])
    nulls = int(df[req.column].isna().sum())
    zeros = 0
    stats = {}
    hist = {"counts": [], "bins": []}

    if numeric_like:
        s = pd.to_numeric(df[req.column], errors="coerce")
        zeros = int((s == 0).sum())
        if s.notna().any():
            stats = {
                "min": float(s.min()),
                "max": float(s.max()),
                "mean": float(s.mean()),
                "median": float(s.median()),
                "std": float(s.std(ddof=1)) if s.count() > 1 else 0.0,
            }
        hist = column_histogram(df, req.column, bins=req.bins or 20)

    return {
        "column": req.column,
        "numeric_like": numeric_like,
        "counts": {
            "total": int(len(df)),
            "non_null": int(df[req.column].notna().sum()),
            "nulls": nulls,
            "zeros": zeros,
        },
        "stats": stats,
        "histogram": hist,
    }


# Convenience endpoint to list numeric-like columns
@app.get("/numeric-columns/{session_id}")
def numeric_columns(session_id: str):
    rec = sessions.get(session_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Session not found.")
    pols = rec.get("policies") or infer_policies_for_df(
        rec["df"]
    )  # fallback if missing
    cols = [c for c, p in pols.items() if p.get("numeric_like")]
    return {"numeric_columns": cols}


# ─────────────────────────────────────────────
# Fill Missing Values
# ─────────────────────────────────────────────
class FillRequest(BaseModel):
    session_id: str
    column: str
    method: str  # mean | median | mode | custom
    custom_value: Optional[str] = None


@app.post("/fill-missing")
def fill_missing(request: Request, req: FillRequest):
    record = sessions.get(req.session_id)
    if not record:
        raise HTTPException(
            status_code=404, detail="Session not found. Please upload a file first."
        )

    df = record["df"].copy()
    col = req.column
    if col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{col}' not found.")

    # ✅ ALWAYS coerce to numeric before mean/median
    if req.method in ("mean", "median"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    try:
        if req.method == "mean":
            df[col] = mean(df, col)
        elif req.method == "median":
            df[col] = median(df, col)
        elif req.method == "mode":
            df[col] = mode(df, col)
        elif req.method == "custom":
            if req.custom_value is None:
                raise HTTPException(
                    status_code=400, detail="custom_value required for custom fill."
                )
            try:
                df[col] = custom(df, col, float(req.custom_value))
            except ValueError:
                df[col] = custom(df, col, req.custom_value)
        else:
            raise HTTPException(
                status_code=400, detail="method must be mean | median | mode | custom"
            )
    except TypeError as e:
        raise HTTPException(status_code=400, detail=f"Column '{col}' not numeric: {e}")

    save_df(req.session_id, df)

    # Audit log
    user = get_user_from_request(request)
    try:
        ds = db_ops.get_dataset_by_session_id(req.session_id)
        ds_id = ds["id"] if ds else None
        if user:
            db_ops.log_action(
                user.get("user_id"),
                f"fill_missing:{col}:{req.method}",
                dataset_id=ds_id,
                details=f"method={req.method}",
            )
    except Exception:
        pass

    result = df_to_json(df)
    # Attach histogram if numeric-like
    if is_numeric_like(df[col]):
        result["histogram"] = column_histogram(df, col)
    return result


# ─────────────────────────────────────────────
# Drop Missing Values
# ─────────────────────────────────────────────
class SessionRequest(BaseModel):
    session_id: str


@app.post("/drop-missing")
def drop_missing(request: Request, req: SessionRequest):
    df = get_df(req.session_id)
    df = drop_missing_values(df)
    save_df(req.session_id, df)
    # Audit
    user = get_user_from_request(request)
    try:
        ds = db_ops.get_dataset_by_session_id(req.session_id)
        ds_id = ds["id"] if ds else None
        if user:
            db_ops.log_action(
                user.get("user_id"),
                "drop_missing",
                dataset_id=ds_id,
                details="dropped rows with missing values",
            )
    except Exception:
        pass
    return df_to_json(df)


@app.post("/verify-session")
def verify_session(req: SessionRequest):
    if req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    elapsed = time.time() - sessions[req.session_id]["created_at"]
    if elapsed > SESSION_TIMEOUT:
        del sessions[req.session_id]
        raise HTTPException(status_code=410, detail="Session timed out.")
    return {"status": "alive"}


# ─────────────────────────────────────────────
# Remove Duplicates
# ─────────────────────────────────────────────
@app.post("/remove-duplicates")
def remove_duplicates(request: Request, req: SessionRequest):
    df = get_df(req.session_id)
    df = drop_duplicates(df)
    save_df(req.session_id, df)
    # Audit
    user = get_user_from_request(request)
    try:
        ds = db_ops.get_dataset_by_session_id(req.session_id)
        ds_id = ds["id"] if ds else None
        if user:
            db_ops.log_action(
                user.get("user_id"),
                "remove_duplicates",
                dataset_id=ds_id,
                details="removed duplicate rows",
            )
    except Exception:
        pass
    return df_to_json(df)


# ─────────────────────────────────────────────
# Standardization
# ─────────────────────────────────────────────
class StandardizeRequest(BaseModel):
    session_id: str
    column: str
    method: str  # lowercase | uppercase | titlecase | date | round
    decimal_places: Optional[int] = 2


@app.post("/standardize")
def standardize(request: Request, req: StandardizeRequest):
    df = get_df(req.session_id)
    col = req.column
    if col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{col}' not found.")

    if req.method == "lowercase":
        df[col] = lower(df, col)
    elif req.method == "uppercase":
        df[col] = upper(df, col)
    elif req.method == "titlecase":
        df[col] = title(df, col)
    elif req.method == "date":
        df[col] = date_time(df, col)
    elif req.method == "round":
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise HTTPException(
                status_code=400, detail=f"Column '{col}' is not numeric."
            )
        df[col] = df[col].round(req.decimal_places)
    else:
        raise HTTPException(status_code=400, detail="Invalid method.")

    save_df(req.session_id, df)
    # Audit
    user = get_user_from_request(request)
    try:
        ds = db_ops.get_dataset_by_session_id(req.session_id)
        ds_id = ds["id"] if ds else None
        if user:
            db_ops.log_action(
                user.get("user_id"),
                f"standardize:{col}:{req.method}",
                dataset_id=ds_id,
                details=str(req.dict()),
            )
    except Exception:
        pass
    return df_to_json(df)


# ─────────────────────────────────────────────
# Remove Outliers (enhanced to return histogram)
# ─────────────────────────────────────────────
class ColumnRequest(BaseModel):
    session_id: str
    column: str
    nonnegative_only: Optional[bool] = False


def outliers(df, col):
    """
    Generic robust outlier filter:
    - Coerce to numeric (non-parsable -> NaN)
    - Trim 1%-99% to reduce leverage of extreme values
    - Compute IQR on trimmed sample; filter to [Q1-1.5*IQR, Q3+1.5*IQR]
    - If column is mostly non-negative, remove negatives outright
    """
    s = pd.to_numeric(df[col], errors="coerce")
    s_nonnull = s.dropna()
    if s_nonnull.empty:
        return df
    # Trim extremes first
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
    mask = s.between(lower, upper, inclusive="both")
    # If >=95% of observed values are non-negative, enforce non-negativity
    nonneg_share = (s_nonnull >= 0).mean()
    if nonneg_share >= 0.95:
        mask = mask & (s >= 0)
    return df[mask.fillna(False)]


@app.post("/remove-outliers")
def remove_outliers(request: Request, req: ColumnRequest):
    df = get_df(req.session_id)
    if req.column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{req.column}' not found.")
    # Always coerce to numeric for outlier math
    df[req.column] = pd.to_numeric(df[req.column], errors="coerce")

    # Robust outlier filtering
    df = outliers(df, req.column)

    # Final non-negative gate if client asked for it
    enforce_nonneg = getattr(req, "nonnegative_only", False)
    if enforce_nonneg:
        mask = pd.to_numeric(df[req.column], errors="coerce").ge(0)
        df = df[mask.fillna(False)]

    save_df(req.session_id, df)

    # Audit
    user = get_user_from_request(request)
    try:
        ds = db_ops.get_dataset_by_session_id(req.session_id)
        ds_id = ds["id"] if ds else None
        if user:
            db_ops.log_action(
                user.get("user_id"),
                f"remove_outliers:{req.column}",
                dataset_id=ds_id,
                details=str(req.dict()),
            )
    except Exception:
        pass

    result = df_to_json(df)
    # Attach histogram for the same column if numeric-like
    if is_numeric_like(df[req.column]):
        result["histogram"] = column_histogram(df, req.column)
    return result


# ─────────────────────────────────────────────
# Normalization
# ─────────────────────────────────────────────
@app.post("/normalize")
def normalize_col(request: Request, req: ColumnRequest):
    df = get_df(req.session_id)
    if req.column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{req.column}' not found.")
    if not pd.api.types.is_numeric_dtype(df[req.column]):
        raise HTTPException(
            status_code=400, detail=f"Column '{req.column}' is not numeric."
        )
    df[req.column] = normalize(df, req.column)
    save_df(req.session_id, df)
    # Audit
    user = get_user_from_request(request)
    try:
        ds = db_ops.get_dataset_by_session_id(req.session_id)
        ds_id = ds["id"] if ds else None
        if user:
            db_ops.log_action(
                user.get("user_id"),
                f"normalize:{req.column}",
                dataset_id=ds_id,
                details="normalized column",
            )
    except Exception:
        pass
    return df_to_json(df)


# ─────────────────────────────────────────────
# One-Hot Encoding
# ─────────────────────────────────────────────
class OneHotRequest(BaseModel):
    session_id: str
    columns: List[str]


@app.post("/one-hot-encoding")
def one_hot(request: Request, req: OneHotRequest):
    df = get_df(req.session_id)
    for col in req.columns:
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{col}' not found.")
    df = One_hot_encoding(df, req.columns)
    save_df(req.session_id, df)
    # Audit
    user = get_user_from_request(request)
    try:
        ds = db_ops.get_dataset_by_session_id(req.session_id)
        ds_id = ds["id"] if ds else None
        if user:
            db_ops.log_action(
                user.get("user_id"),
                f"one_hot:{','.join(req.columns)}",
                dataset_id=ds_id,
                details="one-hot encoded columns",
            )
    except Exception:
        pass
    return df_to_json(df)


# ─────────────────────────────────────────────
# Linear Regression
# ─────────────────────────────────────────────
class RegressionRequest(BaseModel):
    session_id: str
    columns: List[str]  # [feature1, feature2, target]
    predict_values: Optional[List[float]] = None


@app.post("/linear-regression")
def linear_regression(request: Request, req: RegressionRequest):
    df = get_df(req.session_id)
    if len(req.columns) < 3:
        raise HTTPException(
            status_code=400,
            detail="Provide exactly 3 columns: feature1, feature2, target.",
        )
    df_model = df[req.columns].dropna()
    try:
        model = fit_linear_regression(df_model, req.columns)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Model training failed: {e}")

    response = {
        "r_squared": round(
            model.score(
                df_model[[req.columns[0], req.columns[1]]], df_model[req.columns[2]]
            ),
            4,
        )
    }
    if req.predict_values and len(req.predict_values) == 2:
        prediction = model.predict([req.predict_values])
        response["prediction"] = round(float(prediction[0]), 4)
        response["input"] = req.predict_values

    df[f"LinReg_Predicted_{req.columns[2]}"] = model.predict(
        df[[req.columns[0], req.columns[1]]].fillna(0)
    )
    save_df(req.session_id, df)
    # Audit
    user = get_user_from_request(request)
    try:
        ds = db_ops.get_dataset_by_session_id(req.session_id)
        ds_id = ds["id"] if ds else None
        if user:
            db_ops.log_action(
                user.get("user_id"),
                f"linear_regression:{','.join(req.columns)}",
                dataset_id=ds_id,
                details=str(req.predict_values),
            )
    except Exception:
        pass
    response.update(df_to_json(df))
    return response


# ─────────────────────────────────────────────
# Polynomial Regression
# ─────────────────────────────────────────────
class PolyRequest(BaseModel):
    session_id: str
    columns: List[str]
    degree: int = 2
    predict_values: Optional[List[float]] = None


@app.post("/polynomial-regression")
def polynomial_regression(request: Request, req: PolyRequest):
    df = get_df(req.session_id)
    if len(req.columns) < 3:
        raise HTTPException(
            status_code=400,
            detail="Provide exactly 3 columns: feature1, feature2, target.",
        )
    df_model = df[req.columns].dropna()
    try:
        model, poly, var1_poly = fit_poly_regression(df_model, req.columns, req.degree)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Model training failed: {e}")

    response = {}
    if req.predict_values and len(req.predict_values) == 2:
        prediction = model.predict(poly.transform([req.predict_values]))
        response["prediction"] = round(float(prediction[0][0]), 4)
        response["input"] = req.predict_values

    df[f"PolyReg_Predicted_{req.columns[2]}"] = model.predict(
        poly.transform(df[[req.columns[0], req.columns[1]]].fillna(0))
    )
    save_df(req.session_id, df)
    # Audit
    user = get_user_from_request(request)
    try:
        ds = db_ops.get_dataset_by_session_id(req.session_id)
        ds_id = ds["id"] if ds else None
        if user:
            db_ops.log_action(
                user.get("user_id"),
                f"poly_regression:{','.join(req.columns)}",
                dataset_id=ds_id,
                details=str(req.predict_values),
            )
    except Exception:
        pass
    response.update(df_to_json(df))
    return response


# ─────────────────────────────────────────────
# KNN Classifier
# ─────────────────────────────────────────────
class KNNRequest(BaseModel):
    session_id: str
    feature_columns: List[str]
    target_column: str
    k: int = 3
    predict_values: Optional[List[float]] = None


@app.post("/knn")
def knn_classifier(request: Request, req: KNNRequest):
    df = get_df(req.session_id)
    df_model = df[req.feature_columns + [req.target_column]].dropna()
    try:
        model, X, y = fit_knn(df_model, req.feature_columns, req.target_column, req.k)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Model training failed: {e}")

    response = {}
    if req.predict_values:
        if len(req.predict_values) != len(req.feature_columns):
            raise HTTPException(
                status_code=400, detail=f"Expected {len(req.feature_columns)} values."
            )
        prediction = model.predict([req.predict_values])
        response["prediction"] = str(prediction[0])
        response["input"] = req.predict_values

    df[f"KNN_Predicted_{req.target_column}"] = model.predict(
        df[req.feature_columns].fillna(0).values
    )
    save_df(req.session_id, df)
    # Audit
    user = get_user_from_request(request)
    try:
        ds = db_ops.get_dataset_by_session_id(req.session_id)
        ds_id = ds["id"] if ds else None
        if user:
            db_ops.log_action(
                user.get("user_id"),
                f"knn:{req.target_column}",
                dataset_id=ds_id,
                details=str(req.predict_values),
            )
    except Exception:
        pass
    response.update(df_to_json(df))
    return response


# ─────────────────────────────────────────────
# NLP Query
# ─────────────────────────────────────────────
class NLPRequest(BaseModel):
    session_id: str
    query: str


@app.post("/nlp-query")
def nlp_query(req: NLPRequest):
    df = get_df(req.session_id)
    intent, columns, values = classify_intent_with_columns_and_values(
        req.query, df.columns.tolist()
    )
    if intent is None:
        raise HTTPException(status_code=400, detail="Could not understand the query.")
    df = execute_intent(df, intent, columns, values)
    save_df(req.session_id, df)
    return {"intent": intent, "columns": columns, **df_to_json(df)}


# ─────────────────────────────────────────────
# Download
# ─────────────────────────────────────────────
@app.get("/download/{session_id}")
def download(session_id: str):
    df = get_df(session_id)
    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=cleaned_dataset.csv"},
    )


# ─────────────────────────────────────────────
# Reset
# ─────────────────────────────────────────────
@app.post("/reset")
def reset(req: SessionRequest):
    """Remove session so user can start fresh."""
    if req.session_id in sessions:
        del sessions[req.session_id]
    return {"message": "Session reset successfully."}


# ─────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


class PolicyRequest(BaseModel):
    session_id: str
    column: str
    zero_as_missing: Optional[bool] = None
    nonnegative_only: Optional[bool] = None


@app.get("/policies/{session_id}")
def get_policies(session_id: str):
    rec = sessions.get(session_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Session not found.")
    # Recalculate policies based on current dataframe state (after transformations)
    current_policies = infer_policies_for_df(rec["df"])
    rec["policies"] = current_policies  # Update stored policies
    return current_policies


@app.post("/policies/update")
def update_policy(req: PolicyRequest):
    rec = sessions.get(req.session_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Session not found.")
    pols = rec.setdefault("policies", {})
    pol = pols.setdefault(
        req.column,
        {"numeric_like": False, "zero_as_missing": False, "nonnegative_only": False},
    )
    if req.zero_as_missing is not None:
        pol["zero_as_missing"] = bool(req.zero_as_missing)
        if req.zero_as_missing:
            rec["df"] = apply_zero_as_missing(rec["df"], req.column)
    if req.nonnegative_only is not None:
        pol["nonnegative_only"] = bool(req.nonnegative_only)
        if req.nonnegative_only:
            rec["df"] = enforce_nonnegative_as_nan(rec["df"], req.column)
    save_df(req.session_id, rec["df"])
    return {"updated": True, "policy": pol, **df_to_json(rec["df"])}
