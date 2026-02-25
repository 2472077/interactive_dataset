# import io
# import uuid
# import time 
# import pandas as pd
# from fastapi import FastAPI, UploadFile, File, HTTPException
# from fastapi.responses import StreamingResponse
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import List, Optional
# from functions import *
# from nlp_query import classify_intent_with_columns_and_values, execute_intent

# app = FastAPI(title="Dataset Cleaner API")

# # Allow Flask frontend to talk to this API
# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=["http://localhost:5000"],
# #     allow_methods=["*"],
# #     allow_headers=["*"],
# # )


# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:5000", "http://127.0.0.1:5000"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# # In-memory session store: { session_id: dataframe }
# # Session store: { session_id: {"df": dataframe, "created_at": timestamp} }
# sessions = {}
# SESSION_TIMEOUT = 600  # 10 minutes in seconds


# def get_df(session_id: str) -> pd.DataFrame:
#     if session_id not in sessions:
#         raise HTTPException(status_code=404, detail="Session not found. Please upload a file first.")
    
#     # Check if session has timed out
#     elapsed = time.time() - sessions[session_id]["created_at"]
#     if elapsed > SESSION_TIMEOUT:
#         del sessions[session_id]
#         raise HTTPException(status_code=410, detail="Session timed out. Please upload your file again.")
    
#     return sessions[session_id]["df"].copy()


# def save_df(session_id: str, df: pd.DataFrame):
#     if session_id in sessions:
#         # Keep original created_at so timeout is based on session start
#         sessions[session_id]["df"] = df
#     else:
#         sessions[session_id] = {"df": df, "created_at": time.time()}


# def df_to_json(df: pd.DataFrame) -> dict:
#     # Convert to JSON string using pandas (handles NaN/Inf natively) then parse back
#     import json
#     df = df.replace([float('inf'), float('-inf')], None)
#     records = json.loads(df.to_json(orient="records"))
#     return {
#         "columns": df.columns.tolist(),
#         "data": records,
#         "shape": list(df.shape)
#     }


# # ─────────────────────────────────────────────
# # Upload
# # ─────────────────────────────────────────────


# @app.post("/upload")
# async def upload_file(file: UploadFile = File(...)):
#     contents = await file.read()
#     try:
#         if file.filename.endswith(".csv"):
#             df = pd.read_csv(io.BytesIO(contents))
#         elif file.filename.endswith(".xlsx"):
#             df = pd.read_excel(io.BytesIO(contents))
#         elif file.filename.endswith(".json"):
#             df = pd.read_json(io.BytesIO(contents))
#         else:
#             raise HTTPException(status_code=400, detail="Unsupported file type.")
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Error reading file: {e}")

#     # Normalize common missing tokens and drop empty rows
#     df = df.replace(r'^\s*$', pd.NA, regex=True)
#     df = df.replace(["NaN", "N/A", "NULL", "null", "None"], pd.NA)
#     df = df.dropna(how='all')

#     # 1) Infer policies from the data itself
#     policies = infer_policies_for_df(df)

#     # 2) Apply the policies generically (no dataset-specific lists)
#     for col, pol in policies.items():
#         if pol.get("numeric_like"):
#             if pol.get("zero_as_missing"):
#                 df = apply_zero_as_missing(df, col)
#             if pol.get("nonnegative_only"):
#                 df = enforce_nonnegative_as_nan(df, col)

#     # 3) Save session with df and policies
#     session_id = str(uuid.uuid4())
#     sessions[session_id] = {
#         "df": df,
#         "created_at": time.time(),
#         "policies": policies
#     }

#     return {"session_id": session_id, **df_to_json(df)}



# # ─────────────────────────────────────────────
# # Fill Missing Values
# # ─────────────────────────────────────────────

# class FillRequest(BaseModel):
#     session_id: str
#     column: str
#     method: str              # mean | median | mode | custom
#     custom_value: Optional[str] = None


# @app.post("/fill-missing")
# def fill_missing(req: FillRequest):
#     record = sessions.get(req.session_id)
#     if not record:
#         raise HTTPException(status_code=404, detail="Session not found. Please upload a file first.")

#     df = record["df"].copy()
#     col = req.column

#     if col not in df.columns:
#         raise HTTPException(status_code=400, detail=f"Column '{col}' not found.")

#     # ✅ ALWAYS coerce to numeric before mean/median
#     if req.method in ("mean", "median"):
#         df[col] = pd.to_numeric(df[col], errors="coerce")

#     try:
#         if req.method == "mean":
#             df[col] = mean(df, col)
#         elif req.method == "median":
#             df[col] = median(df, col)
#         elif req.method == "mode":
#             df[col] = mode(df, col)
#         elif req.method == "custom":
#             if req.custom_value is None:
#                 raise HTTPException(status_code=400, detail="custom_value required for custom fill.")
#             try:
#                 df[col] = custom(df, col, float(req.custom_value))
#             except ValueError:
#                 df[col] = custom(df, col, req.custom_value)
#         else:
#             raise HTTPException(status_code=400, detail="method must be mean | median | mode | custom")

#     except TypeError as e:
#         raise HTTPException(status_code=400, detail=f"Column '{col}' not numeric: {e}")

#     save_df(req.session_id, df)
#     sessions[req.session_id]["df"] = df
#     return df_to_json(df)




# # ─────────────────────────────────────────────
# # Drop Missing Values
# # ─────────────────────────────────────────────

# class SessionRequest(BaseModel):
#     session_id: str


# @app.post("/drop-missing")
# def drop_missing(req: SessionRequest):
#     df = get_df(req.session_id)
#     df = drop_missing_values(df)
#     save_df(req.session_id, df)
#     return df_to_json(df)

# @app.post("/verify-session")
# def verify_session(req: SessionRequest):
#     if req.session_id not in sessions:
#         raise HTTPException(status_code=404, detail="Session not found.")
    
#     elapsed = time.time() - sessions[req.session_id]["created_at"]
#     if elapsed > SESSION_TIMEOUT:
#         del sessions[req.session_id]
#         raise HTTPException(status_code=410, detail="Session timed out.")
    
#     return {"status": "alive"}


# # ─────────────────────────────────────────────
# # Remove Duplicates
# # ─────────────────────────────────────────────

# @app.post("/remove-duplicates")
# def remove_duplicates(req: SessionRequest):
#     df = get_df(req.session_id)
#     df = drop_duplicates(df)
#     save_df(req.session_id, df)
#     return df_to_json(df)


# # ─────────────────────────────────────────────
# # Standardization
# # ─────────────────────────────────────────────

# class StandardizeRequest(BaseModel):
#     session_id: str
#     column: str
#     method: str   # lowercase | uppercase | titlecase | date | round
#     decimal_places: Optional[int] = 2


# @app.post("/standardize")
# def standardize(req: StandardizeRequest):
#     df = get_df(req.session_id)
#     col = req.column

#     if col not in df.columns:
#         raise HTTPException(status_code=400, detail=f"Column '{col}' not found.")

#     if req.method == "lowercase":
#         df[col] = lower(df, col)
#     elif req.method == "uppercase":
#         df[col] = upper(df, col)
#     elif req.method == "titlecase":
#         df[col] = title(df, col)
#     elif req.method == "date":
#         df[col] = date_time(df, col)
#     elif req.method == "round":
#         if not pd.api.types.is_numeric_dtype(df[col]):
#             raise HTTPException(status_code=400, detail=f"Column '{col}' is not numeric.")
#         df[col] = df[col].round(req.decimal_places)
#     else:
#         raise HTTPException(status_code=400, detail="Invalid method.")

#     save_df(req.session_id, df)
#     return df_to_json(df)


# # ─────────────────────────────────────────────
# # Remove Outliers
# # ─────────────────────────────────────────────

# class ColumnRequest(BaseModel):
#     session_id: str
#     column: str
#     nonnegative_only: Optional[bool] = False



# def outliers(df, col):
#     """
#     Generic robust outlier filter:
#     - Coerce to numeric (non-parsable -> NaN)
#     - Trim 1%-99% to reduce leverage of extreme values
#     - Compute IQR on trimmed sample; filter to [Q1-1.5*IQR, Q3+1.5*IQR]
#     - If column is mostly non-negative, remove negatives outright
#     """
#     s = pd.to_numeric(df[col], errors="coerce")
#     s_nonnull = s.dropna()
#     if s_nonnull.empty:
#         return df

#     # Trim extremes first
#     q_low, q_high = s_nonnull.quantile([0.01, 0.99])
#     trimmed = s_nonnull[(s_nonnull >= q_low) & (s_nonnull <= q_high)]
#     if trimmed.empty:
#         trimmed = s_nonnull

#     Q1 = trimmed.quantile(0.25)
#     Q3 = trimmed.quantile(0.75)
#     IQR = Q3 - Q1

#     if pd.isna(Q1) or pd.isna(Q3) or IQR == 0:
#         lower, upper = q_low, q_high
#     else:
#         lower = Q1 - 1.5 * IQR
#         upper = Q3 + 1.5 * IQR

#     mask = s.between(lower, upper, inclusive="both")

#     # If >=95% of observed values are non-negative, enforce non-negativity
#     nonneg_share = (s_nonnull >= 0).mean()
#     if nonneg_share >= 0.95:
#         mask = mask & (s >= 0)

#     return df[mask.fillna(False)]


# @app.post("/remove-outliers")
# def remove_outliers(req: ColumnRequest):
#     df = get_df(req.session_id)

#     if req.column not in df.columns:
#         raise HTTPException(status_code=400, detail=f"Column '{req.column}' not found.")

#     # Always coerce to numeric for outlier math
#     df[req.column] = pd.to_numeric(df[req.column], errors="coerce")

#     # Robust outlier filtering (see functions.py below)
#     df = outliers(df, req.column)

#     # Backward-compatible read of flag (in case old clients don't send it)
#     enforce_nonneg = getattr(req, "nonnegative_only", False)

#     # Final non-negative gate if client asked for it
#     if enforce_nonneg:
#         mask = pd.to_numeric(df[req.column], errors="coerce").ge(0)
#         df = df[mask.fillna(False)]

#     save_df(req.session_id, df)
#     return df_to_json(df)





# # ─────────────────────────────────────────────
# # Normalization
# # ─────────────────────────────────────────────

# @app.post("/normalize")
# def normalize_col(req: ColumnRequest):
#     df = get_df(req.session_id)
#     if req.column not in df.columns:
#         raise HTTPException(status_code=400, detail=f"Column '{req.column}' not found.")
#     if not pd.api.types.is_numeric_dtype(df[req.column]):
#         raise HTTPException(status_code=400, detail=f"Column '{req.column}' is not numeric.")
#     df[req.column] = normalize(df, req.column)
#     save_df(req.session_id, df)
#     return df_to_json(df)


# # ─────────────────────────────────────────────
# # One-Hot Encoding
# # ─────────────────────────────────────────────

# class OneHotRequest(BaseModel):
#     session_id: str
#     columns: List[str]


# @app.post("/one-hot-encoding")
# def one_hot(req: OneHotRequest):
#     df = get_df(req.session_id)
#     for col in req.columns:
#         if col not in df.columns:
#             raise HTTPException(status_code=400, detail=f"Column '{col}' not found.")
#     df = One_hot_encoding(df, req.columns)
#     save_df(req.session_id, df)
#     return df_to_json(df)


# # ─────────────────────────────────────────────
# # Linear Regression
# # ─────────────────────────────────────────────

# class RegressionRequest(BaseModel):
#     session_id: str
#     columns: List[str]        # [feature1, feature2, target]
#     predict_values: Optional[List[float]] = None


# @app.post("/linear-regression")
# def linear_regression(req: RegressionRequest):
#     df = get_df(req.session_id)

#     if len(req.columns) < 3:
#         raise HTTPException(status_code=400, detail="Provide exactly 3 columns: feature1, feature2, target.")

#     df_model = df[req.columns].dropna()
#     try:
#         model = fit_linear_regression(df_model, req.columns)
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Model training failed: {e}")

#     response = {"r_squared": round(model.score(df_model[[req.columns[0], req.columns[1]]], df_model[req.columns[2]]), 4)}

#     if req.predict_values and len(req.predict_values) == 2:
#         prediction = model.predict([req.predict_values])
#         response["prediction"] = round(float(prediction[0]), 4)
#         response["input"] = req.predict_values

#     # Add predictions column to df
#     df[f'LinReg_Predicted_{req.columns[2]}'] = model.predict(df[[req.columns[0], req.columns[1]]].fillna(0))
#     save_df(req.session_id, df)
#     response.update(df_to_json(df))
#     return response


# # ─────────────────────────────────────────────
# # Polynomial Regression
# # ─────────────────────────────────────────────

# class PolyRequest(BaseModel):
#     session_id: str
#     columns: List[str]
#     degree: int = 2
#     predict_values: Optional[List[float]] = None


# @app.post("/polynomial-regression")
# def polynomial_regression(req: PolyRequest):
#     df = get_df(req.session_id)

#     if len(req.columns) < 3:
#         raise HTTPException(status_code=400, detail="Provide exactly 3 columns: feature1, feature2, target.")

#     df_model = df[req.columns].dropna()
#     try:
#         model, poly, var1_poly = fit_poly_regression(df_model, req.columns, req.degree)
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Model training failed: {e}")

#     response = {}

#     if req.predict_values and len(req.predict_values) == 2:
#         prediction = model.predict(poly.transform([req.predict_values]))
#         response["prediction"] = round(float(prediction[0][0]), 4)
#         response["input"] = req.predict_values

#     df[f'PolyReg_Predicted_{req.columns[2]}'] = model.predict(
#         poly.transform(df[[req.columns[0], req.columns[1]]].fillna(0))
#     )
#     save_df(req.session_id, df)
#     response.update(df_to_json(df))
#     return response


# # ─────────────────────────────────────────────
# # KNN Classifier
# # ─────────────────────────────────────────────

# class KNNRequest(BaseModel):
#     session_id: str
#     feature_columns: List[str]
#     target_column: str
#     k: int = 3
#     predict_values: Optional[List[float]] = None


# @app.post("/knn")
# def knn_classifier(req: KNNRequest):
#     df = get_df(req.session_id)
#     df_model = df[req.feature_columns + [req.target_column]].dropna()

#     try:
#         model, X, y = fit_knn(df_model, req.feature_columns, req.target_column, req.k)
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Model training failed: {e}")

#     response = {}

#     if req.predict_values:
#         if len(req.predict_values) != len(req.feature_columns):
#             raise HTTPException(status_code=400, detail=f"Expected {len(req.feature_columns)} values.")
#         prediction = model.predict([req.predict_values])
#         response["prediction"] = str(prediction[0])
#         response["input"] = req.predict_values

#     df[f'KNN_Predicted_{req.target_column}'] = model.predict(
#         df[req.feature_columns].fillna(0).values
#     )
#     save_df(req.session_id, df)
#     response.update(df_to_json(df))
#     return response


# # ─────────────────────────────────────────────
# # NLP Query
# # ─────────────────────────────────────────────

# class NLPRequest(BaseModel):
#     session_id: str
#     query: str


# @app.post("/nlp-query")
# def nlp_query(req: NLPRequest):
#     df = get_df(req.session_id)
#     intent, columns, values = classify_intent_with_columns_and_values(req.query, df.columns.tolist())

#     if intent is None:
#         raise HTTPException(status_code=400, detail="Could not understand the query.")

#     df = execute_intent(df, intent, columns, values)
#     save_df(req.session_id, df)
#     return {"intent": intent, "columns": columns, **df_to_json(df)}


# # ─────────────────────────────────────────────
# # Download
# # ─────────────────────────────────────────────

# @app.get("/download/{session_id}")
# def download(session_id: str):
#     df = get_df(session_id)
#     output = io.StringIO()
#     df.to_csv(output, index=False)
#     output.seek(0)
#     return StreamingResponse(
#         iter([output.getvalue()]),
#         media_type="text/csv",
#         headers={"Content-Disposition": "attachment; filename=cleaned_dataset.csv"}
#     )


# # ─────────────────────────────────────────────
# # Reset
# # ─────────────────────────────────────────────

# @app.post("/reset")
# def reset(req: SessionRequest):
#     """Remove session so user can start fresh."""
#     if req.session_id in sessions:
#         del sessions[req.session_id]
#     return {"message": "Session reset successfully."}


# # ─────────────────────────────────────────────
# # Health check
# # ─────────────────────────────────────────────

# @app.get("/health")
# def health():
#     return {"status": "ok"}



# class PolicyRequest(BaseModel):
#     session_id: str
#     column: str
#     zero_as_missing: bool | None = None
#     nonnegative_only: bool | None = None

# @app.get("/policies/{session_id}")
# def get_policies(session_id: str):
#     rec = sessions.get(session_id)
#     if not rec:
#         raise HTTPException(status_code=404, detail="Session not found.")
#     return rec.get("policies", {})

# @app.post("/policies/update")
# def update_policy(req: PolicyRequest):
#     rec = sessions.get(req.session_id)
#     if not rec:
#         raise HTTPException(status_code=404, detail="Session not found.")
#     pols = rec.setdefault("policies", {})
#     pol = pols.setdefault(req.column, {"numeric_like": False, "zero_as_missing": False, "nonnegative_only": False})
#     if req.zero_as_missing is not None:
#         pol["zero_as_missing"] = bool(req.zero_as_missing)
#         if req.zero_as_missing:
#             rec["df"] = apply_zero_as_missing(rec["df"], req.column)
#     if req.nonnegative_only is not None:
#         pol["nonnegative_only"] = bool(req.nonnegative_only)
#         if req.nonnegative_only:
#             rec["df"] = enforce_nonnegative_as_nan(rec["df"], req.column)

#     save_df(req.session_id, rec["df"])
#     return {"updated": True, "policy": pol, **df_to_json(rec["df"])}