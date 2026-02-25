import os
import json
import requests
from flask import Flask, render_template, request, session, redirect, url_for, Response

app = Flask(__name__)
app.secret_key = "dataset-cleaner-secret-key"

API_BASE = "http://localhost:8000"


def login_required(fn):
    def wrapped(*args, **kwargs):
        if not session.get("user"):
            return redirect(url_for("login", next=request.path))
        return fn(*args, **kwargs)
    wrapped.__name__ = fn.__name__
    return wrapped


def api_post(endpoint, payload=None, files=None, use_auth=True):
    """Make POST request to backend with optional Bearer token auth."""
    try:
        headers = {}
        if use_auth and session.get("token"):
            headers["Authorization"] = f"Bearer {session.get('token')}"
        if files:
            r = requests.post(f"{API_BASE}/{endpoint}", files=files, headers=headers)
        else:
            r = requests.post(f"{API_BASE}/{endpoint}", json=payload, headers=headers)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.RequestException as e:
        try:
            detail = e.response.json().get("detail", str(e))
            if e.response.status_code in (404, 410, 401):
                session.clear()
        except Exception:
            detail = str(e)
        return None, detail


def api_get(endpoint, use_auth=True):
    """Make GET request to backend with optional Bearer token auth."""
    try:
        headers = {}
        if use_auth and session.get("token"):
            headers["Authorization"] = f"Bearer {session.get('token')}"
        r = requests.get(f"{API_BASE}/{endpoint}", headers=headers)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.RequestException as e:
        try:
            detail = e.response.json().get("detail", str(e))
            if e.response.status_code in (401, 403):
                session.clear()
        except Exception:
            detail = str(e)
        return None, detail


@app.route("/", methods=["GET", "POST"])
@login_required
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if not file:
            return render_template("index.html", error="No file selected.")

        data, err = api_post("upload", files={"file": (file.filename, file.stream, file.content_type)})
        if err:
            return render_template("index.html", error=err)

        session["session_id"] = data["session_id"]
        session["columns"]    = data["columns"]
        session["table"]      = data

        return render_template("index.html",
                               table=data,
                               columns=data["columns"],
                               session_id=data["session_id"],
                               success="File uploaded successfully!")

    # On every GET request, verify token is still valid
    if not session.get("token"):
        return redirect(url_for("login"))

    table   = session.get("table")
    columns = session.get("columns", [])
    return render_template("index.html", table=table, columns=columns, session_id=session.get("session_id"))


@app.route("/fill-missing", methods=["POST"])
def fill_missing():
    col    = request.form.get("column")
    method = request.form.get("method")
    custom = request.form.get("custom_value", "")
    data, err = api_post("fill-missing", {
        "session_id":   session.get("session_id"),
        "column":       col,
        "method":       method,
        "custom_value": custom or None
    })
    return _render(data, err, f"Filled missing values in '{col}' using {method}.")


@app.route("/drop-missing", methods=["POST"])
def drop_missing():
    data, err = api_post("drop-missing", {"session_id": session.get("session_id")})
    return _render(data, err, "Dropped all rows with missing values.")


@app.route("/remove-duplicates", methods=["POST"])
def remove_duplicates():
    data, err = api_post("remove-duplicates", {"session_id": session.get("session_id")})
    return _render(data, err, "Removed duplicate rows.")


@app.route("/standardize", methods=["POST"])
def standardize():
    col     = request.form.get("column")
    method  = request.form.get("method")
    decimal = int(request.form.get("decimal_places", 2))
    data, err = api_post("standardize", {
        "session_id":     session.get("session_id"),
        "column":         col,
        "method":         method,
        "decimal_places": decimal
    })
    return _render(data, err, f"Standardized column '{col}' using {method}.")


@app.route("/remove-outliers", methods=["POST"])
def remove_outliers():
    col = request.form.get("column")
    nonneg = request.form.get("nonnegative_only") == "on"
    data, err = api_post("remove-outliers", {
        "session_id": session.get("session_id"),
        "column":     col,
        "nonnegative_only": nonneg
    })
    return _render(data, err, f"Removed outliers from '{col}'.")


@app.route("/normalize", methods=["POST"])
def normalize():
    col = request.form.get("column")
    data, err = api_post("normalize", {
        "session_id": session.get("session_id"),
        "column":     col
    })
    return _render(data, err, f"Normalized column '{col}'.")


@app.route("/one-hot-encoding", methods=["POST"])
def one_hot():
    cols = request.form.getlist("columns")
    data, err = api_post("one-hot-encoding", {
        "session_id": session.get("session_id"),
        "columns":    cols
    })
    return _render(data, err, f"One-hot encoded: {', '.join(cols)}.")


@app.route("/linear-regression", methods=["POST"])
def linear_regression():
    cols  = [request.form.get("feature1"), request.form.get("feature2"), request.form.get("target")]
    raw   = request.form.get("predict_values", "")
    pvals = [float(v) for v in raw.split(",") if v.strip()] if raw.strip() else None
    data, err = api_post("linear-regression", {
        "session_id": session.get("session_id"), "columns": cols, "predict_values": pvals
    })
    prediction = data.get("prediction") if data else None
    r_squared  = data.get("r_squared")  if data else None
    return _render(data, err, "Linear Regression trained.", prediction=prediction, r_squared=r_squared)


@app.route("/polynomial-regression", methods=["POST"])
def polynomial_regression():
    cols   = [request.form.get("feature1"), request.form.get("feature2"), request.form.get("target")]
    degree = int(request.form.get("degree", 2))
    raw    = request.form.get("predict_values", "")
    pvals  = [float(v) for v in raw.split(",") if v.strip()] if raw.strip() else None
    data, err = api_post("polynomial-regression", {
        "session_id": session.get("session_id"), "columns": cols, "degree": degree, "predict_values": pvals
    })
    prediction = data.get("prediction") if data else None
    return _render(data, err, "Polynomial Regression trained.", prediction=prediction)


@app.route("/knn", methods=["POST"])
def knn():
    feats  = request.form.getlist("feature_columns")
    target = request.form.get("target_column")
    k      = int(request.form.get("k", 3))
    raw    = request.form.get("predict_values", "")
    pvals  = [float(v) for v in raw.split(",") if v.strip()] if raw.strip() else None
    data, err = api_post("knn", {
        "session_id": session.get("session_id"), "feature_columns": feats,
        "target_column": target, "k": k, "predict_values": pvals
    })
    prediction = data.get("prediction") if data else None
    return _render(data, err, "KNN Classifier trained.", prediction=prediction)


@app.route("/nlp-query", methods=["POST"])
def nlp_query():
    query = request.form.get("query", "")
    data, err = api_post("nlp-query", {
        "session_id": session.get("session_id"), "query": query
    })
    msg = f"Intent: {data.get('intent')} | Columns: {data.get('columns')}" if data else None
    return _render(data, err, msg)


@app.route("/download")
def download():
    sid = session.get("session_id")
    if not sid:
        return redirect(url_for("index"))
    r = requests.get(f"{API_BASE}/download/{sid}", stream=True)
    return Response(
        r.iter_content(chunk_size=1024),
        content_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=cleaned_dataset.csv"}
    )


@app.route("/reset", methods=["POST"])
def reset():
    sid = session.get("session_id")
    if sid:
        api_post("reset", payload={"session_id": sid})
    session.clear()
    return redirect(url_for("index"))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        role = request.form.get('role', 'user')
        if not username or not password:
            return render_template('register.html', error='Username and password required')
        # Call backend to register
        data, err = api_post('auth/register', {'username': username, 'password': password, 'role': role}, use_auth=False)
        if err:
            return render_template('register.html', error=err)
        return render_template('login.html', success='Account created, please login')
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        if not username or not password:
            return render_template('login.html', error='Username and password required')
        # Call backend to login
        data, err = api_post('auth/login', {'username': username, 'password': password}, use_auth=False)
        if err:
            return render_template('login.html', error=err)
        # Store token and user info in session
        session['token'] = data.get('token')
        session['user'] = data.get('username')
        session['role'] = data.get('role', 'user')
        # redirect to next param if present
        nxt = request.args.get('next') or url_for('index')
        return redirect(nxt)
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


def _render(data, err, success_msg=None, **kwargs):
    if err:
        return render_template("index.html",
                               table=session.get("table"),
                               columns=session.get("columns", []),
                               session_id=session.get("session_id", ""),
                               error=err, **kwargs)
    session["table"]   = data
    session["columns"] = data["columns"]
    return render_template("index.html",
                           table=data,
                           columns=data["columns"],
                           session_id=session.get("session_id"),
                           success=success_msg,
                           **kwargs)


@app.route('/admin/dashboard')
@login_required
def admin_dashboard():
    # Check if user is admin
    if session.get('role') != 'admin':
        return render_template('index.html', error='Admin access required')
    
    # Fetch audit logs
    logs, log_err = api_get('admin/audit-logs')
    logs = logs.get('logs', []) if logs else []
    
    # Fetch datasets
    datasets, ds_err = api_get('admin/datasets')
    datasets = datasets.get('datasets', []) if datasets else []
    
    # Fetch users
    users, user_err = api_get('admin/users')
    users = users.get('users', []) if users else []
    
    return render_template('admin_dashboard.html', 
                           logs=logs, 
                           datasets=datasets, 
                           users=users,
                           log_err=log_err,
                           ds_err=ds_err,
                           user_err=user_err)


if __name__ == "__main__":
    app.run(debug=True, port=5000)