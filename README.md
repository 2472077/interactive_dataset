# Interactive Dataset Cleaner — Flask + FastAPI

## Project Structure

```
project/
├── backend/
│   ├── main.py          # FastAPI — all data processing API endpoints
│   ├── functions.py     # Data cleaning logic
│   └── nlp_query.py     # NLP intent classification
├── frontend/
│   ├── app.py           # Flask — UI routing
│   └── templates/
│       └── index.html   # Single-page UI
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Running the app

You need TWO terminals:

### Terminal 1 — Start FastAPI backend (port 8000)
```bash
cd backend
uvicorn main:app --reload --port 8000
```

### Terminal 2 — Start Flask frontend (port 5000)
```bash
cd frontend
python app.py
```

Then open your browser at: **http://localhost:5000**

## API Endpoints (FastAPI)

| Method | Endpoint                  | Description               |
|--------|---------------------------|---------------------------|
| POST   | /upload                   | Upload CSV/Excel/JSON     |
| POST   | /fill-missing             | Fill missing values       |
| POST   | /drop-missing             | Drop rows with NaN        |
| POST   | /remove-duplicates        | Remove duplicate rows     |
| POST   | /standardize              | Lowercase/uppercase/round |
| POST   | /remove-outliers          | IQR-based outlier removal |
| POST   | /normalize                | Min-max normalization     |
| POST   | /one-hot-encoding         | One-hot encode columns    |
| POST   | /linear-regression        | Train + predict           |
| POST   | /polynomial-regression    | Train + predict           |
| POST   | /knn                      | Train + predict           |
| POST   | /nlp-query                | Natural language query    |
| GET    | /download/{session_id}    | Download cleaned CSV      |
| POST   | /reset                    | Clear session             |

FastAPI interactive docs available at: **http://localhost:8000/docs**
