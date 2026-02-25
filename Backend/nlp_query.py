# import re
# from transformers import pipeline
# from fuzzywuzzy import process
# from functions import *

# # Initialize zero-shot classification pipeline
# nlp_pipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# intents = {
#     'fill_missing_mean': ['fill with mean', 'mean', 'average'],
#     'fill_missing_median': ['fill with median', 'median', 'middle', 'mid'],
#     'fill_missing_mode': ['fill with mode', 'mode', 'recurrent', 'most frequent', 'most common'],
#     'fill_missing_custom': ['fill with custom', 'custom', 'personal'],
#     'drop_missing': ['drop', 'missing', 'null', 'nan'],
#     'apply_normalization': ['normalize', 'normalization', 'scaling'],
#     'one_hot_encoding': ['one-hot', 'encoding', 'encode'],
#     'remove_duplicates': ['duplicate', 'duplicates', 'drop'],
#     'remove_outliers': ['outlier', 'outliers', 'remove'],
#     'capitalize': ['capitalize', 'capital', 'uppercase'],
#     'lowercase': ['lowercase', 'lower'],
#     'uppercase': ['uppercase', 'upper'],
#     'linear_regression': ['linear regression', 'linear', 'regression'],
#     'polynomial_regression': ['polynomial regression', 'polynomial'],
#     'knn': ['knn', 'k-nearest neighbors', 'neighbors']
# }


# def classify_intent_with_columns_and_values(text, columns):
#     if not text or text.strip() == "":
#         return None, None, None

#     intent_column_combinations = [f"{intent} on {col}" for intent in intents for col in columns]

#     try:
#         result = nlp_pipeline(text, intent_column_combinations)
#         top_match = result['labels'][0]

#         if " on " in top_match:
#             intent, first_column = top_match.split(" on ")
#         else:
#             return None, None, None

#         refined_intent = refine_intent_classification(intent, text)
#         mentioned_columns = [col for col in columns if col in text]

#         if len(mentioned_columns) == 0 and "," in text:
#             column_candidates = text.split(",")
#             mentioned_columns = [col.strip() for col in column_candidates if col.strip() in columns]

#         if not mentioned_columns:
#             mentioned_columns = [first_column]

#         values = extract_values_from_text(text)
#         return refined_intent, mentioned_columns, values

#     except Exception as e:
#         print(f"Error during classification: {e}")
#         return None, None, None


# def refine_intent_classification(intent, text):
#     for refined_intent, keywords in intents.items():
#         for keyword in keywords:
#             if keyword.lower() in text.lower():
#                 return refined_intent
#     return intent


# def extract_values_from_text(text):
#     values = re.findall(r'\d+\.?\d*', text)
#     if values:
#         return [float(value) for value in values]
#     return []


# def execute_intent(df_cleaned, intent, column, values):
#     if intent == 'fill_missing_mean':
#         for col in column:
#             df_cleaned[col] = mean(df_cleaned, col)
#         return df_cleaned
#     elif intent == 'fill_missing_median':
#         for col in column:
#             df_cleaned[col] = median(df_cleaned, col)
#         return df_cleaned
#     elif intent == 'fill_missing_mode':
#         for col in column:
#             df_cleaned[col] = mode(df_cleaned, col)
#         return df_cleaned
#     elif intent == 'drop_missing':
#         return drop_missing_values(df_cleaned)
#     elif intent == 'apply_normalization':
#         for col in column:
#             df_cleaned[col] = normalize(df_cleaned, col)
#         return df_cleaned
#     elif intent == 'one_hot_encoding':
#         return One_hot_encoding(df_cleaned, column)
#     elif intent == 'remove_duplicates':
#         return drop_duplicates(df_cleaned)
#     elif intent == 'remove_outliers':
#         for col in column:
#             df_cleaned = outliers(df_cleaned, col)
#         return df_cleaned
#     elif intent in ('capitalize', 'title'):
#         for col in column:
#             df_cleaned[col] = title(df_cleaned, col)
#         return df_cleaned
#     elif intent == 'lowercase':
#         for col in column:
#             df_cleaned[col] = lower(df_cleaned, col)
#         return df_cleaned
#     elif intent == 'uppercase':
#         for col in column:
#             df_cleaned[col] = upper(df_cleaned, col)
#         return df_cleaned
#     else:
#         return df_cleaned




# Backend/nlp_query.py
import re
from fuzzywuzzy import process
from functions import *

# --- Optional transformers import (do NOT fail app if unavailable) ---
try:
    from transformers import pipeline
    _transformers_available = True
except Exception:
    pipeline = None
    _transformers_available = False

# Intent dictionary (same as yours; you can expand it)
intents = {
    'fill_missing_mean':   ['fill with mean', 'mean', 'average'],
    'fill_missing_median': ['fill with median', 'median', 'middle', 'mid'],
    'fill_missing_mode':   ['fill with mode', 'mode', 'recurrent', 'most frequent', 'most common'],
    'fill_missing_custom': ['fill with custom', 'custom', 'personal'],
    'drop_missing':        ['drop', 'missing', 'null', 'nan'],
    'apply_normalization': ['normalize', 'normalization', 'scaling'],
    'one_hot_encoding':    ['one-hot', 'encoding', 'encode'],
    'remove_duplicates':   ['duplicate', 'duplicates', 'drop'],
    'remove_outliers':     ['outlier', 'outliers', 'remove'],
    'capitalize':          ['capitalize', 'capital', 'uppercase'],
    'lowercase':           ['lowercase', 'lower'],
    'uppercase':           ['uppercase', 'upper'],
    'linear_regression':   ['linear regression', 'linear', 'regression'],
    'polynomial_regression':['polynomial regression', 'polynomial'],
    'knn':                 ['knn', 'k-nearest neighbors', 'neighbors']
}

# -------------------- Lazy HF pipeline --------------------
_nlp_pipeline = None
def get_nlp_pipeline():
    """Try to instantiate the HF pipeline only when needed.
       If internet / SSL blocks it, return None and let the caller fallback."""
    global _nlp_pipeline
    if _nlp_pipeline is not None:
        return _nlp_pipeline

    if not _transformers_available:
        return None

    try:
        # Attempt without local_files_only first; if SSL fails, we'll catch and fallback.
        _nlp_pipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        return _nlp_pipeline
    except Exception as e:
        # Final fallback: no HF pipeline available
        print(f"[NLP] Could not initialize HF pipeline: {e}")
        _nlp_pipeline = None
        return None

# -------------------- Classification helpers --------------------
def refine_intent_classification(intent, text):
    for refined_intent, keywords in intents.items():
        for keyword in keywords:
            if keyword.lower() in text.lower():
                return refined_intent
    return intent

def extract_values_from_text(text):
    values = re.findall(r'\d+\.?\d*', text)
    if values:
        return [float(value) for value in values]
    return []

def classify_intent_with_columns_and_values(text, columns):
    """Try HF zero-shot first; if unavailable, do a fuzzy fallback."""
    if not text or text.strip() == "":
        return None, None, None

    # Try HF pipeline
    nlp = get_nlp_pipeline()
    if nlp:
        intent_column_combinations = [f"{intent} on {col}" for intent in intents for col in columns]
        try:
            result = nlp(text, intent_column_combinations)
            top_match = result['labels'][0]
            if " on " in top_match:
                intent, first_column = top_match.split(" on ")
            else:
                return None, None, None
            refined_intent = refine_intent_classification(intent, text)

            mentioned_columns = [col for col in columns if col in text]
            if len(mentioned_columns) == 0 and "," in text:
                column_candidates = text.split(",")
                mentioned_columns = [col.strip() for col in column_candidates if col.strip() in columns]
            if not mentioned_columns:
                mentioned_columns = [first_column]

            values = extract_values_from_text(text)
            return refined_intent, mentioned_columns, values
        except Exception as e:
            print(f"[NLP] HF classification failed at runtime: {e}")

    # Fallback: rule-based fuzzy matching (no internet required)
    # 1) pick the best intent by fuzzy ratio against all keywords
    all_pairs = []
    for key, kws in intents.items():
        for kw in kws:
            all_pairs.append((key, kw))

    best = process.extractOne(
        text.lower(),
        [kw for _, kw in all_pairs]
    )
    if best is None:
        return None, None, None

    best_kw = best[0]
    # Map keyword back to intent
    matched_intent = None
    for key, kw in all_pairs:
        if kw == best_kw:
            matched_intent = key
            break

    if matched_intent is None:
        return None, None, None

    # Columns: choose any mentioned directly; else default to first numeric column or first column
    mentioned_columns = [col for col in columns if col.lower() in text.lower()]
    if not mentioned_columns:
        mentioned_columns = [columns[0]] if columns else []

    values = extract_values_from_text(text)
    return matched_intent, mentioned_columns, values

def execute_intent(df_cleaned, intent, column, values):
    if intent == 'fill_missing_mean':
        for col in column:
            df_cleaned[col] = mean(df_cleaned, col)
        return df_cleaned
    elif intent == 'fill_missing_median':
        for col in column:
            df_cleaned[col] = median(df_cleaned, col)
        return df_cleaned
    elif intent == 'fill_missing_mode':
        for col in column:
            df_cleaned[col] = mode(df_cleaned, col)
        return df_cleaned
    elif intent == 'fill_missing_custom':
        # If a value is supplied in 'values', use first; else leave as-is
        if values and len(values) > 0:
            for col in column:
                df_cleaned[col] = custom(df_cleaned, col, values[0])
        return df_cleaned
    elif intent == 'drop_missing':
        return drop_missing_values(df_cleaned)
    elif intent == 'apply_normalization':
        for col in column:
            df_cleaned[col] = normalize(df_cleaned, col)
        return df_cleaned
    elif intent == 'one_hot_encoding':
        return One_hot_encoding(df_cleaned, column)
    elif intent == 'remove_duplicates':
        return drop_duplicates(df_cleaned)
    elif intent == 'remove_outliers':
        for col in column:
            df_cleaned = outliers(df_cleaned, col)
        return df_cleaned
    elif intent in ('capitalize', 'title'):
        for col in column:
            df_cleaned[col] = title(df_cleaned, col)
        return df_cleaned
    elif intent == 'lowercase':
        for col in column:
            df_cleaned[col] = lower(df_cleaned, col)
        return df_cleaned
    elif intent == 'uppercase':
        for col in column:
            df_cleaned[col] = upper(df_cleaned, col)
        return df_cleaned
    elif intent == 'linear_regression':
        # Optional: you can route to your linear endpoint instead
        return df_cleaned
    elif intent == 'polynomial_regression':
        return df_cleaned
    elif intent == 'knn':
        return df_cleaned
    return df_cleaned
