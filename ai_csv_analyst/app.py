import json
import re
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st
import requests

st.set_page_config(page_title="AI CSV Analyst", page_icon="📊", layout="wide")

st.title("📊 AI CSV Analyst")
st.caption("Ask questions about a CSV in plain English. Ollama proposes a small JSON plan; trusted Pandas functions perform the calculations.")

OLLAMA_URL = st.sidebar.text_input("Ollama URL", value="http://localhost:11434")
MODEL = st.sidebar.text_input("Local model", value="qwen2.5:3b")
st.sidebar.markdown("---")
st.sidebar.caption("Your CSV is processed by this app. If Ollama is running locally, your questions and column names stay on your machine.")

def json_from_text(text: str) -> dict:
    """Parse a JSON object from an LLM response, tolerating surrounding code fences."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("The model did not return a JSON object.")
    result = json.loads(cleaned[start:end + 1])
    if not isinstance(result, dict):
        raise ValueError("Plan must be a JSON object.")
    return result

def build_prompt(question: str, df: pd.DataFrame) -> str:
    schema = []
    for col in df.columns:
        series = df[col]
        schema.append({
            "name": str(col),
            "dtype": str(series.dtype),
            "examples": [str(x)[:80] for x in series.dropna().head(3).tolist()],
            "numeric": bool(pd.api.types.is_numeric_dtype(series)),
            "datetime": bool(pd.api.types.is_datetime64_any_dtype(series)),
        })
    return f"""You convert a user's data question into a safe JSON plan. Do not write code.
Return exactly one JSON object with these keys:
{{
  "operation": "summary" | "aggregate" | "trend" | "anomalies",
  "group_by": string | null,
  "value_column": string | null,
  "aggregation": "sum" | "mean" | "count" | "min" | "max" | null,
  "date_column": string | null,
  "limit": integer
}}
Rules:
- Choose only one operation from the allowed list.
- Use exact column names from the schema, or null.
- summary: overall summary statistics; other fields may be null.
- aggregate: group rows by group_by and calculate aggregation on value_column. Use count if the user asks for counts.
- trend: group values over time using date_column and value_column; aggregation should be sum or mean.
- anomalies: find outliers in value_column using a robust IQR rule; group_by may optionally identify groups.
- limit must be an integer from 1 to 100, normally 20.
- Never include SQL, Python, expressions, extra keys, or instructions.
CSV schema:
{json.dumps(schema, ensure_ascii=False)}
User question: {question}
JSON plan:"""

def ask_ollama(question: str, df: pd.DataFrame) -> dict:
    prompt = build_prompt(question, df)
    response = requests.post(
        f"{OLLAMA_URL.rstrip('/')}/api/generate",
        json={"model": MODEL, "prompt": prompt, "stream": False, "format": "json"},
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    return json_from_text(payload.get("response", ""))


def fallback_plan(question: str, df: pd.DataFrame) -> dict:
    """Dependency-free planner for Streamlit Cloud when local Ollama is unavailable.
    It maps common plain-English patterns to the same allow-listed JSON plan schema.
    """
    q = question.lower()
    columns = list(map(str, df.columns))
    numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
    date_cols = [c for c in columns if
                 pd.api.types.is_datetime64_any_dtype(df[c]) or
                 any(token in c.lower() for token in ("date", "time", "month", "year"))]
    categorical_cols = [c for c in columns if c not in numeric_cols and c not in date_cols]

    def mentions(col: str) -> bool:
        return col.lower() in q

    value = next((c for c in numeric_cols if mentions(c)), None)
    if value is None and numeric_cols:
        # Common semantic aliases for typical business datasets.
        aliases = {
            "sales": ("sales", "revenue", "income", "turnover"),
            "profit": ("profit", "margin", "earnings"),
            "amount": ("amount", "price", "cost", "value", "transaction"),
        }
        for c in numeric_cols:
            cl = c.lower()
            if any(alias in q and alias in cl for alias in aliases.get(cl, ())):
                value = c
                break
        if value is None:
            value = numeric_cols[0]

    date_col = next((c for c in date_cols if mentions(c)), date_cols[0] if date_cols else None)
    group_col = next((c for c in categorical_cols if mentions(c)), None)
    if group_col is None and categorical_cols:
        # Common grouping words.
        preferred = ("region", "category", "product", "department", "segment", "country", "city", "status")
        group_col = next((c for p in preferred for c in categorical_cols if p in c.lower()), categorical_cols[0])

    agg = "mean" if any(w in q for w in ("average", "mean", "avg")) else "count" if any(w in q for w in ("count", "number of", "how many")) else "sum"
    if any(w in q for w in ("anomal", "outlier", "unusual", "abnormal", "unexpected")):
        operation = "anomalies"
        if value is None:
            raise ValueError("I couldn't find a numeric column for anomaly detection.")
        return {"operation": operation, "group_by": group_col, "value_column": value,
                "aggregation": "sum", "date_column": date_col, "limit": 20}
    if any(w in q for w in ("trend", "over time", "over the", "monthly", "daily", "weekly", "by date", "timeline")):
        operation = "trend"
        if not date_col or not value:
            raise ValueError("Trend questions need a date column and a numeric value column.")
        return {"operation": operation, "group_by": None, "value_column": value,
                "aggregation": agg if agg in {"sum", "mean", "count"} else "sum",
                "date_column": date_col, "limit": 100}
    if any(w in q for w in ("summary", "statistics", "describe", "overview", "dataset profile")):
        return {"operation": "summary", "group_by": None, "value_column": None,
                "aggregation": None, "date_column": None, "limit": 20}
    if any(w in q for w in ("by ", "group", "region", "category", "product", "highest", "lowest", "top", "total", "average", "mean", "count")):
        if not group_col or not value:
            return {"operation": "summary", "group_by": None, "value_column": None,
                    "aggregation": None, "date_column": None, "limit": 20}
        return {"operation": "aggregate", "group_by": group_col, "value_column": value,
                "aggregation": agg, "date_column": None, "limit": 20}
    return {"operation": "summary", "group_by": None, "value_column": None,
            "aggregation": None, "date_column": None, "limit": 20}

def validate_plan(plan: dict, df: pd.DataFrame) -> dict:
    allowed_keys = {"operation", "group_by", "value_column", "aggregation", "date_column", "limit"}
    if set(plan) - allowed_keys:
        raise ValueError("The plan contains unsupported fields.")
    operation = plan.get("operation")
    if operation not in {"summary", "aggregate", "trend", "anomalies"}:
        raise ValueError("Unsupported operation.")
    columns = set(map(str, df.columns))
    for key in ("group_by", "value_column", "date_column"):
        val = plan.get(key)
        if val is not None and val not in columns:
            raise ValueError(f"Unknown column for {key}: {val}")
    aggregation = plan.get("aggregation")
    if aggregation not in {None, "sum", "mean", "count", "min", "max"}:
        raise ValueError("Unsupported aggregation.")
    try:
        limit = int(plan.get("limit", 20))
    except (TypeError, ValueError):
        limit = 20
    plan["limit"] = max(1, min(limit, 100))
    if operation == "aggregate":
        if not plan.get("group_by") or not plan.get("value_column"):
            raise ValueError("Aggregate needs group_by and value_column.")
        if aggregation is None:
            plan["aggregation"] = "sum"
    if operation == "trend":
        if not plan.get("date_column") or not plan.get("value_column"):
            raise ValueError("Trend needs date_column and value_column.")
        if aggregation not in {"sum", "mean", "count"}:
            plan["aggregation"] = "sum"
    if operation == "anomalies" and not plan.get("value_column"):
        raise ValueError("Anomaly detection needs a numeric value_column.")
    if operation in {"aggregate", "trend", "anomalies"}:
        value = plan.get("value_column")
        if value and plan["aggregation"] != "count" and not pd.api.types.is_numeric_dtype(df[value]):
            raise ValueError(f"'{value}' is not numeric. Try a numeric column.")
    return plan

def execute_plan(plan: dict, df: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
    """Execute only fixed, reviewed operations. No model-generated code is evaluated."""
    op = plan["operation"]
    value = plan.get("value_column")
    agg = plan.get("aggregation") or "sum"
    limit = plan["limit"]

    if op == "summary":
        numeric = df.select_dtypes(include="number")
        if numeric.empty:
            result = pd.DataFrame({"message": ["No numeric columns found."]})
        else:
            result = numeric.describe().T.reset_index(names="column")
        return result, "Dataset summary", "table"

    if op == "aggregate":
        group = plan["group_by"]
        if agg == "count":
            result = df.groupby(group, dropna=False)[value].count().reset_index(name="count")
            y = "count"
        else:
            result = df.groupby(group, dropna=False)[value].agg(agg).reset_index(name=f"{agg}_{value}")
            y = f"{agg}_{value}"
        result = result.sort_values(y, ascending=False).head(limit)
        return result, f"{agg.title()} of {value} by {group}", "bar"

    if op == "trend":
        date_col = plan["date_column"]
        work = df[[date_col, value]].copy()
        work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
        work[value] = pd.to_numeric(work[value], errors="coerce")
        work = work.dropna(subset=[date_col, value])
        if work.empty:
            raise ValueError("No valid date/value rows found. Check the selected date and value columns.")
        result = work.groupby(date_col, dropna=False)[value].agg(agg).reset_index()
        result = result.sort_values(date_col).tail(limit)
        result.columns = [date_col, f"{agg}_{value}"]
        return result, f"{agg.title()} trend for {value}", "line"

    if op == "anomalies":
        work = df.copy()
        work[value] = pd.to_numeric(work[value], errors="coerce")
        work = work.dropna(subset=[value])
        if work.empty:
            raise ValueError("No numeric values available for anomaly detection.")
        q1, q3 = work[value].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        result = work[(work[value] < low) | (work[value] > high)].copy()
        result["anomaly_reason"] = result[value].apply(
            lambda x: "Below IQR range" if x < low else "Above IQR range"
        )
        result = result.sort_values(value, ascending=False).head(limit)
        result.attrs["bounds"] = (float(low), float(high))
        return result, f"Potential outliers in {value}", "table"

    raise ValueError("Unsupported operation.")

def make_summary(question: str, plan: dict, result: pd.DataFrame) -> str:
    op = plan["operation"]
    if op == "summary":
        return f"Calculated summary statistics for the numeric columns. The table contains descriptive statistics such as count, mean, standard deviation, and quartiles."
    if op == "aggregate":
        if result.empty:
            return "No groups were returned for this calculation."
        value_col = result.columns[-1]
        top = result.iloc[0]
        return f"Grouped **{plan['value_column']}** by **{plan['group_by']}** using **{plan['aggregation']}**. The highest displayed group is **{top[plan['group_by']]}** with a value of **{top[value_col]:,.4g}**."
    if op == "trend":
        if result.empty:
            return "No valid observations were available for the trend."
        y = result.columns[-1]
        first, last = result.iloc[0], result.iloc[-1]
        direction = "increased" if last[y] > first[y] else "decreased" if last[y] < first[y] else "was unchanged"
        return f"The displayed {plan['aggregation']} trend for **{plan['value_column']}** {direction} from **{first[y]:,.4g}** to **{last[y]:,.4g}** across the selected period."
    if op == "anomalies":
        bounds = result.attrs.get("bounds")
        return f"Flagged **{len(result)}** potential outlier row(s) using the 1.5×IQR rule for **{plan['value_column']}**. These are candidates for review, not proof of an error."
    return "Analysis completed."

uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded:
    try:
        df = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read this CSV: {exc}")
        st.stop()

    if df.empty:
        st.warning("This CSV has no rows.")
        st.stop()

    st.subheader("Dataset preview")
    a, b, c = st.columns(3)
    a.metric("Rows", f"{len(df):,}")
    b.metric("Columns", f"{len(df.columns):,}")
    c.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
    st.dataframe(df.head(100), use_container_width=True)

    with st.expander("Column profile"):
        profile = pd.DataFrame({
            "column": df.columns.astype(str),
            "dtype": [str(x) for x in df.dtypes],
            "missing": [int(df[col].isna().sum()) for col in df.columns],
            "unique values": [int(df[col].nunique(dropna=True)) for col in df.columns],
        })
        st.dataframe(profile, use_container_width=True)

    question = st.text_input(
        "Ask a question about your data",
        placeholder="e.g. Which region has the highest total sales? / Show monthly revenue trend / Find unusual order amounts",
    )
    if st.button("Analyze", type="primary", disabled=not question.strip()):
        with st.spinner("Creating a safe analysis plan..."):
            try:
                try:
                    raw_plan = ask_ollama(question, df)
                    planner_mode = "Ollama local model"
                except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                    raw_plan = fallback_plan(question, df)
                    planner_mode = "Built-in free fallback (Ollama unavailable)"
                plan = validate_plan(raw_plan, df)
                st.caption(f"Planner: {planner_mode}. The calculation is performed by Pandas.")
                st.subheader("Analysis plan")
                st.json(plan)
                result, title, chart_type = execute_plan(plan, df)
                st.subheader(title)
                if chart_type in {"bar", "line"} and not result.empty:
                    x_col = result.columns[0]
                    y_col = result.columns[-1]
                    if chart_type == "bar":
                        fig = px.bar(result, x=x_col, y=y_col, hover_data=list(result.columns), title=title)
                    else:
                        fig = px.line(result, x=x_col, y=y_col, markers=True, hover_data=list(result.columns), title=title)
                    fig.update_layout(xaxis_title=x_col, yaxis_title=y_col)
                    st.plotly_chart(fig, use_container_width=True)
                st.markdown("### Plain-English insight")
                st.markdown(make_summary(question, plan, result))
                st.dataframe(result, use_container_width=True)
                st.download_button(
                    "Download results as CSV",
                    data=result.to_csv(index=False).encode("utf-8"),
                    file_name="analysis_results.csv",
                    mime="text/csv",
                )
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to Ollama. Start Ollama, pull a model, and try again. See README for setup.")
            except requests.exceptions.Timeout:
                st.error("The local model took too long to respond. Try a smaller model such as qwen2.5:3b.")
            except requests.exceptions.HTTPError as exc:
                st.error(f"Ollama returned an HTTP error: {exc}")
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                st.error(f"Could not safely run the analysis: {exc}")
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")
else:
    st.info("Upload a CSV to get started.")
    st.markdown("**Try asking:**")
    st.markdown("- What are the summary statistics for this dataset?\n- Which category has the highest total sales?\n- Show the trend of revenue over time.\n- Find anomalous values in transaction_amount.")
