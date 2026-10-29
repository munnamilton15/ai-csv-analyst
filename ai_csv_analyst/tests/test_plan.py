import pandas as pd
import pytest
from app import validate_plan, execute_plan

@pytest.fixture
def sample():
    return pd.DataFrame({
        "region": ["North", "South", "North", "West"],
        "sales": [10, 20, 15, 1000],
        "date": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"],
    })

def test_rejects_unknown_operation(sample):
    with pytest.raises(ValueError):
        validate_plan({"operation": "__import__", "limit": 10}, sample)

def test_rejects_unknown_column(sample):
    with pytest.raises(ValueError):
        validate_plan({
            "operation": "aggregate", "group_by": "not_a_column",
            "value_column": "sales", "aggregation": "sum", "limit": 10
        }, sample)

def test_aggregate(sample):
    plan = validate_plan({
        "operation": "aggregate", "group_by": "region",
        "value_column": "sales", "aggregation": "sum", "limit": 20
    }, sample)
    result, _, _ = execute_plan(plan, sample)
    assert result.loc[result["region"] == "North", "sum_sales"].iloc[0] == 25

def test_anomaly_detection(sample):
    plan = validate_plan({
        "operation": "anomalies", "value_column": "sales",
        "group_by": None, "aggregation": None, "date_column": None, "limit": 20
    }, sample)
    result, _, _ = execute_plan(plan, sample)
    assert 1000 in result["sales"].values
