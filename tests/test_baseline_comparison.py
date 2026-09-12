import pandas as pd

from validation.baseline_comparison import (
    calculate_realized_establishment,
    calculate_realized_outcome,
    estimate_initial_moisture,
    load_data,
)


def test_estimate_initial_moisture_ignores_future_rainfall():
    data = load_data()

    decision_date = pd.Timestamp("2024-06-15")

    baseline = estimate_initial_moisture(
        data=data,
        decision_date=decision_date,
        soil_type="medium_black",
    )

    modified = data.copy()

    future_mask = (
        modified["date"] >= decision_date
    )

    modified.loc[
        future_mask,
        "rainfall_mm",
    ] = 999.0

    result = estimate_initial_moisture(
        data=modified,
        decision_date=decision_date,
        soil_type="medium_black",
    )

    assert result == baseline


def test_estimate_initial_moisture_uses_pre_decision_window():
    data = load_data()

    decision_date = pd.Timestamp("2024-06-15")

    result = estimate_initial_moisture(
        data=data,
        decision_date=decision_date,
        soil_type="medium_black",
    )

    assert 0.0 <= result <= 60.0


def test_realized_establishment_uses_held_out_rainfall():
    data = load_data()

    decision_date = pd.Timestamp("2024-06-15")

    import src.backtesting as b

    future = b.get_actual_future_data(
        data,
        decision_date,
        14,
    )

    initial_moisture = estimate_initial_moisture(
        data=data,
        decision_date=decision_date,
        soil_type="medium_black",
    )

    result = calculate_realized_establishment(
        actual_future=future,
        crop_name="cotton",
        soil_type="medium_black",
        initial_moisture_mm=initial_moisture,
    )

    assert result in (0.0, 1.0)


def test_realized_outcome_wait_uses_future_only_for_evaluation():
    data = load_data()

    decision_date = pd.Timestamp("2024-06-15")

    import src.backtesting as b

    future = b.get_actual_future_data(
        data,
        decision_date,
        14,
    )

    initial_moisture = estimate_initial_moisture(
        data=data,
        decision_date=decision_date,
        soil_type="medium_black",
    )

    result = calculate_realized_outcome(
        decision="WAIT 5 DAYS",
        actual_future=future,
        soil_type="medium_black",
        initial_moisture_mm=initial_moisture,
    )

    assert result in (0.0, 1.0)