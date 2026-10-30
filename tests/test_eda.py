from fraud import eda
from fraud.data import FEATURES, TARGET


def test_eda_plots_write_figures_and_return_numbers(frame, tmp_path):
    frame.loc[frame[TARGET] == 1, "V14"] -= 3.0  # give one feature a clear signal
    balance = eda.plot_class_balance(frame, tmp_path)
    assert balance["fraud"] == int(frame[TARGET].sum())
    assert 0 < balance["fraud_rate"] < 1

    amounts = eda.plot_amount_time(frame, tmp_path)
    assert set(amounts.index) == {0, 1}

    assert eda.top_separating_features(frame, k=3)[0] == "V14"
    assert eda.plot_top_features(frame, k=4, out_dir=tmp_path)[0] == "V14"
    corr = eda.plot_correlation(frame, tmp_path)
    assert corr.index[0] == "V14" and set(corr.index) == set(FEATURES)

    by_hour = eda.plot_fraud_by_hour(frame, tmp_path)
    assert by_hour["n"].sum() == len(frame) and by_hour.index.max() <= 23

    names = {p.name for p in tmp_path.iterdir()}
    assert names == {
        "class_balance.png",
        "amount_time_by_class.png",
        "top_features_by_class.png",
        "correlation_heatmap.png",
        "fraud_rate_by_hour.png",
    }
