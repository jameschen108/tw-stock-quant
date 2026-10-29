import numpy as np
import pandas as pd
import pytest

import themes as T


def test_theme_returns_equal_weight_skips_missing():
    r = pd.DataFrame({"A": [0.10, 0.10], "B": [0.00, np.nan], "C": [0.5, 0.5]})
    out = T.theme_returns(r, ["A", "B", "X"])      # X 沒行情、C 不在主題
    assert out.tolist() == pytest.approx([0.05, 0.10])


def test_theme_returns_no_members():
    r = pd.DataFrame({"A": [0.1, 0.2]})
    assert T.theme_returns(r, ["X"]).isna().all()


def test_window_return_compounds_and_needs_full_window():
    d = pd.Series([0.1, 0.1, 0.1])
    assert T.window_return(d, 2) == pytest.approx(0.21)
    assert np.isnan(T.window_return(d, 5))


def test_load_themes_rejects_duplicates(tmp_path):
    p = tmp_path / "t.toml"
    p.write_text('[themes.x]\nname="x"\ncodes=["1101","1101"]\n')
    with pytest.raises(ValueError):
        T.load_themes(p)


def test_repo_themes_file_is_valid():
    t = T.load_themes()
    assert t and all(v["codes"] for v in t.values())
