"""Run with:  python -m pip install pytest   then   pytest -q"""
import matching

df = matching.load_data("europe_scholarships_matching.csv")


def names(result):
    return [r["name"] for r in result["results"]]


def test_degree_filter_excludes_wrong_level():
    res = matching.recommend(df, "bachelor", 3.5, 4, 50000)
    assert "Chevening Scholarships" not in names(res)  # master only


def test_budget_never_exceeded():
    res = matching.recommend(df, "master", 3.5, 4, 3000)
    assert all(r["out_of_pocket_usd_per_year"] <= 3000 for r in res["results"])
    assert "Holland Scholarship" not in names(res)  # leaves a large gap


def test_published_minimum_excludes_low_percentage():
    res = matching.recommend(df, "master", 70, 100, 50000)
    assert "Turkiye Burslari" not in names(res)  # published minimum is about 75%


def test_low_cgpa_never_target_or_safer_for_high_competition():
    res = matching.recommend(df, "master", 2.5, 4, 50000)
    high = {"Chevening Scholarships", "Gates Cambridge Scholarship", "Eiffel Excellence Scholarship"}
    for r in res["results"]:
        if r["name"] in high:
            assert r["fit"] == "reach"


def test_invalid_inputs_return_error():
    assert "error" in matching.recommend(df, "diploma", 3.5, 4, 5000)
    assert "error" in matching.recommend(df, "master", 5.5, 4, 5000)


def test_cgpa_conversion():
    assert matching.to_cgpa_4(3.4, 4)[0] == 3.4
    assert matching.to_cgpa_4(4.0, 5)[0] == 3.2
