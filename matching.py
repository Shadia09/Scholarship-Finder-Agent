"""
Matching logic for the Europe scholarship agent.
No LLM calls here: all filtering, budget math and CGPA conversion happen in code.
"""
import pandas as pd

NOT_RECORDED = "Not recorded - check the official website"
FIT_ORDER = {"target": 0, "safer": 1, "reach": 2}


def load_data(path: str = "europe_scholarships_matching.csv") -> pd.DataFrame:
    df = pd.read_csv(path).fillna("")
    for col in ["est_annual_cost_usd", "est_annual_support_usd", "suggested_competitive_cgpa_4"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def normalize_degree(text: str):
    t = (text or "").lower()
    if "phd" in t or "doctor" in t:
        return "phd"
    if "master" in t or "msc" in t or "postgrad" in t:
        return "master"
    if "bachelor" in t or "undergrad" in t or "bsc" in t:
        return "bachelor"
    return None


def to_cgpa_4(cgpa: float, scale: float):
    """Convert to an approximate 4.0 scale. Returns (value, note)."""
    if scale in (4, 4.0):
        return float(cgpa), "Used as given (4.0 scale)."
    if scale in (5, 5.0, 10, 10.0):
        return round(cgpa / scale * 4, 2), (
            f"Converted linearly from a {scale:g}-point scale; this is approximate."
        )
    if scale == 100:
        table = [(85, 4.0), (80, 3.7), (75, 3.3), (70, 3.0), (65, 2.7), (60, 2.3)]
        value = next((v for p, v in table if cgpa >= p), 2.0)
        return value, "Converted from a percentage using a rough table; approximate."
    return round(cgpa / scale * 4, 2), "Converted linearly; approximate."


def fit_label(cgpa_4: float, suggested) -> str:
    if pd.isna(suggested):
        return "target"
    gap = cgpa_4 - suggested
    if gap >= 0.2:
        return "safer"
    if gap >= -0.1:
        return "target"
    return "reach"


def _to_result(row, out_of_pocket, fit, warnings):
    return {
        "name": row["scholarship_name"],
        "country": row["country"],
        "fit": fit,
        "coverage_level": row["coverage_level"],
        "coverage_summary": row["coverage_summary"],
        "out_of_pocket_usd_per_year": int(out_of_pocket),
        "suggested_competitive_cgpa_4": None if pd.isna(row["suggested_competitive_cgpa_4"])
        else float(row["suggested_competitive_cgpa_4"]),
        "academic_note": row["academic_note"],
        "english_requirement": row["english_requirement_note"],
        "work_experience": row["work_experience_note"],
        "eligible_nationalities": row["eligible_nationalities"],
        "warnings": warnings,
        "deadline": row["deadline"] or NOT_RECORDED,
        "official_website": row["official_website"],
        "data_confidence": row["confidence"],
    }


def recommend(df, degree, cgpa, cgpa_scale, budget_usd, nationality=None, country=None, limit=8):
    degree_n = normalize_degree(degree)
    if degree_n is None:
        return {"error": "Degree level not recognised. Use bachelor, master, or phd."}
    if cgpa_scale <= 0 or cgpa < 0 or cgpa > cgpa_scale:
        return {"error": "CGPA must be between 0 and the CGPA scale."}

    cgpa_4, conversion_note = to_cgpa_4(cgpa, cgpa_scale)
    results, over_budget = [], []

    for _, row in df.iterrows():
        if degree_n not in row["degree_levels"]:
            continue
        if country and country.lower() not in row["country"].lower():
            continue

        warnings = []

        # Published minimum is the only thing allowed to exclude a student on grades
        official = row["min_grade_percent_official"]
        if official != "":
            if cgpa_scale == 100:
                if cgpa < float(official):
                    continue
            else:
                warnings.append(
                    f"Published minimum is about {official}%. Check how your CGPA converts to a percentage."
                )

        elig = str(row["eligible_nationalities"])
        if any(k in elig.lower() for k in ("only", "partner", "check", "income", "varies")):
            who = f" (you said: {nationality})" if nationality else ""
            warnings.append(f"Check the nationality rules{who}: {elig}")

        cost, support = row["est_annual_cost_usd"], row["est_annual_support_usd"]
        out_of_pocket = max(0, (cost if pd.notna(cost) else 0) - (support if pd.notna(support) else 0))
        fit = fit_label(cgpa_4, row["suggested_competitive_cgpa_4"])
        item = _to_result(row, out_of_pocket, fit, warnings)

        (results if out_of_pocket <= budget_usd else over_budget).append(item)

    results.sort(key=lambda r: (r["out_of_pocket_usd_per_year"], FIT_ORDER[r["fit"]]))
    over_budget.sort(key=lambda r: r["out_of_pocket_usd_per_year"])

    out = {
        "student_cgpa_4_used": cgpa_4,
        "conversion_note": conversion_note,
        "budget_usd_per_year": budget_usd,
        "count": len(results),
        "results": results[:limit],
        "note": "Costs and support amounts are rough estimates. Fit labels are heuristics, not admission chances.",
    }
    if not results:
        out["closest_but_over_budget"] = over_budget[:3]
    return out
