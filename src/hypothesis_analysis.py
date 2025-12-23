# src/task3_tests.py
import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.multitest import multipletests

# ---------- Config ----------
PROCESSED_CSV = "../data/insurance_cleaned.csv"
OUT_DIR = "../data/results/task3"
TOP_N_ZIPS = 10   # number of top zipcodes for pairwise tests
ALPHA = 0.05

# ---------- Utilities ----------
def ensure_out_dir():
    os.makedirs(OUT_DIR, exist_ok=True)

def load_data(path=PROCESSED_CSV):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed CSV not found: {path}\nRun preprocessing to create it.")
    df = pd.read_csv(path, parse_dates=[c for c in ["TransactionMonth","VehicleIntroDate"] if c in pd.read_csv(path, nrows=0).columns], dtype={"PostalCode": str})
    # Derived columns
    df["HasClaim"] = (df["TotalClaims"] > 0).astype(int)
    df["ClaimSeverity"] = df["TotalClaims"].where(df["HasClaim"] == 1, np.nan)
    df["Margin"] = df["TotalPremium"] - df["TotalClaims"]
    return df

def effect_size_proportion(p1, p2):
    return p1 - p2

def cohen_d(a, b):
    # compute Cohen's d (unbiased pooled std)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    ma, mb = np.mean(a), np.mean(b)
    sa, sb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = np.sqrt(((na-1)*sa + (nb-1)*sb) / (na+nb-2))
    if pooled == 0:
        return 0.0
    return (ma - mb) / pooled

# ---------- Province tests ----------
def test_provinces(df):
    results = {}
    # Frequency: chi-square
    ct = pd.crosstab(df["Province"], df["HasClaim"])
    chi2, p, dof, exp = stats.chi2_contingency(ct)
    results["freq_chi2"] = {"chi2": float(chi2), "p": float(p), "dof": int(dof)}
    # Severity: Kruskal-Wallis (non-parametric)
    sev = df[df["HasClaim"]==1].dropna(subset=["ClaimSeverity"])
    groups = [g["ClaimSeverity"].values for _, g in sev.groupby("Province") if len(g) > 0]
    if len(groups) >= 2:
        kw_stat, kw_p = stats.kruskal(*groups)
        results["severity_kruskal"] = {"stat": float(kw_stat), "p": float(kw_p)}
    else:
        results["severity_kruskal"] = {"stat": None, "p": None}
    return results, ct

# ---------- Zipcode tests ----------
def test_zipcodes(df, top_n=TOP_N_ZIPS):
    results = {}
    # overall frequency chi-square
    ct = pd.crosstab(df["PostalCode"], df["HasClaim"])
    try:
        chi2, p, dof, exp = stats.chi2_contingency(ct)
        results["freq_chi2_overall"] = {"chi2": float(chi2), "p": float(p), "dof": int(dof)}
    except Exception as e:
        results["freq_chi2_overall"] = {"error": str(e)}
    # pairwise tests for top N postal codes by count
    top_zips = df["PostalCode"].value_counts().nlargest(top_n).index.tolist()
    pairs = []
    for i in range(len(top_zips)):
        for j in range(i+1, len(top_zips)):
            za, zb = top_zips[i], top_zips[j]
            da = df[df["PostalCode"]==za]
            db = df[df["PostalCode"]==zb]
            na, nb = len(da), len(db)
            if na < 10 or nb < 10:
                continue
            ca, cb = da["HasClaim"].sum(), db["HasClaim"].sum()
            # two-proportion z-test
            stat, pval = proportions_ztest([ca, cb], [na, nb])
            # confidence intervals
            p1 = ca / na; p2 = cb / nb
            ci1_low, ci1_high = proportion_confint(ca, na, method="wilson")
            ci2_low, ci2_high = proportion_confint(cb, nb, method="wilson")
            # effect size (difference)
            prop_diff = p1 - p2
            # severity comparison (Mann-Whitney)
            sev_a = da.loc[da["HasClaim"]==1, "ClaimSeverity"].dropna()
            sev_b = db.loc[db["HasClaim"]==1, "ClaimSeverity"].dropna()
            if len(sev_a) >= 3 and len(sev_b) >= 3:
                try:
                    mw_stat, mw_p = stats.mannwhitneyu(sev_a, sev_b, alternative="two-sided")
                    d_sev = cohen_d(sev_a, sev_b)
                except Exception:
                    mw_stat, mw_p, d_sev = (np.nan, np.nan, np.nan)
            else:
                mw_stat, mw_p, d_sev = (np.nan, np.nan, np.nan)
            # margin t-test
            m_a = da["Margin"].dropna(); m_b = db["Margin"].dropna()
            if len(m_a) >= 3 and len(m_b) >= 3:
                try:
                    t_stat, t_p = stats.ttest_ind(m_a, m_b, equal_var=False, nan_policy="omit")
                    d_margin = cohen_d(m_a, m_b)
                except Exception:
                    t_stat, t_p, d_margin = (np.nan, np.nan, np.nan)
            else:
                t_stat, t_p, d_margin = (np.nan, np.nan, np.nan)

            pairs.append({
                "zip_a": za, "zip_b": zb,
                "n_a": int(na), "n_b": int(nb),
                "count_a": int(ca), "count_b": int(cb),
                "prop_a": float(p1), "prop_b": float(p2), "prop_diff": float(prop_diff),
                "prop_z_stat": float(stat), "prop_p": float(pval),
                "prop_ci_a": [float(ci1_low), float(ci1_high)], "prop_ci_b": [float(ci2_low), float(ci2_high)],
                "sev_mw_stat": mw_stat, "sev_p": mw_p, "sev_cohen_d": d_sev,
                "margin_t_stat": t_stat, "margin_p": t_p, "margin_cohen_d": d_margin
            })
    pairs_df = pd.DataFrame(pairs)
    # multiple testing correction for proportion p-values
    if not pairs_df.empty:
        pvals = pairs_df["prop_p"].fillna(1).values
        reject, pvals_adj, _, _ = multipletests(pvals, alpha=ALPHA, method="bonferroni")
        pairs_df["prop_p_adj"] = pvals_adj
        pairs_df["prop_reject"] = reject
    results["pairwise_top_zips"] = pairs_df
    return results, ct

# ---------- Gender tests ----------
def test_gender(df):
    results = {}
    ct = pd.crosstab(df["Gender"], df["HasClaim"])
    try:
        chi2, p, dof, exp = stats.chi2_contingency(ct)
        results["freq_chi2"] = {"chi2": float(chi2), "p": float(p), "dof": int(dof)}
    except Exception as e:
        results["freq_chi2"] = {"error": str(e)}
    # severity Mann-Whitney
    sev_m = df[(df["Gender"]=="Male") & (df["HasClaim"]==1)]["ClaimSeverity"].dropna()
    sev_f = df[(df["Gender"]=="Female") & (df["HasClaim"]==1)]["ClaimSeverity"].dropna()
    if len(sev_m) >= 3 and len(sev_f) >= 3:
        try:
            mw_stat, mw_p = stats.mannwhitneyu(sev_m, sev_f, alternative="two-sided")
            d_sev = cohen_d(sev_m, sev_f)
            results["severity_mw"] = {"stat": float(mw_stat), "p": float(mw_p), "cohen_d": float(d_sev)}
        except Exception:
            results["severity_mw"] = {"stat": None, "p": None, "cohen_d": None}
    else:
        results["severity_mw"] = {"stat": None, "p": None, "cohen_d": None}
    # margin t-test
    m_m = df[df["Gender"]=="Male"]["Margin"].dropna()
    m_f = df[df["Gender"]=="Female"]["Margin"].dropna()
    if len(m_m) >= 3 and len(m_f) >= 3:
        t_stat, t_p = stats.ttest_ind(m_m, m_f, equal_var=False, nan_policy="omit")
        results["margin_t"] = {"stat": float(t_stat), "p": float(t_p), "cohen_d": float(cohen_d(m_m, m_f))}
    else:
        results["margin_t"] = {"stat": None, "p": None, "cohen_d": None}
    return results, ct

# ---------- Runner ----------
def run_all(processed_csv=PROCESSED_CSV, top_n=TOP_N_ZIPS, alpha=ALPHA):
    ensure_out_dir()
    df = load_data(processed_csv)
    # sanity check required columns
    required = ["Province","PostalCode","Gender","TotalPremium","TotalClaims"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Required column missing: {c}")

    # Run tests
    prov_res, prov_ct = test_provinces(df)
    zip_res, zip_ct = test_zipcodes(df, top_n=top_n)
    gender_res, gender_ct = test_gender(df)

    # Save results
    prov_ct.to_csv(os.path.join(OUT_DIR, "province_hasclaim_ct.csv"))
    zip_ct.to_csv(os.path.join(OUT_DIR, "zipcode_hasclaim_ct.csv"))
    gender_ct.to_csv(os.path.join(OUT_DIR, "gender_hasclaim_ct.csv"))

    if not zip_res["pairwise_top_zips"].empty:
        zip_res["pairwise_top_zips"].to_csv(os.path.join(OUT_DIR, "zip_pairwise_top.csv"), index=False)

    summary = {
        "province": prov_res,
        "zipcode": {"overall": zip_res.get("freq_chi2_overall"), "pairwise_top_path": "zip_pairwise_top.csv" if not zip_res["pairwise_top_zips"].empty else None},
        "gender": gender_res
    }
    with open(os.path.join(OUT_DIR, "task3_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("\n=== TASK 3 SUMMARY ===")
    print("Province frequency chi2:", prov_res.get("freq_chi2"))
    print("Province severity kruskal:", prov_res.get("severity_kruskal"))
    print("Zipcode overall frequency chi2:", zip_res.get("freq_chi2_overall"))
    print("Gender frequency chi2:", gender_res.get("freq_chi2"))
    print("Gender severity:", gender_res.get("severity_mw"))
    print("Gender margin:", gender_res.get("margin_t"))
    print("\nSaved outputs to", OUT_DIR)
    return summary

if __name__ == "__main__":
    run_all()


   

