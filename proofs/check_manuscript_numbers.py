"""Print every number quoted in arxiv_submission/main.tex from the result files.

Run from the repository root:
    python proofs/check_manuscript_numbers.py [results_dir] [logs_dir]
Defaults: experiments/results and repro_out/logs. Prints the rounded values
exactly as the manuscript quotes them so the two can be compared line by line.
"""
import csv, json, re, sys, os
from scipy.stats import norm

root = os.path.join(os.path.dirname(__file__), "..")
res = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "experiments", "results")
logs = sys.argv[2] if len(sys.argv) > 2 else os.path.join(root, "repro_out", "logs")

def rows(p):
    with open(p) as fh:
        return list(csv.DictReader(fh))

print("== Table 2 / abstract: tv_gap.csv")
tv = rows(os.path.join(res, "tv_gap.csv"))
ratios = [float(r["delta_over_tv"]) for r in tv if r["delta_over_tv"] != "nan"]
print(f"delta/TV range: {min(ratios):.4f} .. {max(ratios):.4f}  -> quote as {round(100*min(ratios))}--{round(100*max(ratios))}%")
for r in tv:
    print(f"  bins={r['n_bins']:>2} mu={float(r['mu']):.2f} TV={float(r['true_tv']):.3f} "
          f"delta={float(r['mean_delta_hat']):.4f} hyp={100*float(r['hypothesis_tv_le_2delta_rate']):.0f}% "
          f"c2viol={100*float(r['c2_bound_violation_rate']):.0f}% loss={float(r['mean_coverage_loss']):.4f}")
print("true TV formula 2*Phi(mu/2)-1 at mu=0.25,0.5,0.75,1:",
      [round(2*norm.cdf(m/2)-1, 3) for m in (0.25, 0.5, 0.75, 1.0)])

for name, label in (("camelyon_fallback", "Table 3 Covertype"), ("amazon_fallback", "Table 4 20NG")):
    print(f"== {label}: wilds_{name}_domains.csv / _summary.json")
    for r in rows(os.path.join(res, f"wilds_{name}_domains.csv")):
        print(f"  {r['domain_name']:<13} n={r['n_samples']:>5} cov={float(r['coverage']):.3f} "
              f"ECE={float(r['ece']):.3f} delta={float(r['delta']):.3f} loss={float(r['coverage_loss']):.4f}")
    s = json.load(open(os.path.join(res, f"wilds_{name}_summary.json")))["summary"]
    print(f"  means: cov={s['coverage_mean']:.4f} loss={s['coverage_loss_mean']:.4f} ECE={s['ece_mean']:.4f} "
          f"delta={s['delta_mean']:.4f}; pearson r={s['pearson_r_loss_delta']:.3f} p={s['pearson_p_loss_delta']:.3f}; "
          f"spearman={s['spearman_r_loss_delta']:.2f} p={s['spearman_p_loss_delta']:.2f}; "
          f"n_cal={s['n_cal']} n_train={s['n_train']}")
    eces = [float(r["ece"]) for r in rows(os.path.join(res, f"wilds_{name}_domains.csv"))]
    print(f"  ECE range {min(eces):.3f}..{max(eces):.3f}")

print("== Theorem-1 study: theorem1.log MANUSCRIPT_SUMMARY")
t1 = open(os.path.join(logs, "theorem1.log")).read()
m = dict(re.findall(r"(\w+)=([\d.]+)", t1.split("MANUSCRIPT_SUMMARY")[-1]))
n = int(float(m["total_valid_trials"]))
print(f"  c1 viol {m['c1_violations']}/{n} = {100*float(m['c1_violation_rate']):.1f}%  "
      f"c2 viol {m['c2_violations']}/{n} = {100*float(m['c2_violation_rate']):.2f}%  r={float(m['pearson_r']):.3f}")

print("== Second synthetic experiment: drift.log")
d = open(os.path.join(logs, "drift.log")).read()
tab = re.findall(r"^\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+OK", d, re.M)
d1 = {float(a): float(dd) for a, _, _, dd, _ in tab}
r = re.search(r"Pearson r\(coverage_loss, .*?\) = ([\d.]+)\s+\(p=([\d.]+)\)", d)
mu2 = float(re.search(r"mean delta = ([\d.]+)", d).group(1))
print(f"  r={float(r.group(1)):.3f} p={float(r.group(2)):.4f}; delta(mu=1)={d1[1.0]:.3f}; delta(mu=2)={mu2:.3f}; ratio={mu2/d1[1.0]:.2f}")

tlog = os.path.join(logs, "theorem1_table.log")
if os.path.exists(tlog):
    print("== Table 1: per-mu means from non-quiet theorem1 run")
    for line in open(tlog):
        if re.match(r"^\s*[01]\.\d{2}\s", line):
            print("  " + line.rstrip())
