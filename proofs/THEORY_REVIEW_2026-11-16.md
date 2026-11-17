# Theory review of the arXiv v1 source, 2026-11-16

Scope: every numbered result in `arxiv_submission/main.tex` (promoted from
`paper/main_v2_draft.tex` the same day), every appendix proof, and every numerical
claim in Sections 4 and 5. Each proof was re-derived line by line. Numerical claims
were recomputed from `experiments/results/` and `repro_out/logs/` with
`proofs/check_manuscript_numbers.py`, and the full `reproduce.sh` was rerun (see
the repository README for the reproduction command).

Numbering is the PDF's after the fixes below (the new Corollary 2 shifted the two
propositions to 3 and 4). Pages refer to the 7-page build of 2026-11-16.

## Verdict

No result is false. Two statements needed an assumption or a definition made
explicit (B), one empirical claim needed its construction stated (B), and the rest
are wording and rounding (C). The withdrawn "coverage degrades by at most 2ε"
claim of the Oct 5 source does not appear anywhere in the v1 source; the paper
proves the coefficient-1 bound, shows it is sharp, and shows that the deficiency
estimate does not bound TV. The session brief's description of the core result as
a 2ε bound predates the 2026-10-28 withdrawal (git 4030f38) and is superseded.

## Result-by-result

| Result (page) | Status | Derivation / notes |
|---|---|---|
| Thm 1, coverage transfer (p. 2) | OK after B1, B2 | Fix the calibration set so `B` is a fixed event. `P_test(B) - P_cal(B) <= sup_A |P_cal(A) - P_test(A)| = TV` with TV = sup convention. With `eta_n = max(0, P_cal(B) - alpha)`, `P_test(B) <= alpha + eta_n + TV`. Coefficient 1 is correct under the sup convention; under the L1 convention (`||P-Q||_1`, values in [0,2]) the same bound reads `TV_L1 / 2`, so a coefficient 2 never arises in either convention for *this* statement. |
| Cor 2, marginal form (p. 2) | OK (new) | `P(miscover) = E[P_test(B_cal)] <= E[P_cal(B_cal)] + TV`, since the conditional inequality holds for every realised calibration set and the TV term does not depend on it. `E[P_cal(B_cal)] <= alpha` is the split conformal guarantee (quantile index `ceil((1-alpha)(n+1))`, test pair exchangeable with the calibration pairs). Independence of the test pair from the calibration set is needed so that `P_test(B_cal)` is the conditional miscoverage. This is the only place exchangeability is used; the test law is never assumed exchangeable with anything. |
| Prop 3, coefficient 1 unimprovable (p. 2; proof App. A, p. 6) | OK | `P_test = (alpha+eps) P_cal(.|B) + (1-alpha-eps) P_cal(.|B^c)` is a probability measure for `eps <= 1-alpha` and needs `0 < alpha < 1` so both conditionals exist. `P_test - P_cal = eps (P_cal(.|B) - P_cal(.|B^c))`; the two conditionals have disjoint supports (`B`, `B^c`), so `sup_A` of the difference is attained at `A = B` and equals `eps`. `P_test(B) = alpha + eps`, coverage `1 - alpha - eps`. The bound is therefore sharp for every `alpha` and every `eps in [0, 1-alpha]`; this is the two-point example the brief asked for, with the two points being `B` and `B^c`. |
| Prop 4(1), `delta <= max_theta TV(P_theta, Q_theta)` (p. 2–3; App. B, p. 6) | OK after B3 | Take `K = I`; requires `P_theta` and `Q_theta` on a common sample space, which the estimator guarantees by binning both samples with the calibration edges. Now stated. |
| Prop 4(2), disjoint supports give `delta = 0` (p. 2–3; App. B) | OK | Column `x` of `K` set to `Q_theta` for `x in S_theta`; then `K P_theta = sum_x P_theta(x) K(., x) = Q_theta` because `P_theta` charges only `S_theta`. Arbitrary stochastic columns elsewhere keep `K` column-stochastic. Checked numerically by `tests/test_deficiency_vs_tv.py` (passes). |
| Remark 1, risk transfer vs fixed-set coverage (p. 2) | OK | Risk transfer with sup-TV and losses in [0,1] has constant 1 (Le Cam; same as Def 3 / item M1 of `../CausalTheory/JMLR_Submission_2026-11/THEORY_REVIEW_2026-11-16.md`). The matched rule is `C_alpha o K`, not `C_alpha`, so no fixed-set statement follows; the remark is right that a "2ε" form appears only under the L1 normalisation. |
| Sec 4, noise-floor bullet | OK after C4 | `Y = 1[X_1 > 0]` makes the continuous class-conditional supports disjoint (`X_1 < 0` vs `X_1 > 0`), so Prop 4(2) gives population deficiency 0 for the *continuous* experiment. The binned experiment is not exactly zero (the bin straddling 0 mixes classes), which is what the text calls a discretisation effect. Wording now says "continuous class-conditional experiment". |
| Sec 5.1, true joint TV `2 Phi(mu/2) - 1` | OK | `Y` is a function of `X` under both laws, so joint TV equals the TV of the `X` marginals; TV between `N(0, I)` and `N(mu e_1, I)` is `2 Phi(mu/2) - 1`. Values 0.099, 0.197, 0.292, 0.383 reproduced by `check_manuscript_numbers.py`. |

## Convention cross-check with the sibling theory paper

`../CausalTheory/JMLR_Submission_2026-11/THEORY_REVIEW_2026-11-16.md` records the sup
convention for TV and risk-transfer constant `B` for losses in `[0, B]`. This paper
uses the same convention (now stated once in Section 3), the same deficiency
definition (`min_K max_theta TV(K P_theta, Q_theta)`, sup-TV), and the same
constant 1 for losses in [0,1]. No difference needs to be stated in the text.

## Findings and fixes

Severity A (false or unproved): none.

Severity B (missing assumption or definition), all fixed in `arxiv_submission/main.tex`:

- **B1 (Thm 1).** `eta_n` was never defined and the theorem was phrased as an
  implication on an undefined quantity. Now: calibration set fixed, `eta_n` defined
  as the realised conditional slack.
- **B2 (Sec 3).** The TV convention was stated only for finite vectors inside the
  deficiency definition while Theorem 1 is about general laws and its proof uses the
  sup form. One sentence now fixes `TV = sup_A |P(A) - Q(A)| = (1/2) ||P - Q||_1`
  for the whole paper. Added Corollary 2 so that the role of exchangeability is
  explicit (marginal form, `eta_n = 0`).
- **B3 (Prop 4(1)).** `K = I` needs a common sample space; stated.
- **B4 (Sec 5.4, 20 Newsgroups).** `experiments/wilds_benchmark.py` line 328 defines
  coverage loss as `max(0, 0.9 - coverage)`, which is zero for six of the seven
  domains. The quoted `r = -0.90 (p = 0.006)` was therefore a correlation of a
  one-nonzero-entry vector with `delta_hat`. The text now states the clipping, gives
  the unclipped value (`r = -0.88`, from `proofs/check_manuscript_numbers.py` and the
  inline check on 2026-11-16), and drops the p-value emphasis. The Covertype
  correlation is unaffected (all four losses positive).

Severity C (wording and rounding), applied:

- C1 abstract: "4–9%" of true TV → "4–8%" (`delta_over_tv` ranges 0.0390–0.0786 in `tv_gap.csv`).
- C2 Sec 5.1: `delta_hat = 0.327` at `mu = 2` → `0.326` (`drift.log`: 0.3263).
- C3 Sec 5.4: ECE range "0.24–0.42" → "0.20–0.42" (`alt` has ECE 0.201).
- C4 Sec 4 and 5.2: "population deficiency is zero" → "deficiency of the continuous class-conditional experiment is zero", with a pointer to Prop 4(2).
- C5 abstract and contributions: "We release ... open-source package" → "implemented in an accompanying Python package (MIT licence)". No public GitHub repository or release exists (`api.github.com/repos/denizakdemir/ReliabilityLayer` returns 404 on 2026-11-16).
- C6 title restored to "Deficiency as a Shift-Severity Monitor for Conformal Predictors" (the Oct 5 title and the one in the session brief; the v2 draft had shortened it).
- C7 related work: one-sentence distinctions for weighted conformal (needs the covariate likelihood ratio and unchanged `P(Y|X)`), adaptive conformal (online feedback control, not a fixed-predictor bound), and beyond-exchangeability (weighted sum of one-swap TV distances; Theorem 1 is the elementary two-law case).
- C8 Sec 4: solver sentence now matches `defkit/solvers.py` (`scipy.optimize.linprog`, HiGHS, sparse constraints, interior point when the LP has more than 5000 kernel entries, which includes every benchmark run).
- C9 one overfull box (Sec 4 definition of `P_{b,k}`) moved to display math.

## Open

- Sharpness is settled (Prop 3). Whether some *other* label-aware statistic bounds
  the joint TV from samples is open and is stated as such in the conclusion.
