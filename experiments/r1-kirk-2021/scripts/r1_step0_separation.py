import contextlib, io, os, sys
sys.path.insert(0, r"c:/Users/podgortsev123/Documents/Projects/average-fit-research/github/paper-8-the-floor-test/experiments/r1-kirk-2021/scripts")
import pandas as pd
import r1_step0_replicate as R
repo = "S:/r/kirk"
R.compatibility_shim()
ns = R.load_their_functions(repo, ["Generate_Freq_Matrices.py", "Logistic_Regression.py"])
freq = R.build_freq_matrix(repo, ns)
reg = ns["make_regression_df"](freq, None, "ethnicity", "range", 140.0)
print("journalist counts by category x gender:")
print(reg.groupby(["category", "gender"])["journalist"].sum().unstack().to_string())
print("model counts by category x gender:")
print(reg.groupby(["category", "gender"])["model"].sum().unstack().to_string())
pub = pd.read_csv(os.path.join(repo, "regression_results", "ethnicity_logistic_regressions.csv"), index_col=0)
pub_jobs = {c[:-7] for c in pub.columns if c.endswith(":p<0.05")}
for job in [c for c in reg.columns if c not in ("gender", "category")]:
    with contextlib.redirect_stdout(io.StringIO()):
        res = ns["run_regression_interaction"](reg, job, "base", "man")
    ok = res is not None
    if ok != (job in pub_jobs):
        print(f"fit differs: {job:22} ours {'fitted' if ok else 'failed'}, published {'present' if job in pub_jobs else 'absent'}")
with contextlib.redirect_stdout(io.StringIO()):
    est = ns["run_regression_interaction"](reg, "journalist", "base", "man")[0]
print("journalist, our fit:")
print(pd.DataFrame({"coef": est.params, "se": est.bse, "p": est.pvalues}).round(4).to_string())
