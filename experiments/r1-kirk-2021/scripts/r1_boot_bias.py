import sys, os
sys.path.insert(0, r"c:/Users/podgortsev123/Documents/Projects/average-fit-research/github/paper-8-the-floor-test/experiments/r1-kirk-2021/scripts")
import pandas as pd, numpy as np
from multiprocessing import Pool
import r1_analyse as A
if __name__ == "__main__":
    A._init("S:/r/kirk")
    theirs = pd.read_csv("S:/r/kirk/data/GPT-2/NER_output/identity_occupations_template.csv", index_col=0)
    sub = A.social_subset(theirs, "religion")
    point = A.interaction_pct(sub, "religion")["mean"]
    with Pool(12, initializer=A._init, initargs=("S:/r/kirk",)) as pool:
        boots = pool.map(A._boot_one, [(sub, "religion", 777 + i) for i in range(24)])
    b = np.array(boots)
    print(f"their religion point estimate {point:.3f}")
    print(f"24 bootstrap resamples: median {np.median(b):.3f}, min {b.min():.3f}, max {b.max():.3f}")
    print(f"share of resamples above the point estimate: {np.mean(b > point):.2f}")
    print("sorted:", np.round(np.sort(b), 3).tolist())
