"""
Diagnose 20 Newsgroups coverage collapse.
The benchmark trains on comp domain, calibrates on comp domain,
then evaluates on all test domains. Coverage of 0.0 on soc/talk
means the conformal threshold is set wrong or the model has no
support for those classes.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from reliability.conformal import ConformalClassifier

train_bundle = fetch_20newsgroups(subset='train', remove=('headers','footers','quotes'))
test_bundle  = fetch_20newsgroups(subset='test',  remove=('headers','footers','quotes'))

X_train_text = np.asarray(train_bundle.data, dtype=object)
y_train_all  = np.asarray(train_bundle.target, dtype=int)
X_test_text  = np.asarray(test_bundle.data, dtype=object)
y_test_all   = np.asarray(test_bundle.target, dtype=int)
target_names = train_bundle.target_names

def topic_prefix(targets, names):
    return np.asarray([names[int(y)].split('.')[0] for y in targets], dtype=object)

train_domains = topic_prefix(y_train_all, target_names)
test_domains  = topic_prefix(y_test_all,  target_names)

cal_domain = 'comp'
rng = np.random.default_rng(42)

cal_pool = np.where(train_domains == cal_domain)[0]
cal_idx  = rng.choice(cal_pool, size=min(len(cal_pool), 8000), replace=False)
train_pool = np.setdiff1d(np.arange(len(y_train_all)), cal_idx)
train_idx  = rng.choice(train_pool, size=min(len(train_pool), 12000), replace=False)

vectorizer = TfidfVectorizer(max_features=20000, stop_words='english', min_df=2)
X_tr = vectorizer.fit_transform(X_train_text[train_idx].tolist())
y_tr = y_train_all[train_idx]
X_cal = vectorizer.transform(X_train_text[cal_idx].tolist())
y_cal = y_train_all[cal_idx]

model = LogisticRegression(max_iter=700, random_state=42, solver='lbfgs')
model.fit(X_tr, y_tr)

# Check what classes the model knows
print(f"Model classes: {model.classes_}")
print(f"n_classes: {len(model.classes_)}")
print(f"Cal label range: {y_cal.min()}..{y_cal.max()}, unique: {np.unique(y_cal)}")

conformal = ConformalClassifier(model=model, alpha=0.10, score='softmax')
conformal.calibrate(X_cal, y_cal)

print(f"\nConformal threshold (q_hat): {conformal._q_hat:.4f}")
print()

for domain in sorted(np.unique(test_domains)):
    idx = np.where(test_domains == domain)[0][:4000]
    X_d = vectorizer.transform(X_test_text[idx].tolist())
    y_d = y_test_all[idx]
    cov = float(conformal.empirical_coverage(X_d, y_d))
    probs = model.predict_proba(X_d)
    max_prob = float(np.max(probs, axis=1).mean())
    print(f"domain={domain:6s}  n={len(idx):4d}  coverage={cov:.3f}  mean_max_prob={max_prob:.3f}  label_range={y_d.min()}..{y_d.max()}")
