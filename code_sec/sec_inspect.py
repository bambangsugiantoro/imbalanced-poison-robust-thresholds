import sys, openml, pandas as pd
for did in sys.argv[1:]:
    d = openml.datasets.get_dataset(int(did), download_qualities=False, download_features_meta_data=False)
    X, y, cat, names = d.get_data(target=d.default_target_attribute)
    print('==', did, d.name, X.shape, 'target', d.default_target_attribute, y.value_counts().to_dict())
    print(list(X.columns)[:40]); print('object cols:', [c for c in X.columns if X[c].dtype == 'O' or str(X[c].dtype) == 'category'][:25])
