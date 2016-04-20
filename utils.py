import numpy as np
import pandas as pd

def run_cv_tests(data, target, skf, model):
    for train_index, test_index in skf:
        X_train, X_test = peak_data.iloc[train_index], peak_data.iloc[test_index]
        y_train, y_test = target[train_index], target[test_index]

        impt = X_train.mean().copy()
        X_train = X_train.fillna(impt)
        X_test = X_test.fillna(impt)

        scl = StandardScaler()
        X_train_scl = scl.fit_transform(X_train)
        X_test_scl = scl.transform(X_test)

        ecv = ElasticNetCV(l1_ratio=[0., .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.], 
                           eps=None, n_alphas=None,
                           alphas=[.01, .05, .1, .5, 1., 5., 10., 50., 100.], 
                           cv=4, n_jobs=-1, random_state=42)
        ecv.fit(X_train_scl, y_train)
        preds = ecv.predict(X_test_scl)

        print(roc_auc_score(y_test, preds))

        lr = LogisticRegression(random_state=42, n_jobs=-1)
        lr.fit(X_train_scl, y_train)
        preds = lr.predict_proba(X_test_scl)[:, 1]

        print(roc_auc_score(y_test, preds))
        
def log_progress(sequence, every=None, size=None):
    from ipywidgets import IntProgress, HTML, VBox
    from IPython.display import display

    is_iterator = False
    if size is None:
        try:
            size = len(sequence)
        except TypeError:
            is_iterator = True
    if size is not None:
        if every is None:
            if size <= 200:
                every = 1
            else:
                every = size / 200     # every 0.5%
    else:
        assert every is not None, 'sequence is iterator, set every'

    if is_iterator:
        progress = IntProgress(min=0, max=1, value=1)
        progress.bar_style = 'info'
    else:
        progress = IntProgress(min=0, max=size, value=0)
    label = HTML()
    box = VBox(children=[label, progress])
    display(box)

    index = 0
    try:
        for index, record in enumerate(sequence, 1):
            if index == 1 or index % every == 0:
                if is_iterator:
                    label.value = '{index} / ?'.format(index=index)
                else:
                    progress.value = index
                    label.value = u'{index} / {size}'.format(
                        index=index,
                        size=size
                    )
            yield record
    except:
        progress.bar_style = 'danger'
        raise
    else:
        progress.bar_style = 'success'
        progress.value = index
        label.value = str(index or '?')