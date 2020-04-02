import numpy as np
import pandas as pd
import click
import pickle
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier


def features(dec, cand, quantiles, rank_intervalls, min_pairs, max_pairs, wikidata_gt=None, stat_funcs=None):

    if stat_funcs is None:
        stat_funcs = ['min', 'max', 'mean', 'std', 'median']

    data = list()

    # normalize rank, i.e, rank is afterwards in between [0,1]
    cand['rank'] = [r / (len(cand) - 1 if len(cand) > 1 else 1) for r in range(0, len(cand))]
    dec = dec.sort_values('scores', ascending=False)

    # merge per candidate information with per sentence pair information
    cols_to_use = list(cand.columns.difference(dec.columns)) + ['guessed_title']
    dec = dec.merge(cand[cols_to_use], left_on='guessed_title', right_on='guessed_title')

    for wikidata, part in dec.groupby('wikidata'):

        # compute statistics for ALL OTHER sentence pairs that have been evaluated for ALL OTHER candidates
        dec_values = dec.loc[dec.wikidata != wikidata].select_dtypes(exclude=['object'])
        overall = pd.concat([dec_values.apply(stat_funcs), dec_values.quantile(q=quantiles)])

        # here we compute per single candidate statistics

        # compute relative number of occurences among the first X-percent of best sentence pairs (rank information)
        occur = dec.wikidata == wikidata
        cum_occur = pd.DataFrame(occur.cumsum() / occur.sum())
        cum_occur['pos'] = [pos / (len(cum_occur) - 1 if len(cum_occur) > 1 else 1) for pos in range(len(cum_occur))]

        pos_stat = [cum_occur.loc[cum_occur.pos < p].wikidata.max() for p in rank_intervalls]
        pos_stat = pd.DataFrame(pos_stat, index=rank_intervalls, columns=['among_top'])

        # compute other statistical descriptors
        part = part.select_dtypes(exclude=['object'])

        repeats = 1
        if len(part) > max_pairs and wikidata == wikidata_gt:

            repeats = int(np.ceil((len(part) - max_pairs) / 10) + 1)

        for i in range(repeats):

            if len(part) > max_pairs:
                rnd = np.random.permutation(len(part))[0:np.random.randint(min_pairs, max_pairs)]
                part_subset = part.iloc[rnd]
            else:
                part_subset = part

            case = pd.concat([part_subset.apply(stat_funcs), part_subset.quantile(q=quantiles)])

            # compute also difference between overall statistics and candidate statistics
            diff = case - overall

            # rename columns such that they are more meaningful for interpretation
            case.columns = ['case_' + str(col) for col in case.columns]
            overall.columns = ['overall_' + str(col) for col in overall.columns]
            diff.columns = ['diff_' + str(col) for col in diff.columns]

            # join all the statistical information
            statistics = pd.concat([case.unstack(), overall.unstack(), diff.unstack(), pos_stat.unstack()])

            statistics['label'] = float(int(wikidata == wikidata_gt)) if wikidata_gt is not None else None
            statistics['wikidata_gt'] = wikidata_gt

            data.append(statistics)

    return data


@click.command()
@click.argument('data-file', type=click.Path(exists=True), required=True, nargs=1)
@click.argument('model-file', type=click.Path(exists=False), required=True, nargs=1)
@click.option('--n-jobs', type=int, default=8, help='default: 8.')
def train(data_file, model_file, n_jobs):

    df = pd.read_pickle(data_file)

    X = df.apply(pd.to_numeric, errors='ignore')

    X_columns = X.columns.difference(['label', 'wikidata_gt'])

    y = X['label']
    X = X[X_columns]

    k_fold = GroupKFold(n_splits=10)

    estimator = RandomForestClassifier()

    X[X.isnull()] = 0

    scores = cross_val_score(estimator, X, y, cv=k_fold, scoring='roc_auc', n_jobs=n_jobs,
                             groups=df.wikidata_gt)

    estimator.fit(X, y)

    imp = pd.DataFrame(estimator.feature_importances_, index=X.columns, columns=['importance']).\
        sort_values('importance', ascending=False)

    import ipdb;
    ipdb.set_trace()

    with open(model_file, 'wb') as fw:
        pickle.dump(estimator, fw)

    # df.wikidata_gt.astype('category').cat.codes

    print("Mean CV-ROC-AUC: {}".format(np.mean(scores)))
    print(imp.head(50))
