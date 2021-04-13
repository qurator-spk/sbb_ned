import numpy as np
import pandas as pd
import click
import pickle
import json
import logging
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from collections import OrderedDict

logger = logging.getLogger(__name__)


class DeciderTask:

    decider = None
    entities = None

    def __init__(self, entity_id, decision, candidates, quantiles, rank_intervalls, threshold, return_full=False,
                 **kwargs):

        self._entity_id = entity_id
        self._decision = decision
        self._candidates = candidates
        self._quantiles = quantiles
        self._rank_intervalls = rank_intervalls
        self._threshold = threshold
        self._return_full = return_full

    def __call__(self, *args, **kwargs):

        if self._candidates is None:
            return self._entity_id, None

        decider_features = features(self._decision, self._candidates, self._quantiles, self._rank_intervalls)

        ranking = pd.DataFrame()
        if DeciderTask.decider is not None:
            prediction = predict(decider_features, DeciderTask.decider)

            prediction = self._candidates[['surface', 'guessed_title']].merge(prediction, on='guessed_title')

            ranking = prediction[(prediction.proba_1 >= self._threshold) |
                                 (prediction.guessed_title.str.lower() == prediction.surface.str.lower())].\
                sort_values(['proba_1', 'case_rank_min'], ascending=[False, True]).\
                drop_duplicates(['guessed_title']).set_index('guessed_title')

        result = dict()

        if len(ranking) > 0:
            ranking['wikidata'] = [DeciderTask.entities.loc[guessed_title, 'QID']
                                   for guessed_title, _ in ranking.iterrows()]

            result['ranking'] = [i for i in ranking[['proba_1', 'wikidata']].T.to_dict(into=OrderedDict).items()]

        if self._return_full:
            self._candidates['wikidata'] = [DeciderTask.entities.loc[v.guessed_title, 'QID']
                                            for _, v in self._candidates.iterrows()]

            decision = self._decision.merge(self._candidates[['guessed_title', 'wikidata']],
                                            left_on='guessed_title', right_on='guessed_title')

            result['decision'] = json.loads(decision.to_json(orient='split'))
            result['candidates'] = json.loads(self._candidates.to_json(orient='split'))

        return self._entity_id, result

    @staticmethod
    def initialize(decider, entities):

        DeciderTask.decider = decider
        DeciderTask.entities = entities


def features(dec, cand, quantiles, rank_intervalls, min_pairs=np.inf, max_pairs=np.inf,
             wikidata_gt=None, stat_funcs=None):

    if stat_funcs is None:
        stat_funcs = ['min', 'max', 'mean', 'std', 'median']

    data = list()
    cand = cand.copy()

    # normalize rank, i.e, rank is afterwards in between [0,1]
    cand['rank'] = [r / (len(cand) - 1 if len(cand) > 1 else 1) for r in range(0, len(cand))]
    dec = dec.sort_values('scores', ascending=False)

    # merge per candidate information with per sentence pair information
    cols_to_use = list(cand.columns.difference(dec.columns)) + ['guessed_title']
    dec = dec.merge(cand[cols_to_use], left_on='guessed_title', right_on='guessed_title')

    for guessed_title, part in dec.groupby('guessed_title'):

        # compute statistics for ALL OTHER sentence pairs that have been evaluated for ALL OTHER candidates
        dec_values = dec.loc[dec.guessed_title != guessed_title].select_dtypes(exclude=['object'])

        overall = pd.concat([dec_values.apply(stat_funcs), dec_values.quantile(q=quantiles)])

        # rename columns such that they are more meaningful for interpretation
        overall_renamed = pd.DataFrame(overall.values, index=overall.index,
                                       columns=['overall_' + str(col) for col in overall.columns])

        # here we compute per single candidate statistics

        # compute relative number of occurences among the first X-percent of best sentence pairs (rank information)
        occur = dec.guessed_title == guessed_title
        cum_occur = pd.DataFrame(occur.cumsum() / occur.sum())
        cum_occur['pos'] = [pos / (len(cum_occur) - 1 if len(cum_occur) > 1 else 1) for pos in range(len(cum_occur))]

        pos_stat = [cum_occur.loc[cum_occur.pos < p].guessed_title.max() for p in rank_intervalls]
        pos_stat = pd.DataFrame(pos_stat, index=rank_intervalls, columns=['among_top'])

        # compute other statistical descriptors
        wikidata = part.wikidata.iloc[0] if 'wikidata' in part.columns else None
        part = part.select_dtypes(exclude=['object'])

        repeats = 1
        if len(part) > max_pairs and wikidata is not None and wikidata == wikidata_gt:

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

            diff.columns = ['diff_' + str(col) for col in diff.columns]

            # join all the statistical information
            statistics = pd.concat([case.unstack(), overall_renamed.unstack(), diff.unstack(), pos_stat.unstack()])

            statistics['label'] = float(int(wikidata == wikidata_gt)) if wikidata_gt is not None and wikidata is not None else None
            statistics['wikidata_gt'] = wikidata_gt
            statistics['wikidata'] = wikidata
            statistics['guessed_title'] = guessed_title

            data.append(statistics)

    if len(data) < 1:
        return None

    data = pd.concat(data, axis=1).T

    data.columns = ["_".join([str(pa) for pa in col if len(str(pa)) > 0]) for col in data.columns]

    return data


@click.command()
@click.argument('data-file', type=click.Path(exists=True), required=True, nargs=1)
@click.argument('model-file', type=click.Path(exists=False), required=True, nargs=1)
@click.option('--n-jobs', type=int, default=8, help='default: 8.')
@click.option('--include-prefix', type=str, multiple=True, default=[])
@click.option('--exclude-prefix', type=str, multiple=True, default=[])
def train(data_file, model_file, n_jobs, include_prefix, exclude_prefix):

    df = pd.read_pickle(data_file)

    X = df.apply(pd.to_numeric, errors='ignore')

    X_columns = X.columns.difference(
        ['label', 'wikidata_gt', 'wikidata', 'guessed_title'])

    if len(include_prefix) > 0:
        X_columns = [c for c in X_columns
                    if any([c.startswith(p) for p in include_prefix])]

    if len(exclude_prefix) > 0:
        X_columns =[c for c in X_columns 
                    if not any([c.startswith(p) for p in exclude_prefix])]

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

    with open(model_file, 'wb') as fw:
        pickle.dump(estimator, fw)

    # df.wikidata_gt.astype('category').cat.codes

    print("Mean CV-ROC-AUC: {}".format(np.mean(scores)))
    print(imp.head(50))


def predict(df, estimator, include_prefix=None, exclude_prefix=None):

    meta_columns = ['label', 'wikidata_gt', 'wikidata', 'guessed_title']

    if include_prefix is None:
        include_prefix = []

    if exclude_prefix is None:
        exclude_prefix = []

    X = df.apply(pd.to_numeric, errors='ignore')

    X_columns = X.columns.difference(meta_columns)

    if len(include_prefix) > 0:
        X_columns = [c for c in X_columns if any([c.startswith(p) for p in include_prefix])]

    if len(exclude_prefix) > 0:
        X_columns = [c for c in X_columns if not any([c.startswith(p) for p in exclude_prefix])]

    X = X[X_columns]

    X[X.isnull()] = 0

    proba = estimator.predict_proba(X)

    df['proba_0'] = proba[:, 0]
    df['proba_1'] = proba[:, 1]

    return df


@click.command()
@click.argument('data-file', type=click.Path(exists=True), required=True, nargs=1)
@click.argument('model-file', type=click.Path(exists=True), required=True, nargs=1)
@click.argument('output-file', type=click.Path(exists=False), required=True, nargs=1)
@click.option('--include-prefix', type=str, multiple=True, default=[])
@click.option('--exclude-prefix', type=str, multiple=True, default=[])
def test(data_file, model_file, output_file, include_prefix, exclude_prefix):

    df = pd.read_pickle(data_file)

    with open(model_file, 'rb') as fr:
        estimator = pickle.load(fr)

    df = predict(df, estimator, include_prefix, exclude_prefix)

    df.to_pickle(output_file)

