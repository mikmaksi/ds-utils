from dataclasses import dataclass, field
from itertools import combinations
from math import ceil, sqrt
from numbers import Real
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as pn
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from qs_take_home.helpers.utils import validate_transform_output
from scipy.cluster.hierarchy import dendrogram, leaves_list
from scipy.stats import iqr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import AgglomerativeClustering, FeatureAgglomeration
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, SelectorMixin, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import scale
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted


class ScaledVarianceThreshold(SelectorMixin, BaseEstimator):
    _parameter_constraints: dict = {"threshold": [Interval(Real, 0, None, closed="left")]}

    def __init__(self, threshold=0.0, pre_normed: bool = False):
        self.threshold = threshold
        self.pre_normed = pre_normed

    def fit(self, X, y=None):
        # validate data to be able to use .get_feature_names_out
        X = self._validate_data(
            X,
            accept_sparse=("csr", "csc"),
            dtype=np.float64,
            force_all_finite="allow-nan",
        )
        if not self.pre_normed:
            X = scale(X, with_mean=True, with_std=True)
        self.variances_ = iqr(X, axis=0)
        return self

    def _get_support_mask(self):
        check_is_fitted(self)
        mask = self.variances_ > self.threshold
        return mask

    @property
    def filtered_features_(self):
        check_is_fitted(self)
        return np.array(list(set(self.feature_names_in_) - set(self.get_feature_names_out())))

    def plot_filtered_hist(self, X: pd.DataFrame):
        """Visualize the distributions of the filtered metrics."""
        check_is_fitted(self)
        x = X.copy()
        if not self.pre_normed:
            x = scale(x, with_mean=True, with_std=True)
        x = pd.DataFrame(x, columns=x.columns)
        p = pn.ggplot(data=x[self.filtered_features_].melt(), mapping=pn.aes("value"))
        p = p + pn.geom_histogram(binwidth=0.25)
        p = p + pn.geom_rug(size=1, color="blue")
        p = p + pn.facet_wrap("~variable", scales="free")
        p = p + pn.theme_gray(base_size=16)
        return p


class CorrFeatureAgglomeration(FeatureAgglomeration):
    @staticmethod
    def pool_by_pca(X, axis=1):
        # if a single feature comes in, return it directly
        if X.shape[1] == 1:
            return X.reshape(X.shape[0])

        # if multiple features, return the first PCA component
        reduced = PCA(n_components=1).fit_transform(X).to_numpy()
        return reduced.reshape(reduced.shape[0])

    def __init__(self, r: float = 0.9):
        """__init__.

        Args:
            r (float): maximum correlation coefficient allowed between a pair of predictors.
        """
        # inputs
        self.r = r
        super().__init__(
            n_clusters=None,
            metric="precomputed",
            distance_threshold=1 - self.r,
            linkage="complete",
            pooling_func=self.pool_by_pca,
        )

        # set attributes
        self.metric_clusters = []

    def fit(self, X, y=None):
        dist_mat = 1 - X.corr().abs()
        dist_mat[dist_mat < 0] = 0
        super().fit(dist_mat)

        # set metric clusters
        for i, cluster_id in enumerate(np.unique(self.labels_)):
            self.metric_clusters.append(
                MetricCluster(
                    cluster_id=cluster_id,
                    selected=self.get_feature_names_out()[i],
                    all_metrics=X.columns[self.labels_ == cluster_id],
                )
            )

        return self

    @property
    def leaf_order(self):
        return self.feature_names_in_[leaves_list(self.linkage_matrix)].tolist()

    @property
    def linkage_matrix(self) -> np.ndarray:
        """Create a matrix representing the linkage from an AgglomerativeClustering model
            source: https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html

        Args:
            model (cluster.AgglomerativeClustering): an agglomerative clustering model
        """
        # create the counts of samples under each node
        counts = np.zeros(self.children_.shape[0])
        n_samples = len(self.labels_)
        for i, merge in enumerate(self.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        return np.column_stack([self.children_, self.distances_, counts]).astype(float)

    @property
    def correlated_pairs(self) -> list[tuple]:
        # make a list of pairs to plot
        pairs = []
        for metric_cluster in self.metric_clusters:
            if len(metric_cluster.all_metrics) > 1:
                pairs.extend(list(combinations(metric_cluster.all_metrics, 2)))
        return pairs

    @property
    def out_in_feature_map(self) -> dict[str, list[str]]:
        feature_map = {}
        for metric_cluster in self.metric_clusters:
            feature_map.setdefault(
                metric_cluster.selected,
                metric_cluster.all_metrics.tolist(),
            )
        return feature_map


@dataclass
class MetricCluster:
    cluster_id: int
    selected: str
    all_metrics: list[str]


class IterativeCorrFeatureAgglomeration(BaseEstimator, TransformerMixin):
    def __init__(self, r: float = 0.9):
        self.r = r
        self.steps: list[CorrFeatureAgglomeration] = []

    def fit(self, X, y=None):
        is_correlated = True
        Xr = X.copy()
        step_num = 1
        while is_correlated:
            # run step
            step = CorrFeatureAgglomeration(r=self.r)
            step.fit(Xr)
            Xr = step.transform(Xr)

            # check remaining correlation
            is_correlated = np.any(np.triu(Xr.corr().abs(), k=1) > self.r)

            # update steps
            self.steps.append(step)
            step_num += 1
        return self

    def transform(self, X, y=None):
        X = X.copy()
        for step in self.steps:
            X = step.transform(X)
        return X

    @staticmethod
    def sub_dict_values(first: dict, second: dict):
        result = {}
        for k, values in first.items():
            new_values = []
            for v in values:
                new_values.extend(second[v])
            result.setdefault(k, new_values)
        return result

    @property
    def out_in_feature_map(self) -> dict[str, list[str]]:
        feature_map_list = [step.out_in_feature_map for step in self.steps]
        comb = feature_map_list[-1]
        for step in feature_map_list[:-1][::-1]:
            comb = self.sub_dict_values(comb, step)
        return comb


class FeatureAgglomerationDisplay:
    def __init__(
        self, X, transformer: Union[Union[CorrFeatureAgglomeration, IterativeCorrFeatureAgglomeration], Pipeline]
    ):
        """If PIpeline then Agglomeration transformer must be the last step"""

        # set attributes
        self.X = X
        self.preproc = None
        if isinstance(transformer, Pipeline):
            self.transformer = transformer[-1]
            self.preproc = transformer[:-1]
        else:
            self.transformer = transformer

    @property
    def _X(self):
        if self.preproc is None:
            return self.X
        else:
            return self.preproc.transform(self.X)

    def get_step(self, step_num: int = 0):
        if isinstance(self.transformer, IterativeCorrFeatureAgglomeration):
            return self.transformer.steps[step_num]
        else:
            if step_num > 0:
                raise Exception("CorrFeatureAgglomeration is a single step transformer.")
            return self.transformer

    def plot_dendrogram(self, step_num: int = 0) -> Figure:
        """Create diagnostic plot with a clustering dendrogram and correlation heatmap.

        Args:
            X:

        Returns:
        """
        fig, ax = plt.subplots()
        step: ClusteringStep = self.get_step(step_num)
        dendrogram(
            step.linkage_matrix,
            ax=ax,
            color_threshold=1 - step.r,
            orientation="right",
            labels=step.leaf_order,
        )
        fig.tight_layout()
        return fig

    def plot_in_corr_map(self, step_num: int = 0) -> Figure:
        # apply preprocess if necessary
        X = self._X.copy()

        # transform if more than one step
        for i in range(step_num):
            X = self.get_step(i).transform(X)
        fig = sns.clustermap(X.corr(), annot=True, fmt=".2f")
        return fig

    def plot_out_corr_map(self) -> Figure:
        # apply preprocess if necessary
        X = self.transformer.transform(self._X)
        fig = sns.clustermap(X.corr(), annot=True, fmt=".2f")
        return fig

    def plot_bivariate(self, X, correlated: bool = True) -> Figure:
        """
        TODO: this has not been adapted or tested
        Make a list of pairs to plot of correlated pairs from each clustering step.
        """
        if correlated:
            pairs = sum([step.get_correlated_pairs() for step in self.clustering_steps_], [])
        else:
            pairs = list(combinations(self.get_feature_names_out(), r=2))

        # create a plot
        nrows = ceil(sqrt(len(pairs)))
        if nrows != 0:
            fig, axs = plt.subplots(nrows=nrows, ncols=ceil(len(pairs) / nrows))
            if isinstance(axs, Axes):
                pair = pairs[0]
                sns.scatterplot(data=X[list(pair)], x=pair[0], y=pair[1], ax=axs)
            else:
                for i, (pair, ax) in enumerate(zip(pairs, axs.flat)):
                    sns.scatterplot(data=X[list(pair)], x=pair[0], y=pair[1], ax=ax)
            fig.tight_layout()
        else:
            fig, axs = plt.subplots()
        return fig
