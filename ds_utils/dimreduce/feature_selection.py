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


@dataclass
class MetricCluster:
    cluster_id: int
    selected: str
    all_metrics: list[str]


@dataclass
class ClusteringStep:
    step: int
    model: AgglomerativeClustering = None
    metric_clusters: list[MetricCluster] = field(default_factory=list)
    linkage_matrix = None
    leaf_order = None

    @property
    def selected_metrics(self) -> list[str]:
        # put selected metrics into a list
        mets = [c.selected for c in self.metric_clusters]

        # sort the metrics by the leaf order from clusterin
        # note: metrics is a subset of leaf order
        mets = sorted(mets, key=lambda x: self.leaf_order.index(x))
        return mets

    def cluster(self, X: pd.DataFrame, r: float = 0.9) -> None:
        """Cluster the features in X based n Pearson correlation

        Args:
            r (float): Pearson correlation coefficient threshold for similarity (0, 1).

        Returns:
            None:
        """
        # calculate the distance matrix using Pearson correlation
        dist_mat = 1 - X.corr().abs()
        dist_mat[dist_mat < 0] = 0

        # cluster based on precomputed distance matrix
        self.model = AgglomerativeClustering(
            distance_threshold=1 - r, n_clusters=None, metric="precomputed", linkage="complete"
        )
        self.model.fit(dist_mat)

        # set the linkage matrix and link order
        self.set_linkage_matrix(self.model)
        self.leaf_order = X.columns[leaves_list(self.linkage_matrix)].tolist()

    def select(self, X: pd.DataFrame, y: pd.Series, score_func: callable = f_classif) -> None:
        """Select the single best metric from each cluster based on a univariate statistical test.

        Args:
            score_func (callable): one of a list of sklearn support scoring functions for classification
                chi2, f_classif, mutual_info_classif

        Returns:
            None:
        """

        for cluster_id in np.unique(self.model.labels_):
            selector = SelectKBest(score_func, k=1)
            selector.fit(X.iloc[:, np.where(self.model.labels_ == cluster_id)[0]], y)
            self.metric_clusters.append(
                MetricCluster(
                    cluster_id=cluster_id,
                    selected=selector.get_feature_names_out()[0],
                    all_metrics=selector.feature_names_in_.tolist(),
                )
            )

    def set_linkage_matrix(self, model: AgglomerativeClustering, **kwargs) -> None:
        """Create a matrix representing the linkage from an AgglomerativeClustering model
            source: https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html

        Args:
            model (cluster.AgglomerativeClustering): an agglomerative clustering model
        """
        # create the counts of samples under each node
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        self.linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)

    def get_correlated_pairs(self) -> list[tuple]:
        # make a list of pairs to plot
        pairs = []
        for metric_cluster in self.metric_clusters:
            if len(metric_cluster.all_metrics) > 1:
                pairs.extend(list(combinations(metric_cluster.all_metrics, 2)))
        return pairs


class AgglomerativeSelector(SelectorMixin, BaseEstimator):
    def __init__(self, r: float = 0.9, score_func: callable = f_classif):
        """__init__.

        Args:
            X (pd.DataFrame): a DataFrame of predictors optinally indexed by observation index
            y (pd.Series): as Series of y labels optionally indexed by observation index
        """

        # inputs
        self.r = r
        self.score_func = score_func

        # outputs
        self.clustering_steps_: list[ClusteringStep] = []
        self.fig_diag = None

    def fit(self, X, y) -> None:
        """Main loop of iterative dimensionality reduction

        Args:
            r (float): threshold for Pearson correlation coefficient
            score_func (callable): score_func to be passed to the feature selector

        Returns:
            None:
        """
        # check that pandas output was set
        validate_transform_output(self)

        # validate data to be able to use .get_feature_names_out
        X = self._validate_data(
            X,
            accept_sparse=("csr", "csc"),
            dtype=np.float64,
            force_all_finite="allow-nan",
        )

        # transform back to DataFrame
        X = pd.DataFrame(X, columns=self.feature_names_in_)

        i = 1
        not_done = True
        X_r = X.copy()
        while not_done:
            # print(f"Running clustering loop: {i}")

            # create a clustering step
            clustering_step = ClusteringStep(step=i)

            # cluster features
            clustering_step.cluster(X_r, self.r)

            # select the best features from each cluster
            clustering_step.select(X_r, y, self.score_func)

            # add to list
            self.clustering_steps_.append(clustering_step)

            # reduce input X
            X_r = X_r[self.get_feature_names_out()]

            # check if complete
            not_done = np.any(np.triu(X_r.corr().abs(), k=1) > self.r)
            i += 1
        return self

    def _get_support_mask(self):
        if self.clustering_steps_ == []:
            mask = self.feature_names_in_ == self.feature_names_in_
        else:
            selected_metrics = self.clustering_steps_[-1].selected_metrics
            mask = np.isin(self.feature_names_in_, selected_metrics)
        return mask

    @property
    def filtered_features_(self):
        check_is_fitted(self)
        return np.array(list(set(self.feature_names_in_) - set(self.get_feature_names_out())))

    def plot_clust_heatmaps(self, X) -> Figure:
        """Create diagnostic plot with a clustering dendrogram and correlation heatmap.

        Args:
            None:

        Returns:
            None:
        """
        # create figure
        fig, axs = plt.subplots(nrows=3, ncols=1, height_ratios=(1, 0.8, 0.8))

        # visualize the dendrogram for the initial clustering
        step1 = self.clustering_steps_[0]
        dendrogram(
            step1.linkage_matrix,
            ax=axs[0],
            color_threshold=1 - self.r,
            orientation="right",
            labels=X.columns,
        )
        fig.tight_layout()

        # visualize the heatmap for non reduced X
        sns.heatmap(X.corr().loc[step1.leaf_order, step1.leaf_order], ax=axs[1])

        # visualize the correlation map of reduced X for the final step
        # note: selected_metrics are in leaf order
        last_step = self.clustering_steps_[-1]
        sns.heatmap(
            X.corr().loc[last_step.selected_metrics, last_step.selected_metrics],
            ax=axs[2],
        )

        # set figure size
        fig.set_size_inches((6, 10))
        return fig

    def plot_bivariate(self, X, correlated: bool = True) -> Figure:
        """Make a list of pairs to plot of correlated pairs from each clustering step."""
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

    def plot_pairs(self, X, color: pd.Series = None) -> sns.PairGrid:
        g = sns.PairGrid(X)
        g.map_diag(sns.histplot)
        g.map_lower(sns.scatterplot, hue=color)
        g.map_upper(plot_r_coef)
        return g


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

    def fit(self, X, y=None):
        corr_mat = 1 - X.corr().abs()
        super().fit(corr_mat)
        return self

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


class IterativeCorrFeatureAgglomeration(BaseEstimator, TransformerMixin):
    def __init__(self, r: float = 0.9):
        self.r = r
        self.steps = []

    def fit(self, X, y=None):
        not_done = True
        Xr = X.copy()
        while not_done:
            step = CorrFeatureAgglomeration(r=self.r)
            step.fit(Xr)
            Xr = step.transform(Xr)
            not_done = np.any(np.triu(Xr.corr().abs(), k=1) > self.r)
            self.steps.append(step)
        return self

    def transform(self, X, y=None):
        X = X.copy()
        for step in self.steps:
            X = step.transform(X)
        return X


class FeatureAgglomerationDisplay:
    def __init__(
        self, X, transformer: Union[Union[CorrFeatureAgglomeration, IterativeCorrFeatureAgglomeration], Pipeline]
    ):
        """If PIpeline then Agglomeration transformer must be the last step
        """

        # set attributes
        self.X = X
        self.preproc = None
        if isinstance(transformer, Pipeline):
            self.transformer = transformer[-1]
            self.preproc = transformer[:-1]
        else:
            self.transformer = transformer

    def get_cluster_step(self, step_num: int = None):
        if isinstance(self.transformer, IterativeCorrFeatureAgglomeration):
            if step_num is None:
                raise Exception("'step_num' required with IterativeCorrFeatureAgglomeration")
            return self.transformer.steps[step_num]
        else:
            if step_num is not None:
                raise Exception("'step_num' not useable with CorrFeatureAgglomeration")
            return self.transformer

    @property
    def _X(self):
        if self.preroc is not None:
            return self.X
        else:
            return self.preproc.transform(self.X)

    def plot_dendrogram(self, step_num: int = None) -> Figure:
        """Create diagnostic plot with a clustering dendrogram and correlation heatmap.

        Args:
            X:

        Returns:
        """
        fig, ax = plt.subplots()
        step = self.get_cluster_step(step_num)
        dendrogram(
            step.linkage_matrix,
            ax=ax,
            color_threshold=1 - step.r,
            orientation="right",
            # labels=self.X.columns,
        )
        fig.tight_layout()
        return fig

    def plot_in_corr_map(self, X) -> Figure:
        fig = sns.clustermap(X.corr(), annot=True, fmt=".2f")
        return fig

    def plot_out_corr_map(self, X) -> Figure:
        Xr = self.transform(X)
        fig = sns.clustermap(Xr.corr(), annot=True, fmt=".2f")
        return fig

    def plot_bivariate(self, X, correlated: bool = True) -> Figure:
        """Make a list of pairs to plot of correlated pairs from each clustering step."""
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
