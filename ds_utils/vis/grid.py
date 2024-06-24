#!python3

from typing import Literal

import numpy as np
import pandas as pd
import plotnine as pn
from sklearn.base import BaseEstimator


class VisGrid:
    def __init__(self, model: BaseEstimator, X: pd.DataFrame, n_points: int, target_name: str = "y"):
        self.model = model
        self.X = X
        self.n_points = n_points
        self.target_name = target_name

    def make_grid_avg_predictors(self, agg_func: Literal["mean", "median"] = "mean") -> pd.DataFrame:
        """Make a grid varying each predictor along an even linear spacing, while holding other predictors constant
        at their average (mean or median) value.

        Simple and fast to compute, but loss of realistic context, i.e. the aveage values of other predictors may
        not represent realistic combinations of feature values in the dataset.

        Args:
            agg_func (Literal["mean", "median"]): aggregation function to calculate set points for held-out predictors.

        Summary:

            For each predictor:
                1. Create a vector of evenly spaced values between its min and max observed values in the dataset.
                2. Set other predictors to their respective typical values.
                3. Generate predictions using @self.model.

        Return:
            df: DataFrame in long format with ["variable", "value", "ypred"] columns
                - variable: name of the varied predictor
                - value: value of the predictor
                - "ypred": predicted value of the response variable
        """
        # loop over columns
        grid = []
        for col in self.X.columns:
            # generate equally spaced samples
            col_values = pd.Series(np.linspace(self.X[col].min(), self.X[col].max(), self.n_points), name=col)

            # cross with average value of each other column
            other_features = self.X.drop(columns=col)
            other_features = getattr(other_features, agg_func)().to_frame().T
            sub_grid = pd.merge(col_values, other_features, how="cross")

            # predict for every point in the grid
            y_pred = pd.Series(self.model.predict(sub_grid[self.X.columns]), name="ypred")

            # concatenate and add to list
            sub_grid = pd.concat([sub_grid[col].rename("value"), y_pred], axis=1)
            sub_grid = sub_grid.assign(variable=col)
            grid.append(sub_grid)
        grid = pd.concat(grid).reset_index(drop=True)

        # sort
        grid = grid.sort_values(["variable", "value"])
        grid = grid[["variable", "value", "ypred"]]

        return grid

    def make_grid_full_interaction(self) -> pd.DataFrame:
        """Make a grid of all combination of predictor that are individually varried along an even linear spacing
        between their respective min and max values observed in the dataset.

        Examples the entire feature space and captures complex interaction, but computationally intensive for
        large datasets.

        Summary:

            1. Create a vector of evenly spaced values for each predictor between its min and max observed values in
                the dataset
            2. Take the cross product of these vectors to generate all combinations of predictor values
            3. Generate predictions using @self.model
            4. One predictor at a time, group by values of the predictor and calculate the average predicted response

        Return:
            df: DataFrame in long format with ["variable", "value", "ypred", "ypred_std"] columns
                - variable: name of the varied predictor
                - value: value of the predictor
                - "ypred": predicted value of the response variable
                - "ypred_std": standard deviation of the predicted response for each grid point
        """
        # generate equally spaced samples for each predictor
        col_vals = []
        for col in self.X.columns:
            col_vals.append(pd.Series(np.linspace(self.X[col].min(), self.X[col].max(), self.n_points), name=col))

        # sequentially cross predictor series to generate the full combinatorial grid
        grid = col_vals[0]
        for col_val in col_vals[1:]:
            grid = pd.merge(grid, col_val, how="cross")

        # predict response using the model
        y_pred = pd.Series(self.model.predict(grid), name="ypred")

        # combine grid with predictions
        grid = pd.concat([grid, y_pred], axis=1)

        # calculate average predicted value for each set point of each feature
        grid_agg = []
        for col in self.X.columns:
            agg_vals = grid.groupby(col).agg(ypred=("ypred", "mean"), ypred_std=("ypred", "std"))
            agg_vals = agg_vals.rename_axis(index="value").reset_index()
            agg_vals = agg_vals.assign(variable=col)
            grid_agg.append(agg_vals)
        grid_agg = pd.concat(grid_agg).reset_index(drop=True)

        # sort
        grid_agg = grid_agg.sort_values(["variable", "value"])
        grid_agg = grid_agg[["variable", "value", "ypred", "ypred_std"]]

        return grid_agg

    def make_grid_cross_observed(self) -> pd.DataFrame:
        """Make a grid varying each predictor along an even linear spacing for each observed value of the other
        predictors in the dataset.

        Uses actual observations of the predictor, which helps preserve the realistic context of the data.

        Summary:

            For each predictor:
                1. Create a vector of evenly spaced values between its min and max observed values in the dataset.
                2. Take the cross product of this vector and all observations of other predictors in the dataset.
                3. Generate predictions using @self.model.
                4. Calculate the averaged predicted response for each setting of the main predictor.

        Return:
            df: DataFrame in long format with ["variable", "value", "ypred", "ypred_std"] columns
                - variable: name of the varied predictor
                - value: value of the predictor
                - "ypred": predicted value of the response variable
                - "ypred_std": standard deviation of the predicted response for each grid point
        """
        # loop over columns
        grid = []
        for col in self.X.columns:
            # generate equally spaced samples
            col_values = pd.Series(np.linspace(self.X[col].min(), self.X[col].max(), self.n_points), name=col)

            # cross with average value of each other column
            sub_grid = pd.merge(col_values, self.X.drop(columns=col), how="cross")

            # predict for every point in the grid
            y_pred = pd.Series(self.model.predict(sub_grid[self.X.columns]), name="ypred")

            # concatenate
            sub_grid = pd.concat([sub_grid[col].rename("value"), y_pred], axis=1)

            # take average across each grid value
            sub_grid = sub_grid.groupby("value").agg(ypred=("ypred", "mean"), ypred_std=("ypred", "std"))
            sub_grid = sub_grid.reset_index().assign(variable=col)

            # add to grid
            grid.append(sub_grid)
        grid = pd.concat(grid).reset_index(drop=True)

        # sort
        grid = grid.sort_values(["variable", "value"])
        grid = grid[["variable", "value", "ypred", "ypred_std"]]

        return grid

    def make_grid_unq_observed(self) -> pd.DataFrame:
        """Make a grid from combinations of uqnieu values of each predictor and the observed values of other
        predictor. This is very similar to what sklear.inspect.PartialDependenceDisplay does.

        Uses actual observationso of the predictors, but limited to only observed values of predictors in the dataset.

        Summary:

            For each predictor:
                1. Take the vector of unique values of the predictor in observed dataset.
                2. Take the cross product between this vector and all observations of other predictors in the
                   dataset.
                3. Generate predictions using @self.model.
                4. Calculate the averaged predicted response for each setting of the main predictor.

        Return:
            df: DataFrame in long format with ["variable", "value", "ypred", "ypred_std"] columns
                - variable: name of the varied predictor
                - value: value of the predictor
                - "ypred": predicted value of the response variable
                - "ypred_std": standard deviation of the predicted response for each grid point
        """
        # loop over columns
        grid = []
        for col in self.X.columns:
            # generate equally spaced samples
            col_values = self.X[col].drop_duplicates()

            # cross with average value of each other column
            sub_grid = pd.merge(col_values, self.X.drop(columns=col), how="cross")

            # predict for every point in the grid
            y_pred = pd.Series(self.model.predict(sub_grid[self.X.columns]), name="ypred")

            # concatenate
            sub_grid = pd.concat([sub_grid[col].rename("value"), y_pred], axis=1)

            # take average across each grid value
            sub_grid = sub_grid.groupby("value").agg(ypred=("ypred", "mean"), ypred_std=("ypred", "std"))
            sub_grid = sub_grid.reset_index().assign(variable=col)

            # add to grid
            grid.append(sub_grid)
        grid = pd.concat(grid).reset_index(drop=True)

        # sort
        grid = grid.sort_values(["variable", "value"])
        grid = grid[["variable", "value", "ypred", "ypred_std"]]

        return grid

    @staticmethod
    def plot_grid(df: pd.DataFrame, target_name):
        p = pn.ggplot(df, pn.aes("value", target_name))
        p = p + pn.geom_point()
        p = p + pn.geom_line()
        p = p + pn.facet_wrap("variable", scales="free_x", ncol=2)
        p = p + pn.theme(figure_size=(6, 8))
        return p
