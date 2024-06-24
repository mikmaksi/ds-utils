#!python3

import numpy as np
import pandas as pd
import plotnine as pn


class VisGrid:
    def __init__(self, model, X, n_points, target_name: str = "pred"):
        self.model = model
        self.X = X
        self.n_points = n_points
        self.target_name = target_name

    def make_grid_avg(self):
        # loop over columns
        grid = []
        for col in self.X.columns:
            # generate equally spaced samples
            col_values = pd.Series(np.linspace(self.X[col].min(), self.X[col].max(), self.n_points), name=col)

            # cross with average value of each other column
            x_grid = pd.merge(
                col_values,
                pd.DataFrame(self.X.drop(columns=col).mean()).T,
                how="cross",
            )

            # predict for every point in the grid
            y_pred = pd.Series(self.model.predict(x_grid[self.X.columns]), name=f"mean_{self.target_name}")

            # concatenate and add to list
            x_grid = pd.concat([x_grid[col].rename("value"), y_pred], axis=1)
            x_grid = x_grid.assign(variable=col)
            grid.append(x_grid)
        grid = pd.concat(grid).reset_index(drop=True)

        # sort
        grid = grid.sort_values(["variable", f"mean_{self.target_name}"])
        grid = grid[["variable", "value", f"mean_{self.target_name}"]]

        return grid

    def make_grid_full(self):
        # loop over columns
        col_vals = []
        for col in self.X.columns:
            # generate equally spaced samples
            col_vals.append(pd.Series(np.linspace(self.X[col].min(), self.X[col].max(), self.n_points), name=col))

        # sequentially cross series to generate the full grid
        x_grid = col_vals[0]
        for col_val in col_vals[1:]:
            x_grid = pd.merge(x_grid, col_val, how="cross")

        # predict for every point in the grid
        y_pred = pd.Series(self.model.predict(x_grid), name=self.target_name)

        # combine grid with predictions
        grid = pd.concat([x_grid, y_pred], axis=1)

        # loop over columns
        grid_agg = []
        for col in self.X.columns:
            # take average of prediction for each distinct predictor value
            agg_vals = grid.groupby(col).agg(mean=(self.target_name, "mean"), sd=(self.target_name, "std"))
            agg_vals = agg_vals.rename(columns={"mean": f"mean_{self.target_name}", "sd": f"sd_{self.target_name}"})
            agg_vals = agg_vals.rename_axis(index="value").reset_index()
            agg_vals = agg_vals.assign(variable=col)
            grid_agg.append(agg_vals)
        grid_agg = pd.concat(grid_agg).reset_index(drop=True)

        # sort
        grid_agg = grid_agg.sort_values(["variable", f"mean_{self.target_name}"])
        grid_agg = grid_agg[["variable", "value", f"mean_{self.target_name}", f"sd_{self.target_name}"]]

        return grid_agg

    def make_grid_pdp(self):
        # loop over columns
        grid = []
        for col in self.X.columns:
            # generate equally spaced samples
            col_values = pd.Series(np.linspace(self.X[col].min(), self.X[col].max(), self.n_points), name=col)

            # cross with average value of each other column
            x_grid = pd.merge(col_values, self.X.drop(columns=col), how="cross")

            # predict for every point in the grid
            y_pred = pd.Series(self.model.predict(x_grid[self.X.columns]), name=self.target_name)

            # concatenate
            x_grid = pd.concat([x_grid[col].rename("value"), y_pred], axis=1)

            # take average across each grid value
            x_grid = x_grid.groupby("value").agg(mean=(self.target_name, "mean"), sd=(self.target_name, "std"))
            x_grid = x_grid.rename(columns={"mean": f"mean_{self.target_name}", "sd": f"sd_{self.target_name}"})
            x_grid = x_grid.reset_index().assign(variable=col)

            # add to grid
            grid.append(x_grid)
        grid = pd.concat(grid).reset_index(drop=True)

        # sort
        grid = grid.sort_values(["variable", f"mean_{self.target_name}"])
        grid = grid[["variable", "value", f"mean_{self.target_name}", f"sd_{self.target_name}"]]

        return grid

    def make_pdp(self):
        # loop over columns
        grid = []
        for col in self.X.columns:
            # generate equally spaced samples
            col_values = self.X[col].drop_duplicates()

            # cross with average value of each other column
            x_grid = pd.merge(col_values, self.X.drop(columns=col), how="cross")

            # predict for every point in the grid
            y_pred = pd.Series(self.model.predict(x_grid[self.X.columns]), name=self.target_name)

            # concatenate
            x_grid = pd.concat([x_grid[col].rename("value"), y_pred], axis=1)

            # take average across each grid value
            x_grid = x_grid.groupby("value").agg(mean=(self.target_name, "mean"), sd=(self.target_name, "std"))
            x_grid = x_grid.rename(columns={"mean": f"mean_{self.target_name}", "sd": f"sd_{self.target_name}"})
            x_grid = x_grid.reset_index().assign(variable=col)

            # add to grid
            grid.append(x_grid)
        grid = pd.concat(grid).reset_index(drop=True)

        # sort
        grid = grid.sort_values(["variable", f"mean_{self.target_name}"])
        grid = grid[["variable", "value", f"mean_{self.target_name}", f"sd_{self.target_name}"]]

        return grid

    @staticmethod
    def plot_grid(df: pd.DataFrame, target_name):
        p = pn.ggplot(df, pn.aes("value", target_name))
        p = p + pn.geom_point()
        p = p + pn.geom_line()
        p = p + pn.facet_wrap("variable", scales="free_x", ncol=2)
        p = p + pn.theme(figure_size=(6, 8))
        return p
