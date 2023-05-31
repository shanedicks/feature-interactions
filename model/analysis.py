import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import os
import database as db
import output as out
from typing import Any, Dict, List, Set, Tuple, Iterator, Callable, Optional

def role_column(df: pd.DataFrame) -> pd.DataFrame:
    df['role'] = pd.Categorical(df['phenotype'].str.replace('([.a-z])', '', regex=True))

class BasePlot:
    def __init__(
        self,
        db_loc: str,
        db_name: str,
        output_directory: str,
    ) -> None:
        self.db_loc = db_loc
        self.db_name = db_name
        self.db_path = os.path.join(db_loc, db_name)
        self.output_directory = output_directory
        self.plot_dir = self.get_or_create_output_directory()
        self.world_dict = db.get_world_dict(self.db_path)

    def get_or_create_output_directory(self):
        output_path = os.path.join(self.db_loc, self.output_directory)
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        return output_path

    def plot(self):
        ...


class PopulationPlot(BasePlot):
    def __init__(
        self,
        db_loc: str,
        db_name: str,
    ) -> None:
        super().__init__(
            db_loc = db_loc,
            db_name = db_name,
            output_directory = 'pop_plots'
        )

    def plot(self):
        for shadow in [False, True]:
            s = ["", "s_"][shadow]
            for w, n in self.world_dict.items():
                print(f"Trying world {w}")
                ph_df = db.get_phenotypes_df(self.db_path, shadow, worlds=w)
                if not ph_df.empty:
                    m = ph_df.groupby('phenotype')['pop'].sum()
                    phenotypes = tuple(m.nlargest(1000).index)
                    ph_df = db.get_phenotypes_df(self.db_path, shadow, worlds=w, phenotypes=phenotypes)
                    self.save_pivot_fig(ph_df, self.phenotype_pivot, shadow, w, n)
                    role_column(ph_df)
                    self.save_pivot_fig(ph_df, self.role_pivot, shadow, w, n)

    def save_pivot_fig(
        self,
        df: pd.DataFrame,
        pivot: Callable,
        shadow: bool,
        w: int,
        n: int
        ) -> None:
            pop_type = pivot.__name__.split('_')[0].title()
            s = ["", "s_"][shadow]
            prefix = f"{s}{pop_type[0].lower()}"
            title = f"{pop_type} Distribution Over Time\n Network:{n} | World: {w}"
            df.pipe(pivot).pipe(self.pop_plot, title=title)
            plt.tight_layout()
            plt.savefig(f"{self.plot_dir}/{prefix}_n{n}w{w}.png")
            plt.close()

    def phenotype_pivot(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.pivot_table(df, index='step_num', columns="phenotype", values="pop", aggfunc='sum')

    def role_pivot(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.pivot_table(df, index='step_num', columns="role", values="pop", aggfunc='sum')

    def pop_plot(self, df: pd.DataFrame, title: str, legend = False) -> None:
        df.ewm(span=20).mean().plot.area(
            legend=legend,
            title=title,
            xlabel="Step",
            ylabel="Population",
            figsize=(19.2, 9.66)
        )

class ModelVarsPlot(BasePlot):

    def __init__(
        self,
        db_loc: str,
        db_name: str,
    ) -> None:
        super().__init__(
            db_loc = db_loc,
            db_name = db_name,
            output_directory = 'mv_plots'
        )

    def plot(self):
        nd = {
            i: [j for j in self.world_dict if self.world_dict[j] == i]
            for i in self.world_dict.values()
        }
        mv_df = db.get_model_vars_df(self.db_path)
        for column in mv_df.columns[:7]:
            df = mv_df.pivot(index='step_num', columns='world_id', values=column).fillna(0)
            name = column.replace('_', ' ').title()
            for n in nd:
                title = f"{name} Over Time\n Network {n}"
                df[nd[n]].ewm(span=100).mean().plot(
                    title=title,
                    xlabel="Step",
                    ylabel=name,
                    figsize=(19.2, 9.66),
                )
                plt.tight_layout()
                plt.savefig(f"{self.plot_dir}/{column}_{n}.png")
                plt.close()

def activity(df: pd.DataFrame, pivot: Callable) -> pd.DataFrame:
    return df.pipe(pivot).notna().cumsum()

def pop_activity(df: pd.DataFrame, pivot: Callable) -> pd.DataFrame:
    return df.pipe(pivot).cumsum()

def diversity(df: pd.DataFrame, pivot: Callable) -> pd.DataFrame:
    return df.pipe(activity, pivot=pivot).apply(lambda x: x > 0).sum(axis = 1)

def gen_activity_plots(
    df: pd.DataFrame,
    wd: Dict[int, int],
    activity: Callable,
    pivot: Callable,
    save: bool = False,
    suffix: str = '',
    dest: str = '',
    id_list: list = None
    ) -> None:
    if id_list is None:
        id_list = [i for i in df['world_id'].unique()]
    for i in id_list:
        n_id = wd[i]
        title = f"Network: {n_id} | World: {i}"
        activity(df=df, pivot=pivot).plot(legend=False, title=title, xlabel="Step", ylabel="Activity")
        if save:
            plt.savefig(f"{dest}activity_n{n_id}w{i}{suffix}.png")
            plt.close()
        else:
            plt.show()

def get_CAD(
    df: pd.DataFrame,
    activity: Callable,
    pivot: Callable,
    ) -> pd.DataFrame:
    return df.pipe(activity, pivot=pivot).apply(pd.Series.value_counts, axis = 1)

def CAD_plot(
    df: pd.DataFrame,
    s_df: pd.DataFrame,
    activity: Callable,
    pivot: Callable,
    title: str,
    ) -> None:
    total = pivot(df).shape[1] + pivot(s_df).shape[1]
    CAD = get_CAD(df=df, activity=activity, pivot=pivot).sum().div(total).plot(loglog=True, title=title)
    sCAD = get_CAD(df=s_df, activity=activity, pivot=pivot).sum().div(total).plot(loglog=True, title=title)

def gen_CAD_plots(
    db_loc: str,
    db_name: str,
    resolution: Optional[int] = None,
    ) -> None:
    if id_list is None:
        id_list = [i for i in df['world_id'].unique()]
    pop_type = pivot.__name__.split('_')[0].title()
    for i in id_list:
        n_id = wd[i]
        title = f"Network: {n_id} | World: {i}"
        CAD_plot(
            df=df.loc[df["world_id"]==i],
            s_df=s_df.loc[s_df["world_id"]==i],
            activity=activity,
            pivot=pivot,
            title=title
        )
        if save:
            plt.savefig(f"{dest}/CAD_n{n_id}w{i}{suffix}.png")
            plt.close()
        else:
            plt.show()
