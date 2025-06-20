import copy
import os
import json
import logging
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
import networkx as nx
import pandas as pd
import seaborn as sns
import model.database as db
import model.output as out
from model.rolesets import FeatureUtilityCalculator, RoleAnalyzer
from datetime import datetime
from multiprocessing import Pool
from typing import Any, Dict, List, Set, Tuple, Iterator, Callable, Optional, Union
from tqdm import tqdm

class BasePlot:
    """
    Base class for creating visualization plots from simulation data.

    Handles database connection and output directory management that's common
    across all plot types.
    """
    def __init__(
        self,
        db_loc: str,
        db_name: str,
        output_directory: str,
    ) -> None:
        """
        Initialize the plot generator with database and output settings.

        Args:
            db_loc: Directory containing the database file
            db_name: Name of the database file
            output_directory: Directory where plots will be saved
        """
        self.db_loc = db_loc
        self.db_name = db_name
        self.db_path = os.path.join(db_loc, db_name)
        self.output_directory = output_directory
        self.plot_dir = self.get_or_create_output_directory()
        self.world_dict = db.get_world_dict(self.db_path)
        self.params = db.get_params_df(self.db_path)

    def get_or_create_output_directory(self):
        output_path = os.path.join(self.db_loc, self.output_directory)
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        return output_path


class PopulationPlot(BasePlot):
    """
    Generates plots showing population distributions over time for different
    phenotypes and roles in the simulation.
    """

    def plot(self, plot_type = 'area', num_types = 1000, world=None):
        """
        Create population distribution plots for phenotypes and roles.

        Args:
            plot_type: Type of plot to generate ('area' or 'line')
            num_types: Maximum number of phenotypes to include in the plot
            world: Specific world ID to plot, or None for all worlds
        """
        worlds_to_plot = [(world, self.world_dict[world])] if world is not None else self.world_dict.items()
        # Process both actual (shadow=False) and shadow model (neutral baseline) data
        for shadow in [False, True]:
            s = ["", "s_"][shadow]
            for w, n in worlds_to_plot:
                logging.info(f"Trying world {w}")
                ph_df = db.get_phenotypes_df(self.db_path, shadow, worlds=w)
                if not ph_df.empty:
                    self.save_pivot_fig(ph_df, self.role_pivot, plot_type, shadow, w, n)
                    m = ph_df.groupby('phenotype')['pop'].sum()
                    phenotypes = tuple(m.nlargest(num_types).index)
                    # Convert single item tuple to scalar for compatibility with get_phenotypes_df API
                    if len(phenotypes) == 1:
                        phenotypes = phenotypes[0]
                    ph_df = db.get_phenotypes_df(
                        self.db_path, shadow,
                        worlds=w,
                        phenotypes=phenotypes
                    )
                    self.save_pivot_fig(ph_df, self.phenotype_pivot, plot_type, shadow, w, n)

    def save_pivot_fig(
        self,
        df: pd.DataFrame,
        pivot: Callable,
        plot_type: str,
        shadow: bool,
        w: int,
        n: int
        ) -> None:
            pop_type = pivot.__name__.split('_')[0].title()
            s = ["", "s_"][shadow]
            prefix = f"{s}{pop_type[0].lower()}"
            title = f"{pop_type} Distribution Over Time\n Network:{n} | World: {w}"
            if plot_type == 'line':
                df.pipe(pivot).pipe(self.pop_plot, title=title)
            else:
                df.pipe(pivot).pipe(self.area_pop_plot, title=title)
            plt.tight_layout()
            plt.savefig(f"{self.plot_dir}/{prefix}_n{n}w{w}.png", dpi=300)
            plt.close()

    def phenotype_pivot(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.pivot_table(
            df,
            index='step_num',
            columns="phenotype",
            values="pop",
            aggfunc='sum'
        )

    def role_pivot(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.pivot_table(
            df,
            index='step_num',
            columns="role",
            values="pop",
            aggfunc='sum'
        )

    def area_pop_plot(self, df: pd.DataFrame, title: str, legend = False) -> None:
        plt.figure(figsize=(6.5, 4.875))
        ax = df.ewm(span=20).mean().plot.area(legend=legend)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Step", fontsize=10)
        ax.set_ylabel("Population", fontsize=10)
        ax.tick_params(axis='both', labelsize=10)
        plt.tight_layout()

    def pop_plot(self, df: pd.DataFrame, title: str, legend = False) -> None:
        plt.figure(figsize=(6.5, 4.875))
        ax = df.ewm(span=20).mean().plot(legend=legend)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Step", fontsize=10)
        ax.set_ylabel("Population", fontsize=10)
        ax.tick_params(axis='both', labelsize=10)

class HeatmapPlot(BasePlot):
    """
    Generates heatmap visualizations of role population distributions over time.
    These heatmaps show the changing prominence of different roles throughout
    the simulation.
    """

    def plot(self, num_roles=30, num_bins=20, world=None):
        """
        Create heatmap visualizations for role populations across time.

        Args:
            num_roles: Number of top roles to include in the heatmap
            num_bins: Number of time bins to use for the x-axis
            world: Specific world ID to plot, or None for all worlds
        """
        worlds_to_plot = [(world, self.world_dict[world])] if world is not None else self.world_dict.items()
        for shadow in [False, True]:
            s = ["", "s_"][shadow]
            for w, n in worlds_to_plot:
                logging.info(f"Trying world {w}")
                ph_df = db.get_phenotypes_df(self.db_path, shadow, worlds=w)
                if not ph_df.empty:
                    self.save_fig(ph_df, num_roles, num_bins, shadow, w, n)

    def save_fig(
        self,
        df: pd.DataFrame,
        num_roles: int,
        num_bins: int,
        shadow: bool,
        w: int,
        n: int
    ) -> None:
        s = ["", "s_"][shadow]
        prefix = f"{s}role_heatmap"
        title = f"Population Heatmap for Top Roles (Log10 Scale)\n Network:{n} | World: {w}"
        pivot_data = self.create_pivot(df, num_roles, num_bins)
        self.plot_heatmap(pivot_data, title)
        plt.savefig(f"{self.plot_dir}/{prefix}_n{n}w{w}.png", dpi=300)
        plt.close()

    def create_pivot(self, df: pd.DataFrame, num_roles: int, num_bins: int) -> pd.DataFrame:
        top_roles = df.groupby('role')['pop'].sum().nlargest(num_roles).index
        df['step_bin'] = pd.cut(df['step_num'], bins=num_bins)
        pivot_data = df.pivot_table(index='role', columns='step_bin', values='pop', aggfunc='sum')
        pivot_data = np.log10(pivot_data + 1)
        pivot_data = pivot_data.reindex(index=top_roles)
        return pivot_data

    def plot_heatmap_old(self, df: pd.DataFrame, title: str) -> None:
        plt.figure(figsize=(6.5, 4.875))
        sns.heatmap(df, cmap='viridis', cbar_kws={'label': 'Log10(Population)'}, yticklabels=True)
        plt.title(title)
        plt.xlabel("Step Range")
        plt.ylabel("Role")
        plt.yticks(rotation=0, ha='right')

        plt.xticks([])

        plt.tight_layout()

    def plot_heatmap(self, df: pd.DataFrame, title: str) -> None:
        fig_width = 6.5
        fig_height_per_role = 0.2
        num_roles = len(df.index)
        fig_height = num_roles * fig_height_per_role

        plt.figure(figsize=(fig_width, fig_height))

        sns.heatmap(df, cmap='viridis', cbar_kws={'label': 'Log10(Population)'}, yticklabels=True)

        plt.title(title)
        plt.xlabel("Step Range")
        plt.ylabel("Role")
        plt.xticks([])
        plt.yticks(rotation=0, ha='right')
        plt.title(title, fontsize=12)
        plt.xlabel("Step Range", fontsize=10)
        plt.ylabel("Role", fontsize=10)
        plt.yticks(fontsize=8)
        plt.tight_layout()


class ModelVarsPlot(BasePlot):
    """
    Generates plots for various model variables over time, such as population,
    utility measures, number of phenotypes/roles, and interaction counts.
    """
    # Mapping between database column names and human-readable plot labels
    DEFAULT_PLOT_COLUMNS = {
        "pop": "Population",
        "total_utility": "Agent Total Utility",
        "mean_utility": "Agent Mean Utility",
        "med_utility": "Agent Median Utility",
        "num_types": "Number of Phenotypes",
        "num_roles": "Number of Roles",
        "num_features": "Number of Features",
        "agent_int": "Agent/Agent Interactions",
        "env_int": "Agent/Environment Interactions",
    }

    def plot(self, network: List[int] = None, plot_columns: List[Tuple[str, str]] = None):
        """
        Create plots for selected model variables in specific networks.

        Args:
            network: List of network IDs to plot, or None for all networks
            plot_columns: List of column names to plot, or None for default columns
        """
        # Group worlds by their network ID for visualization
        nd = {
            i: [j for j in self.world_dict if self.world_dict[j] == i]
            for i in self.world_dict.values()
        }

        if network is not None:
            nd = {k: v for k, v in nd.items() if k in network}
        mv_df = db.get_model_vars_df(self.db_path)

        plot_columns = columns if columns is not None else list(self.DEFAULT_PLOT_COLUMNS.keys())

        for column in plot_columns:
            try:
                df = mv_df.pivot(index='step_num', columns='world_id', values=column).fillna(0)
            except KeyError as e:
                logging.error(e)
                continue
            for n in nd:
                try:
                    name = self.DEFAULT_PLOT_COLUMNS[column]
                    title = f"{name} Over Time\n Network {n}"
                    fig, ax = plt.subplots(figsize=(6.5, 3.656))  # Creates both figure and axes with the specified size
                    df[nd[n]].ewm(span=100).mean().plot(ax=ax)  # Use the created axes for plotting

                    ax.set_title(title, fontsize=12)
                    ax.set_xlabel("Step", fontsize=10)
                    ax.set_ylabel(name, fontsize=10)
                    ax.tick_params(axis='both', which='major', labelsize=10)

                    plt.tight_layout()
                    plt.savefig(f"{self.plot_dir}/{column}_{n}.png", dpi=300)
                    plt.close(fig)
                except KeyError as e:
                    logging.error(e)
                    continue

    def plot_all_networks(self, plot_columns: List[Tuple[str, str]] = None):
        """
        Create combined plots showing all networks for each selected model variable.

        Args:
            plot_columns: List of column names to plot, or None for default columns
        """
        mv_df = db.get_model_vars_df(self.db_path)

        plot_columns = plot_columns if plot_columns is not None else list(self.DEFAULT_PLOT_COLUMNS.keys())

        for column in plot_columns:
            try:
                df = mv_df.pivot(index='step_num', columns='world_id', values=column).fillna(0)
            except KeyError as e:
                logging.error(e)
                continue

            fig, ax = plt.subplots(figsize=(6.5, 3.656))

            network_ids = set(self.world_dict.values())
            cmap = plt.cm.tab20
            network_colors = {nid: cmap(i % 20) for i, nid in enumerate(sorted(network_ids))}

            legend_elements = []
            for network_id, color in sorted(network_colors.items()):
                legend_elements.append(plt.Line2D([0], [0], color=color, label=f"N{network_id}"))

            for world_id, network_id in self.world_dict.items():
                if world_id in df.columns:
                    world_color = network_colors[network_id]

                    df[world_id].ewm(span=100).mean().plot(
                        ax=ax,
                        color=world_color,
                        label=f"_nolegend_"
                    )

            name = self.DEFAULT_PLOT_COLUMNS.get(column, column)
            title = f"{name} Over Time (All Networks)"
            ax.set_title(title, fontsize=12)
            ax.set_xlabel("Step", fontsize=10)
            ax.set_ylabel(name, fontsize=10)
            ax.tick_params(axis='both', which='major', labelsize=10)
            ax.legend(
                loc='center left',
                bbox_to_anchor=(1, 0.5),
                handles=legend_elements,
                fontsize=10
            )
            plt.subplots_adjust(right=0.85)
            plt.savefig(f"{self.plot_dir}/{column}_all_networks.png", dpi=300, bbox_inches='tight')
            plt.close(fig)


class PayoffPlot(BasePlot):
    """
    Generates visualizations of payoff distributions from agent interactions.
    Shows relationships between initiator and target payoffs across different
    interaction types.
    """

    def __init__(
        self,
        db_loc: str,
        db_name: str,
        output_directory: str,
    ) -> None:
        super().__init__(db_loc, db_name, output_directory)
        self.interactions_df = db.get_interactions_df(self.db_path)

    def plot(self, plot_type, network_ids=None):
        """
        Create payoff distribution plots for specified networks.

        Args:
            plot_type: Type of visualization ('scatter' or 'box_and_whisker')
            network_ids: Specific network IDs to plot, or None for all networks
        """
        # If network_ids is not provided, plot for all unique network IDs
        if network_ids is None:
            network_ids = self.interactions_df['network_id'].unique()
        # If a single network_id is provided, wrap it in a list
        elif isinstance(network_ids, int):
            network_ids = [network_ids]

        # Loop through the provided network IDs
        for network_id in network_ids:
            payoffs_df = db.get_payoffs_df(self.db_path, network_id)
            # Call the plotting function based on plot_type
            if plot_type == 'scatter':
                self.scatter(payoffs_df, self.interactions_df)
            elif plot_type == 'box_and_whisker':
                self.box_and_whisker(payoffs_df, self.interactions_df)
            else:
                raise ValueError(f"Plot type '{plot_type}' is not recognized. Use 'scatter' or 'box_and_whisker'.")


    def scatter(self, payoffs_df, interactions_df):
        fig, ax = plt.subplots(figsize=(6, 4.875))
        for interaction_id in payoffs_df['interaction_id'].unique():
            subset = payoffs_df[payoffs_df['interaction_id'] == interaction_id]
            anchor_row = interactions_df[interactions_df['interaction_id'] == interaction_id].iloc[0]
            initiator, target = anchor_row['initiator'], anchor_row['target']
            label = f"{initiator} -> {target}"

            ax.scatter(subset['initiator_payoff'], subset['target_payoff'], label=label, s=2)

        ax.axhline(0, color='grey', linestyle='--')
        ax.axvline(0, color='grey', linestyle='--')
        ax.set_title('Initiator vs. Target Payoffs by Interaction')
        ax.set_xlabel('Initiator Payoff')
        ax.set_ylabel('Target Payoff')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.grid(True)
        plt.tight_layout()
        plt.show()

    def box_and_whisker(self, payoffs_df, interactions_df):
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for interaction_id in payoffs_df['interaction_id'].unique():
            payoff_subset = payoffs_df[payoffs_df['interaction_id'] == interaction_id]

            median_initiator = np.median(payoff_subset['initiator_payoff'])
            median_target = np.median(payoff_subset['target_payoff'])

            ax.boxplot(payoff_subset['initiator_payoff'], vert=False, positions=[median_target], widths=0.03, patch_artist=True, boxprops=dict(facecolor='skyblue', alpha=0.5),  showfliers=False)
            ax.boxplot(payoff_subset['target_payoff'], vert=True, positions=[median_initiator], widths=0.02, patch_artist=True, boxprops=dict(facecolor='lightgreen', alpha=0.5), showfliers=False)

        ax.axhline(0, color='grey', linestyle='--')
        ax.axvline(0, color='grey', linestyle='--')
        ticks = np.linspace(-1, 1, 21)
        tick_labels = ['{:.1f}'.format(tick) for tick in ticks]
        ax.set_title('Initiator vs. Target Payoffs by Interaction')
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(tick_labels)
        ax.set_yticklabels(tick_labels)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_xlabel('Initiator Payoffs')
        ax.set_ylabel('Target Payoffs')
        plt.tight_layout()
        plt.show()

class NetworkPlot(BasePlot):
    """
    Generates network visualizations showing relationships between features and roles.
    Provides both static network plots and time-series visualizations of evolving networks.
    """

    def __init__(
        self,
        db_loc: str,
        db_name: str,
        output_directory: str,
    ) -> None:
        super().__init__(db_loc, db_name, output_directory)
        self.interactions_df = db.get_interactions_df(self.db_path)
        self.feature_changes_df = db.get_feature_changes_df(self.db_path)
        self.sites_dict = db.get_sites_dict(self.db_path)

    def get_evenly_spaced_steps(self, num_plots: int) -> List[int]:
        all_steps = db.get_steps_list(self.db_path)
        if num_plots >= len(all_steps):
            return all_steps
        interval = max(len(all_steps) // num_plots, 1)
        selected_steps = all_steps[::interval]
        selected_steps.append(all_steps[-1])
        return selected_steps

    def set_node_postions(
        self,
        G: nx.DiGraph,
        env_nodes: List[str],
        other_nodes: List[str]
    ) -> None:
        G_augmented = G.copy()
        for node in other_nodes:
            for env_node in env_nodes:
                if not G_augmented.has_edge(node, env_node):
                    G_augmented.add_edge(node, env_node, weight=0.05)
        pos = {node: (3 * i, 0) for i, node in enumerate(sorted(env_nodes))}
        kwargs = {
            "pos": {**pos, **{node: (0.5 * i, 2) for i, node in enumerate(sorted(other_nodes))}},
            "k": 3,
            "seed": 21,
            "iterations": 100
        }
        if env_nodes:
            kwargs["fixed"] = env_nodes
        if other_nodes:
            new_positions = nx.spring_layout(G_augmented, **kwargs)
            min_y = min(new_positions[node][1] for node in other_nodes)
            y_offset = max(2 - min_y, 2)
            for node in other_nodes:
                x, y = new_positions[node]
                pos[node] = (x, y + y_offset)
        for node, (x, y) in pos.items():
            pos[node] = (float(x), float(y))
        nx.set_node_attributes(G, pos, "pos")

    def construct_features_network(
        self,
        features: pd.DataFrame,
        interactions: pd.DataFrame,
    ) -> nx.DiGraph:
        G = nx.DiGraph()
        change_steps = list(map(int, features['step_num'].unique()))
        G.graph['change_steps'] = change_steps
        env_features = list(features[features['env']==1]['name'].unique())
        agent_features = list(features[features['env']==0]['name'].unique())
        G.add_nodes_from(
            env_features,
            type='env',
            node_color='lightgreen',
            node_shape='s',
        )
        G.add_nodes_from(
            agent_features,
            type='agent',
            node_color='skyblue',
        )
        added_dict = features[features['change'] == 'added'].set_index('name')['step_num'].to_dict()
        removed_dict = features[features['change'] == 'removed'].set_index('name')['step_num'].to_dict()
        for node in G.nodes:
            G.nodes[node]['added'] = added_dict[node]
            if node in removed_dict:
                G.nodes[node]['removed'] = removed_dict[node]
        existing_nodes = set(G.nodes())
        filtered_interactions = interactions[
            (interactions['initiator'].isin(existing_nodes)) & 
            (interactions['target'].isin(existing_nodes))
        ].copy()
        filtered_interactions['edge_attr'] = [
            {'i_anchor': float(i_anchor), 't_anchor': float(t_anchor)}
            for i_anchor, t_anchor in zip(
                filtered_interactions['i_anchor'],
                filtered_interactions['t_anchor']
            )
        ]
        edge_attr_data = list(zip(
            filtered_interactions['initiator'],
            filtered_interactions['target'],
            filtered_interactions['edge_attr']
        ))
        G.add_edges_from(edge_attr_data)
        self.set_node_postions(G, env_features, agent_features)
        return G

    def export_features_networks_to_json(self) -> None:
        """
        Export feature network data to JSON format for all worlds and networks.
        Creates a single JSON file containing all network structures.
        """
        fc_df = self.feature_changes_df
        i_df = self.interactions_df

        networks_data = {}
        for wid, nid in self.world_dict.items():
            if nid not in networks_data:
                networks_data[nid] = {}
            networks_data[nid][wid] = {}

        for nid, worlds in networks_data.items():
            logging.info(f"Beginning network {nid}")
            interactions_subset = i_df[i_df['network_id'] == nid]
            for wid in worlds:
                features_subset = fc_df[fc_df['world_id'] == wid]
                logging.info(f"Constructing features network for world {wid}")
                features_network = self.construct_features_network(
                    features_subset,
                    interactions_subset
                )
                logging.info("Preparing json data")
                validate_and_convert_floats(features_network)
                network_json = nx.node_link_data(features_network)
                networks_data[nid][wid] = network_json
        logging.info("Preparing to write json file")
        output_file_path = os.path.join(self.db_loc, "features_networks_data.json")
        with open(output_file_path, 'w') as json_file:
            json.dump(networks_data, json_file, indent=4)

    def draw_feature_interactions(
        self,
        G: nx.DiGraph,
        title: str,
        file_path: str,
        ) -> None:
        pos = {node: data['pos'] for node, data in G.nodes(data=True)}
        labels = {n: n for n in pos.keys()}
        plt.figure(figsize=(6.5, 4.875)) 
        for node, data in G.nodes(data=True):
            nx.draw_networkx_nodes(
                G,pos,nodelist=[node],
                node_color=data['node_color'],
                node_shape=data.get('node_shape', 'o'),
                node_size=500
            )
        nx.draw_networkx_edges(G, pos, node_size=500)
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        plt.title(title)
        agent_patch = mpatches.Patch(color='skyblue', label='Agent Features')
        env_patch = mpatches.Patch(color='lightgreen', label='Environment Features')
        plt.legend(handles=[agent_patch, env_patch])
        plt.savefig(file_path, dpi=300)
        plt.close()

    def plot_features_networks(
        self,
        network_ids: Optional[List[int]] = None,
        world_ids: Optional[List[int]] = None,
        steps: Optional[List[int]] = None,
        num_plots: Optional[int] = None
        ) -> None:
        """
        Generate visualizations of feature networks for specified simulation steps.

        Args:
            network_ids: List of specific network IDs to plot, or None for all
            world_ids: List of specific world IDs to plot, or None for all
            steps: Specific steps to plot, or None for automatic selection
            num_plots: Number of evenly-spaced timesteps to plot, or None for all
        """
        
        json_file_path = os.path.join(self.db_loc, "features_networks_data.json")
        if not os.path.isfile(json_file_path):
            raise FileNotFoundError(f"File not found at {json_file_path}")
        features_networks_dir = os.path.join(self.plot_dir, 'features_networks')
        os.makedirs(features_networks_dir, exist_ok=True)

        with open(json_file_path, 'r') as file:
            networks_data = json.load(file)

        for nid, worlds in networks_data.items():
            if network_ids is not None and nid not in map(str, network_ids):
                continue
            logging.info(f"Beginning network {nid}")
            for wid, world_data in worlds.items():
                if world_ids is not None and wid not in map(str, world_ids):
                    continue
                logging.info(f"Generating plots for world {wid}")
                G = nx.node_link_graph(world_data)
                if steps:
                    plot_steps = steps
                elif num_plots:
                    plot_steps = self.get_evenly_spaced_steps(num_plots)
                else:
                    plot_steps = G.graph.get('change_steps', [])
                for step in plot_steps:
                    logging.info(f"Plotting world {wid} step {step}")
                    nodes_to_include = [
                        n for n, attr
                        in G.nodes(data=True)
                        if attr['added'] <= step
                        and attr.get('removed', float('inf')) > step
                    ]
                    subG = G.subgraph(nodes_to_include)
                    file_path = os.path.join(features_networks_dir, f'n{nid}_w{wid}_step{step}.png')
                    title = f"Features Network\nN:{nid} W:{wid} Step:{step}"
                    self.draw_feature_interactions(subG, title, file_path)
                    plt.close()

    def construct_role_network(
        self,
        env_nodes: List[str],
        phenotypes_dataframe: pd.DataFrame,
        interactions_dataframe: pd.DataFrame
    ) -> nx.DiGraph:
        ph_df, i_df = phenotypes_dataframe, interactions_dataframe
        G = nx.DiGraph()
        G.add_nodes_from(env_nodes, type='env')
        role_stats = ph_df.groupby('role').agg({
            'step_num': ['min', 'max'],
            'pop': ['min', 'mean', 'max']
        }).reset_index()
        role_stats.columns = ['role', 'min_step', 'max_step', 'min_pop', 'mean_pop', 'max_pop']
        min_steps = [int(x) for x in role_stats['min_step'].unique()]
        max_steps = [int(x) for x in role_stats['max_step'].unique()]
        change_steps = sorted(set(min_steps) | set(max_steps))
        G.graph['change_steps'] = change_steps
        for index, row in role_stats.iterrows():
            G.add_node(row['role'], 
                       type='role', 
                       min_step=row['min_step'], 
                       max_step=row['max_step'], 
                       min_pop=row['min_pop'], 
                       mean_pop=row['mean_pop'], 
                       max_pop=row['max_pop'])

        for role1 in role_stats['role']:
            role1_components = set(role1.split(":"))
            initiator_matches = i_df[i_df['initiator'].isin(role1_components)]
            targets = set(initiator_matches['target'])
            for env_node in env_nodes:
                if env_node in targets:
                    num_interactions = len(initiator_matches[initiator_matches['target'] == env_node])
                    G.add_edge(role1, env_node, weight=num_interactions)
            for role2 in role_stats['role']:
                role2_components = set(role2.split(":"))
                interaction_targets = targets.intersection(role2_components)
                num_interactions = len(initiator_matches[initiator_matches['target'].isin(interaction_targets)])
                if num_interactions > 0:
                    G.add_edge(role1, role2, weight=num_interactions)
        self.set_node_postions(G, env_nodes, list(role_stats['role']))
        return G

    def export_role_networks_to_json(self) -> None:
        """
        Export role network data to JSON format for all worlds and networks.
        Creates one JSON file per network containing role network structures.
        """
        i_df = self.interactions_df
        all_sites = set(site for world_sites in self.sites_dict.values() for site in world_sites)

        for nid in set(self.world_dict.values()):
            networks_data = {}
            logging.info(f"Beginning network {nid}")
            for wid in [w for w, n in self.world_dict.items() if n == nid]:
                logging.info(f"Beginning world {wid}")
                networks_data[wid] = {site: {} for site in all_sites}
                for site in all_sites:
                    logging.info(f"Constructing role network for world {wid} {site}")
                    env_nodes = []
                    if site in self.sites_dict.get(wid, {}):
                        features = self.sites_dict[wid][site]
                        env_nodes = list({feature.split('.')[0] for feature in features})
                    logging.info("Retrieving phenotypes dataframe")
                    ph_df = db.get_phenotypes_df(
                        self.db_path,
                        shadow=False,
                        worlds=wid,
                        sites=site
                    )
                    interactions_subset = i_df[i_df['network_id'] == nid]
                    role_network = self.construct_role_network(
                        env_nodes,
                        ph_df,
                        interactions_subset
                    )
                    logging.info("Preparing json data")
                    validate_and_convert_floats(role_network)
                    network_json = nx.node_link_data(role_network)
                    networks_data[wid][site] = network_json
            logging.info("Preparing to write json file")
            output_file_path = os.path.join(self.db_loc, f"role_networks_data_nid_{nid}.json")
            with open(output_file_path, 'w') as json_file:
                json.dump(networks_data, json_file, indent=4)
            logging.info(f"Networks JSON for nid {nid} written to {output_file_path}")

    def update_role_network(
        self,
        G: nx.DiGraph,
        phenotypes_df: pd.DataFrame
        ) -> nx.DiGraph:
        G = copy.deepcopy(G)
        env_nodes = [node for node, data in G.nodes(data=True) if data['type'] == 'env']
        role_nodes = [role for role in phenotypes_df['role'].unique()]
        G = G.subgraph(env_nodes + role_nodes)
        pops = phenotypes_df.groupby('role').sum()['pop']
        for role in role_nodes:
            G.nodes[role]['pop'] = pops[role]
        return G

    def draw_role_network(
        self,
        G: nx.DiGraph,
        title: str,
        file_path: str
        ) -> None:
        env_nodes = [node for node, data in G.nodes(data=True) if data['type'] == 'env']
        role_nodes = [node for node, data in G.nodes(data=True) if data['type'] == 'role']
        for env in env_nodes:
            G.nodes[env]['color'] = "lightgreen"
        cmap = plt.get_cmap('Blues')
        pops = {role: G.nodes[role]['pop'] for role in role_nodes}
        normalize = plt.Normalize(0, max(pops.values()))
        for role, pop in pops.items():
            G.nodes[role]['color'] = cmap(normalize(pop))
        node_colors = [data['color'] for node, data in G.nodes(data=True)]
        edge_weights = nx.get_edge_attributes(G, 'weight')
        edge_widths = [.2 * edge_weights[edge] for edge in G.edges()]
        pos = {node: data['pos'] for node, data in G.nodes(data=True)}
        plt.figure(figsize=(10, 8))
        for node, data in G.nodes(data=True):
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=[node],
                node_color=[data['color']],
                node_shape=data.get('node_shape', 'o'),
                node_size=1000
            )
            text_color = 'black' if is_light_color(data['color']) else 'white'
            nx.draw_networkx_labels(
                G, pos,
                labels={node: node},
                font_color=text_color,
                font_size=8
            )
        nx.draw_networkx_edges(G, pos, width=edge_widths, node_size=1000)
        sm_pop = plt.cm.ScalarMappable(cmap=cmap, norm=normalize)
        sm_pop.set_array([])
        plt.title(title)
        plt.tight_layout()
        plt.colorbar(sm_pop, label='Population')
        plt.savefig(file_path, dpi=300)
        plt.close()

    def plot_role_networks(
        self,
        network_ids: Optional[List[int]] = None,
        world_ids: Optional[List[int]] = None,
        steps: Optional[List[int]] = None,
        num_plots: Optional[int] = None
        ) -> None:
        """
        Generate visualizations of role networks for specified simulation steps.

        Args:
            network_ids: List of specific network IDs to plot, or None for all
            world_ids: List of specific world IDs to plot, or None for all
            steps: Specific steps to plot, or None for automatic selection
            num_plots: Number of evenly-spaced timesteps to plot, or None for all
        """
        role_networks_dir = os.path.join(self.plot_dir, 'role_networks')
        os.makedirs(role_networks_dir, exist_ok=True)

        if network_ids is None:
            network_ids = set(self.world_dict.values())

        for nid in network_ids:
            json_file_path = os.path.join(self.db_loc, f"role_networks_data_nid_{nid}.json")
            if not os.path.isfile(json_file_path):
                logging.warning(f"File not found for network ID {nid} at {json_file_path}")
                continue

            with open(json_file_path, 'r') as file:
                networks_data = json.load(file)
            for wid, sites in networks_data.items():
                if world_ids is not None and int(wid) not in world_ids:
                    continue
                logging.info(f"Beginning network {nid}, world {wid}")
                for site_id, site_data in sites.items():
                    logging.info(f"Beginning site {site_id}")
                    G = nx.node_link_graph(site_data)
                    if steps:
                        plot_steps = steps
                    elif num_plots:
                        plot_steps = self.get_evenly_spaced_steps(num_plots)
                    else:
                        plot_steps = G.graph.get('change_steps', []) 
                    for step in plot_steps:
                        logging.info(f"Updating role network for world {wid}{site_id} at step {step}")
                        ph_df = db.get_phenotypes_df(
                            self.db_path,
                            shadow=False,
                            worlds=wid,
                            sites=site_id,
                            steps=step
                        )
                        if ph_df.empty:
                            logging.info(f"Empty Phenotypes Dataframe, continuing to next site")
                            break
                        updated_G = self.update_role_network(G, ph_df)
                        logging.info(f"Plotting world {wid}{site_id} step {step}")
                        site_id_formatted = site_id.replace(", ", "_").replace("(", "").replace(")", "")
                        file_path = os.path.join(
                            role_networks_dir,
                            f'n{nid}_w{wid}_{site_id_formatted}_step{step}.png'
                        )
                        title = f"Role Network\nN:{nid} W:{wid} {site_id} Step:{step}"
                        self.draw_role_network(updated_G, title, file_path)
                        plt.close()


def is_light_color(color):
    rgb = mcolors.to_rgb(color)
    luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    return luminance > 0.5

def shannon_entropy(proportions: List[float]):
    """
    Calculate Shannon entropy for a list of proportions.

    Measures diversity/uncertainty in a distribution. Higher values indicate
    more even distributions across different categories.

    Args:
        proportions: List of proportions that sum to 1.0

    Returns:
        Entropy value (in bits)
    """
    proportions = np.array(proportions)
    nonzero_proportions = proportions[proportions != 0]
    if len(nonzero_proportions) == 0:
        return 0.0
    entropy = -np.sum(nonzero_proportions * np.log2(nonzero_proportions))
    return entropy

def add_evolutionary_activity_columns(df, snap_interval=None):
    """
    Add evolutionary activity metrics to a phenotypes dataframe.

    Calculates various activity metrics including persistence, cumulative population,
    growth rates, and selection differentials at both phenotype and role levels.

    Args:
        df: Phenotypes dataframe with world_id, step_num, phenotype, role, and pop columns
        snap_interval: Steps between shadow model resets; used to zero out selection
                      metrics at reset points

    Returns:
        DataFrame with added evolutionary activity columns
    """
    # Convert float64 columns to float32 to reduce memory usage
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')

    # Compute total population per (world_id, step_num)
    step_totals = df.groupby(['world_id', 'step_num'])['pop'].sum()
    # Get previous step's total population per world (used for proportion-based metrics)
    prev_step_totals = step_totals.groupby(level=0).shift(1)

    # PHENOTYPE-LEVEL CALCULATIONS (these work correctly at row level)
    # ---------------------------------------------------------------------
    df['total_pop'] = df.set_index(['world_id', 'step_num']).index.map(step_totals)
    df['prev_total_pop'] = df.set_index(['world_id', 'step_num']).index.map(prev_step_totals)
    # Calculate growth rates based on previous step's population
    df['phenotype_prev_pop'] = df.groupby(['world_id', 'phenotype'])['pop'].shift(1)
    df['phenotype_growth_rate'] = np.where(df['phenotype_prev_pop'] > 0,
                                         (df['pop'] - df['phenotype_prev_pop']) / df['phenotype_prev_pop'],
                                         0)
    # Calculate non-neutral activity (delta N) for phenotypes
    df['phenotype_expected_prop'] = np.where(df['prev_total_pop'] > 0,
                                        df['phenotype_prev_pop'] / df['prev_total_pop'],
                                        0)
    df['phenotype_observed_prop'] = df['pop'] / df['total_pop']
    df['phenotype_delta_N'] = np.where(df['phenotype_observed_prop'] > df['phenotype_expected_prop'],
                                      df['total_pop'] *
                                      (df['phenotype_observed_prop'] - df['phenotype_expected_prop'])**2,
                                      0)
    df = df.drop(['phenotype_expected_prop', 'phenotype_observed_prop', 'phenotype_prev_pop', 'total_pop', 'prev_total_pop'], axis=1)
    # Track how long each phenotype has persisted
    df['phenotype_activity'] = df.groupby(['world_id', 'phenotype']).cumcount() + 1
    # Calculate cumulative population for each phenotype
    df['phenotype_cum_pop'] = df.groupby(['world_id', 'phenotype'])['pop'].cumsum()

    # ROLE-LEVEL CALCULATIONS (use a temporary aggregated dataframe)
    # ---------------------------------------------------------------------
    # Create temporary dataframe with role-level aggregations
    temp_role_df = df.groupby(['world_id', 'step_num', 'role'])['pop'].sum().reset_index(name='role_pop')
    # Drop zero-pop roles before computing activity
    temp_role_df = temp_role_df[temp_role_df['role_pop'] > 0]
    temp_role_df = temp_role_df.sort_values(['world_id', 'role', 'step_num'])
    # Add total_pop and prev_total_pop to temp_role_df via mapping
    temp_role_df['total_pop'] = temp_role_df.set_index(['world_id', 'step_num']).index.map(step_totals)
    temp_role_df['prev_total_pop'] = temp_role_df.set_index(['world_id', 'step_num']).index.map(prev_step_totals)
    # Calculate previous step's role population
    temp_role_df['role_prev_pop'] = temp_role_df.groupby(['world_id', 'role'])['role_pop'].shift(1)
    # Calculate role-level delta_N
    temp_role_df['role_expected_prop'] = np.where(
        temp_role_df['prev_total_pop'] > 0,
        temp_role_df['role_prev_pop'] / temp_role_df['prev_total_pop'],
        0
    )
    temp_role_df['role_observed_prop'] = temp_role_df['role_pop'] / temp_role_df['total_pop']
    temp_role_df['role_delta_N'] = np.where(
        temp_role_df['role_observed_prop'] > temp_role_df['role_expected_prop'],
        temp_role_df['total_pop'] * (temp_role_df['role_observed_prop'] - temp_role_df['role_expected_prop'])**2,
        0
    )
    temp_role_df = temp_role_df.drop(['role_expected_prop', 'role_observed_prop'], axis=1)
    # Calculate role activity (persistence counter)
    temp_role_df['role_activity'] = temp_role_df.groupby(['world_id', 'role']).cumcount() + 1
    # Calculate cumulative population correctly at role level
    temp_role_df['role_cum_pop'] = temp_role_df.groupby(['world_id', 'role'])['role_pop'].cumsum()
    # Calculate role growth rate
    temp_role_df['role_growth_rate'] = np.where(
        temp_role_df['role_prev_pop'] > 0,
        (temp_role_df['role_pop'] - temp_role_df['role_prev_pop']) / temp_role_df['role_prev_pop'],
        0
    )
    # Merge role-level metrics back to the main dataframe
    role_metrics = temp_role_df[['world_id', 'step_num', 'role', 'role_delta_N',
                                 'role_cum_pop', 'role_activity', 'role_growth_rate']]
    df = pd.merge(df, role_metrics, on=['world_id', 'step_num', 'role'], how='left')
    del temp_role_df
    del role_metrics

    # Zero out differential metrics at shadow model reset points and first step
    first_step = df['step_num'].min()
    if snap_interval is not None:
        snap_steps = (df['step_num'] % snap_interval == 0) | (df['step_num'] == first_step)
    else:
        snap_steps = df['step_num'] == first_step

    zero_columns = ['role_delta_N', 'phenotype_delta_N', 'phenotype_growth_rate', 'role_growth_rate']
    df.loc[snap_steps, zero_columns] = 0

    return df

def identify_adaptive_entities(selection_df, shadow_df, component='role', metric='growth_rate', percentile=0.99):
    """
    Identify entities that show evidence of adaptation compared to the shadow model.

    Entities are considered adaptive if their selection metric exceeds a threshold
    determined by the distribution of the same metric in the shadow model.

    Args:
        selection_df: DataFrame containing selection model data
        shadow_df: DataFrame containing shadow (neutral) model data
        component: Entity type to analyze ('role' or 'phenotype')
        metric: Column name of the metric used to identify adaptation
        percentile: Threshold percentile from shadow model distribution

    Returns:
        tuple: (dict of adaptive entities by world, series of proportion of adaptive entities)
    """
    # Choose aggregation method based on metric name
    if 'activity' in metric:
        agg_method = 'max'  # Use max for activity metrics - shows persistence
    else:
        agg_method = 'sum'  # Use sum for other metrics (pop, delta_N, growth_rate)

    # Calculate total metric value for each entity in both models
    shadow_metric = shadow_df.groupby(['world_id', component])[metric].agg(agg_method)
    selection_metric = selection_df.groupby(['world_id', component])[metric].agg(agg_method)

    # Calculate threshold from shadow model distribution
    threshold = shadow_metric.groupby('world_id').quantile(percentile)

    # Identify entities exceeding threshold in each world
    adaptive_entities = {
        'selection': {world_id: selection_metric.xs(world_id, level=0)[selection_metric.xs(world_id, level=0) > threshold]
                         for world_id, threshold in threshold.items()},
        'shadow': {world_id: shadow_metric.xs(world_id, level=0)[shadow_metric.xs(world_id, level=0) > threshold]
                            for world_id, threshold in threshold.items()}
    }

    # Calculate proportion of adaptive entities in each world
    proportions = pd.Series([
            len(adaptive_entities['selection'][world_id])/len(selection_metric.xs(world_id, level=0))
            for world_id in threshold.index
        ], index=threshold.index)

    return adaptive_entities, proportions

def calculate_evolutionary_activity_stats(df):
    """
    Calculate summary statistics for evolutionary activity across timesteps.

    Aggregates and summarizes various evolutionary metrics including activity,
    population, diversity and selection measures at each timestep.

    Args:
        df: DataFrame with evolutionary activity columns (from add_evolutionary_activity_columns)

    Returns:
        DataFrame with summary statistics by world_id and step_num
    """

    # Group data by world and timestep
    grouped = df.groupby(['world_id', 'step_num'])

    # Calculate summary statistics for phenotypes
    phenotype_acum = grouped['phenotype_activity'].sum()
    phenotype_pcum = grouped['phenotype_cum_pop'].sum()
    phenotype_nunique = grouped['phenotype'].nunique()

    # Calculate non-neutral activity metrics
    phenotype_delta_N = grouped['phenotype_delta_N'].sum()

    # Calculate diversity metrics (Shannon entropy)
    phenotype_entropy = []
    role_entropy = []

    for _, group in grouped:
        pop_sum = group['pop'].sum()
        proportions = group['pop'] / pop_sum
        phenotype_entropy.append(shannon_entropy(proportions))
        role_pop = group.groupby('role')['pop'].sum()
        role_proportions = role_pop / pop_sum
        role_entropy.append(shannon_entropy(role_proportions))

    # Role-level metrics (need to aggregate by role first to avoid duplication)
    role_df = df.drop_duplicates(subset=['world_id', 'step_num', 'role'])[
        ['world_id', 'step_num', 'role', 'role_activity', 'role_cum_pop', 'role_delta_N']
    ]

    role_grouped = role_df.groupby(['world_id', 'step_num'])
    role_acum = role_grouped['role_activity'].sum()
    role_pcum = role_grouped['role_cum_pop'].sum()
    role_nunique = role_grouped['role'].nunique()
    role_delta_N = role_grouped['role_delta_N'].sum()

    # Combine all metrics into a single dataframe
    stats_df = pd.DataFrame({
        'world_id': phenotype_acum.index.get_level_values('world_id'),
        'step_num': phenotype_acum.index.get_level_values('step_num'),
        'phenotype_acum': phenotype_acum,
        'phenotype_acum_mean': phenotype_acum / phenotype_nunique,
        'phenotype_pcum': phenotype_pcum,
        'phenotype_pcum_mean': phenotype_pcum / phenotype_nunique,
        'role_acum': role_acum,
        'role_acum_mean': role_acum / role_nunique,
        'role_pcum': role_pcum,
        'role_pcum_mean': role_pcum / role_nunique,
        'phenotype_diversity': phenotype_nunique,
        'role_diversity': role_nunique,
        'phenotype_entropy': phenotype_entropy,
        'role_entropy': role_entropy,
        'phenotype_delta_N': phenotype_delta_N,
        'role_delta_N': role_delta_N
    })

    return stats_df

def plot_world_data(df, shadow_df, world_ids, target_column):
    """
    Create comparative plots of selection and shadow model data for specified worlds.

    For each world ID, generates a plot showing the target column's values over time
    for both selection and shadow (neutral) models.

    Args:
        df: DataFrame containing selection model data
        shadow_df: DataFrame containing shadow (neutral) model data
        world_ids: List of world IDs to plot
        target_column: Column name to plot on y-axis
    """
    for world_id in world_ids:
        filtered_df = df[df['world_id'] == world_id]
        filtered_shadow_df = shadow_df[shadow_df['world_id'] == world_id]

        plt.figure(figsize=(6.5, 4.875))

        plt.plot(filtered_df['step_num'], filtered_df[target_column], label='Selection Model', color='blue')
        plt.plot(filtered_shadow_df['step_num'], filtered_shadow_df[target_column], label='Neutral Shadow Model', color='gray')

        plt.title(f'World {world_id}: {target_column} over Steps')
        plt.xlabel('Step Number')
        plt.ylabel(target_column)
        plt.legend()
        plt.show()

def gen_activity_plots(
    df: pd.DataFrame,
    wd: Dict[int, int],
    activity_col: str,
    save: bool = False,
    suffix: str = '',
    dest: str = '',
    id_list: list = None
) -> None:
    """
    Generate scatter plots of evolutionary activity for each world.

    Creates plots showing activity metrics over time for specified worlds,
    with options to save the plots to disk.

    Args:
        df: DataFrame containing activity data
        wd: Dictionary mapping world IDs to network IDs
        activity_col: Column name of activity metric to plot
        save: Whether to save plots to disk
        suffix: String to append to filenames if saving
        dest: Destination directory if saving
        id_list: Specific world IDs to plot, or None for all worlds
    """
    if id_list is None:
        id_list = [i for i in df['world_id'].unique()]

    for i in id_list:
        n_id = wd[i]
        title = f"Network: {n_id} | World: {i}"

        activity_df = df[df['world_id'] == i]
        activity_df.plot(x='step_num', y=activity_col, kind='scatter', s=0.1, alpha=0.2,
                         title=title, xlabel="Step", ylabel="Activity", legend=False)

        if save:
            plt.savefig(f"{dest}{activity_col}_n{n_id}w{i}{suffix}.png", dpi=300)
            plt.close()
        else:
            plt.show()


class EvolutionaryActivityStatsPlot(BasePlot):
    """
    Generates plots comparing evolutionary activity metrics between selection and shadow models.

    This class calculates and visualizes various evolutionary statistics including diversity,
    accumulation metrics, selection differentials, and adaptive entity counts. Each metric
    is plotted to show the difference between the selection model and the neutral shadow model,
    highlighting evidence of evolutionary adaptation.
    """

    title_map = {
        # Basic evolutionary activity metrics
        'phenotype_acum': 'Phenotype Cumulative Activity',
        'phenotype_acum_mean': 'Mean Phenotype Cumulative Activity',
        'phenotype_pcum': 'Phenotype Cumulative Population',
        'phenotype_pcum_mean': 'Mean Phenotype Cumulative Population',
        'role_acum': 'Role Cumulative Activity',
        'role_acum_mean': 'Mean Role Cumulative Activity',
        'role_pcum': 'Role Cumulative Population',
        'role_pcum_mean': 'Mean Role Cumulative Population',

        # Diversity metrics
        'phenotype_diversity': 'Phenotype Diversity',
        'role_diversity': 'Role Diversity',
        'phenotype_entropy': 'Phenotype Entropy',
        'role_entropy': 'Role Entropy',

        # Non-Netural Activity metrics
        'phenotype_delta_N': 'Phenotype Non-Netural Activity',
        'role_delta_N': 'Role Non-Netural Activity',

        # Novel adaptive entity metrics
        'novel_adaptive_roles_pop': 'Novel Adaptive Roles (by population count)',
        'novel_adaptive_roles_activity': 'Novel Adaptive Roles (by activity counter)',
        'novel_adaptive_roles_delta_N': 'Novel Adaptive Roles (by non-neutral activity)',
        'novel_adaptive_roles_growth_rate': 'Novel Adaptive Roles (by growth rate)',
        'novel_adaptive_phenotypes_pop': 'Novel Adaptive Phenotypes (by population count)',
        'novel_adaptive_phenotypes_activity': 'Novel Adaptive Phenotypes (by activity counter)',
        'novel_adaptive_phenotypes_delta_N': 'Novel Adaptive Phenotypes (by non-neutral activity)',
        'novel_adaptive_phenotypes_growth_rate': 'Novel Adaptive Phenotypes (by growth rate)'
    }

    ylabel_map = {
        # Basic evolutionary activity metrics
        'phenotype_acum': 'Cumulative Activity Count',
        'phenotype_acum_mean': 'Mean Activity Count per Phenotype',
        'phenotype_pcum': 'Cumulative Population',
        'phenotype_pcum_mean': 'Mean Population per Phenotype',
        'role_acum': 'Cumulative Activity Count',
        'role_acum_mean': 'Mean Activity Count per Role',
        'role_pcum': 'Cumulative Population',
        'role_pcum_mean': 'Mean Population per Role',

        # Diversity metrics
        'phenotype_diversity': 'Number of Phenotypes',
        'role_diversity': 'Number of Roles',
        'phenotype_entropy': 'Shannon Entropy',
        'role_entropy': 'Shannon Entropy',

        # Selection metrics
        'phenotype_delta_N': 'Non-Netural Activity',
        'role_delta_N': 'Non-Netural Activity',

        # Novel adaptive entity metrics
        'novel_adaptive_roles_pop': 'Novel by Pop',
        'novel_adaptive_roles_activity': 'Novel by Activity',
        'novel_adaptive_roles_delta_N': 'Novel by Delta N',
        'novel_adaptive_roles_growth_rate': 'Novel by Growth Rate',
        'novel_adaptive_phenotypes_pop': 'Novel by Pop',
        'novel_adaptive_phenotypes_activity': 'Novel by Activity',
        'novel_adaptive_phenotypes_delta_N': 'Novel by Delta N',
        'novel_adaptive_phenotypes_growth_rate': 'Novel by Growth Rate'
    }

    def __init__(self, db_loc: str, db_name: str, output_directory: str) -> None:
        super().__init__(db_loc, db_name, output_directory)

    def prepare_stats_df(self, world_id, percentile=0.95):
        snap_interval = self.params.loc[self.params['world_id'] == world_id, 'snap_interval'].iloc[0]

        df = db.get_phenotypes_df(self.db_path, shadow=False, worlds=world_id)
        s_df = db.get_phenotypes_df(self.db_path, shadow=True, worlds=world_id)

        if df.empty or s_df.empty:
            return None, None

        df = add_evolutionary_activity_columns(df)
        s_df = add_evolutionary_activity_columns(s_df, snap_interval=snap_interval)

        stats_df = calculate_evolutionary_activity_stats(df)
        s_stats_df = calculate_evolutionary_activity_stats(s_df)

        standard_metrics = [
            'pop',
            'delta_N',
            'growth_rate',
            'activity'
        ]

        for entity_type in ['role', 'phenotype']:
            for base_metric in standard_metrics:
                if base_metric == 'pop':
                    metric_name = base_metric
                else:
                    metric_name = f"{entity_type}_{base_metric}"

                novel_col = f"novel_adaptive_{entity_type}s_{base_metric}"

                adaptive_entities, _ = identify_adaptive_entities(
                    df, s_df,
                    component=entity_type,
                    metric=metric_name,
                    percentile=percentile
                )
                for model_type, model_df, stats in [('selection', df, stats_df), ('shadow', s_df, s_stats_df)]:
                    stats[novel_col] = 0
                    adaptive_set = adaptive_entities[model_type].get(world_id, pd.Series()).index

                    if len(adaptive_set) > 0:
                        appearances = model_df[model_df[entity_type].isin(adaptive_set)].groupby(entity_type)['step_num'].min()
                        counts = appearances.reset_index().groupby('step_num').size()

                        for step, count in counts.items():
                            stats.loc[stats['step_num'] == step, novel_col] = count

        return stats_df, s_stats_df

    def plot(self, world_ids=None, columns_to_plot=None, column_groups=None, percentile=0.95):
        """
        Create evolutionary activity comparison plots for all worlds.

        For each world in the database, generates comparison plots showing both
        selection and shadow model data for specified evolutionary metrics.

        Args:
            columns_to_plot: List of specific metrics to plot, or None for defaults
            column_groups: Groups of related metrics to plot together, or None for
                          individual plots
            percentile: Threshold percentile for identifying adaptive entities
        """
        if columns_to_plot and column_groups:
            raise ValueError("Specify either columns_to_plot or column_groups, not both.")
        if not columns_to_plot and not column_groups:
            columns_to_plot = list(self.title_map.keys())

        if world_ids is not None:
            worlds = {w:n for w,n in self.world_dict.items() if w in world_ids}
        else:
            worlds = self.world_dict

        for world_id, network_id in worlds.items():
            stats_df, s_stats_df = self.prepare_stats_df(world_id, percentile)
            if stats_df is None or s_stats_df is None:
                logging.info(f"Skipping plots for World {world_id} (Network {network_id}) - no data available")
                continue
            if column_groups is not None:
                self.plot_grouped_stats(stats_df, s_stats_df, world_id, network_id, column_groups)
            else:
                self.plot_stats(stats_df, s_stats_df, world_id, network_id, columns_to_plot)

    def plot_stats(
        self,
        stats_df: pd.DataFrame,
        s_stats_df: pd.DataFrame,
        world_id: int,
        network_id: int,
        columns_to_plot: List[str],
        ylabels: Optional[Dict[str, str]] = None
    ) -> None:
        for column in columns_to_plot:
            plt.figure(figsize=(6.5, 4.875))
            plt.plot(s_stats_df['step_num'], s_stats_df[column], label='Neutral Shadow', color='gray')
            plt.plot(stats_df['step_num'], stats_df[column], label='Selection', color='blue')
            plt.title(f"{self.title_map.get(column, column.replace('_', ' ').title())}\nNetwork: {network_id}, World: {world_id}")
            plt.xlabel("Step")
            plt.ylabel(ylabels.get(column) if ylabels and column in ylabels else self.ylabel_map.get(column, column.replace('_', ' ').title()))
            plt.legend()

            filename = f"{self.plot_dir}/{column}_n{network_id}w{world_id}.png"
            plt.savefig(filename, dpi=300)
            plt.close()

    def plot_grouped_stats(
        self,
        stats_df: pd.DataFrame,
        s_stats_df: pd.DataFrame,
        world_id: int,
        network_id: int,
        column_groups: List[Union[List[str], Dict[str, Any]]],
        ylabels: Optional[Dict[str, str]] = None
    ):
        for group_idx, group in enumerate(column_groups):
            # Check if the group is a dict with title and columns
            if isinstance(group, dict) and 'plots' in group:
                plots = group['plots']
                group_title = group.get('title')  # Use None if 'title' key doesn't exist
            else:
                # If it's just a list of columns, use no title
                plots = [{'column': col} for col in group]
                group_title = None

            num_plots = len(plots)

            fig, axes = plt.subplots(num_plots, 1, figsize=(6.5, 2*num_plots), sharex=True)
            if num_plots == 1:
                axes = [axes]

            for ax, plot_cfg in zip(axes, plots):
                col = plot_cfg['column']
                ylabel = plot_cfg.get('ylabel', self.ylabel_map.get(col, col.replace('_', ' ').title()))
                ax.plot(s_stats_df['step_num'], s_stats_df[col], label='Neutral Shadow', color='gray')
                ax.plot(stats_df['step_num'], stats_df[col], label='Selection', color='blue')
                ax.set_ylabel(ylabel)
                ax.legend()

            if group_title is not None:
                fig.suptitle(f"{group_title}\nNetwork: {network_id}, World: {world_id}")
            else:
                fig.suptitle(f"Network: {network_id}, World: {world_id}")

            axes[-1].set_xlabel("Step")

            plt.tight_layout()
            plt.subplots_adjust(top=0.9, hspace=0.1)

            filename = f"{self.plot_dir}/grouped_stats_n{network_id}w{world_id}_group{group_idx}.png"
            plt.savefig(filename, dpi=300)
            plt.close()


def get_CAD(df: pd.DataFrame, activity_col: str) -> pd.DataFrame:
    cad_df = df.groupby(['step_num', activity_col]).size().unstack(fill_value=0)
    return cad_df


class ActivityWavePlot(BasePlot):
    """
    Generates scatter plots showing evolutionary activity metrics over time.
    These plots show the "waves" of activity as different phenotypes or roles
    appear, spread, and potentially disappear throughout the simulation.

    Creates separate plots for selection and shadow models, similar to PopulationPlot.
    """

    def __init__(self, db_loc: str, db_name: str, output_directory: str) -> None:
        super().__init__(db_loc, db_name, output_directory)

    def plot(self, activity_columns=None, world_ids=None):
        """
        Create activity wave plots for specified activity metrics across selected worlds.
        Generates separate plots for selection and shadow models.

        Args:
            activity_columns: List of activity metrics to plot, or None for defaults
                            (defaults to phenotype/role activity and cumulative population)
            world_ids: List of specific world IDs to plot, or None for all worlds
        """
        if activity_columns is None:
            activity_columns = ['phenotype_activity', 'phenotype_cum_pop',
                              'role_activity', 'role_cum_pop']

        if world_ids is None:
            world_ids = list(self.world_dict.keys())
        elif isinstance(world_ids, int):
            world_ids = [world_ids]

        # Process both selection (shadow=False) and shadow (shadow=True) data
        for shadow in [False, True]:
            for world_id in world_ids:
                network_id = self.world_dict[world_id]

                # Get phenotypes data
                df = db.get_phenotypes_df(self.db_path, shadow=shadow, worlds=world_id)
                if df.empty:
                    logging.info(f"No {'shadow' if shadow else 'selection'} data for world {world_id}")
                    continue

                # Add evolutionary activity columns
                if shadow:
                    snap_interval = self.params.loc[
                        self.params['world_id'] == world_id, 'snap_interval'
                    ].iloc[0] if not self.params.empty else None

                    df = add_evolutionary_activity_columns(df, snap_interval=snap_interval)
                else:
                    df = add_evolutionary_activity_columns(df)

                # Create plots for each activity column
                for activity_col in activity_columns:
                    if activity_col not in df.columns:
                        logging.warning(f"Column {activity_col} not found in data for world {world_id}")
                        continue

                    self.create_activity_plot(df, activity_col, world_id, network_id, shadow)

    def create_activity_plot(self, df, activity_col, world_id, network_id, shadow):
        """Create a single activity wave plot."""
        plt.figure(figsize=(6.5, 4.875))

        # Plot data
        df.plot(x='step_num', y=activity_col, kind='scatter',
               s=0.1, alpha=0.2, ax=plt.gca(), legend=False)

        # Add title
        model_type = "Shadow" if shadow else "Selection"
        plt.title(f"{activity_col.replace('_', ' ').title()} ({model_type})\nNetwork: {network_id} | World: {world_id}")
        plt.xlabel("Step")
        plt.ylabel(activity_col.replace('_', ' ').title())

        plt.tight_layout()

        # Save the plot with appropriate prefix
        prefix = "s_" if shadow else ""
        filename = f"{self.plot_dir}/{prefix}{activity_col}_n{network_id}_w{world_id}.png"
        plt.savefig(filename, dpi=300)
        plt.close()


class CADPlot(BasePlot):
    """
    Generates Cumulative Activity Distribution plots for evolutionary activity metrics.

    CAD plots show the frequency distribution of activity values on a log-log scale,
    allowing comparison between selection and shadow models to identify evidence of
    evolutionary adaptation through differences in their distributions.
    """
    def __init__(self, db_loc: str, db_name: str, output_directory: str) -> None:
        super().__init__(db_loc, db_name, output_directory)
        self.world_dict = db.get_world_dict(self.db_path)

    def plot(self, activity_columns=None):
        """
        Create CAD plots for specified activity metrics across all worlds.

        For each world and activity metric, generates a log-log plot comparing
        the cumulative frequency distributions of the selection and shadow models.

        Args:
            activity_columns: List of activity metrics to plot, or None for defaults
                            (defaults to phenotype/role activity and cumulative population)
        """
        if activity_columns is None:
            activity_columns = ['phenotype_activity', 'phenotype_cum_pop', 'role_activity', 'role_cum_pop']

        for world_id, network_id in self.world_dict.items():
            df = db.get_phenotypes_df(self.db_path, shadow=False, worlds=world_id)
            s_df = db.get_phenotypes_df(self.db_path, shadow=True, worlds=world_id)

            if df.empty or s_df.empty:
                continue

            df = add_evolutionary_activity_columns(df)
            s_df = add_evolutionary_activity_columns(s_df)

            for column in activity_columns:
                self.plot_cad(df, s_df, column, world_id, network_id)

    def plot_cad(self, df: pd.DataFrame, s_df: pd.DataFrame, activity_col: str, world_id: int, network_id: int):
        plt.figure(figsize=(6.5, 4.875))

        entity_type = 'phenotype' if 'phenotype' in activity_col else 'role'

        CAD = df.groupby(entity_type)[activity_col].max().value_counts().sort_index()
        sCAD = s_df.groupby(entity_type)[activity_col].max().value_counts().sort_index()

        total = CAD.sum() + sCAD.sum()

        CAD = CAD / total
        sCAD = sCAD / total

        sCAD.ewm(span=20).mean().plot(loglog=True, label='Neutral Shadow', color='gray')
        CAD.ewm(span=20).mean().plot(loglog=True, label='Selection', color='blue')

        plt.title(f"CAD Plot: {activity_col.replace('_', ' ').title()}\nNetwork: {network_id}, World: {world_id}")
        plt.xlabel("Activity")
        plt.ylabel("Cumulative Frequency")
        plt.legend()

        filename = f"{self.plot_dir}/cad_{activity_col}_n{network_id}w{world_id}.png"
        plt.savefig(filename, dpi=300)
        plt.close()


class RolesetPlot(BasePlot):
    """
    Generates visualizations of role categories (possible, viable, adjacent,
    occupiable, occupied) at different model time steps.

    Uses the RoleAnalyzer from rolesets.py to analyze the adaptive
    landscape and plot how different role categories change over time.
    """

    def __init__(self, db_loc: str, db_name: str, output_directory: str) -> None:
        super().__init__(db_loc, db_name, output_directory)

    def plot(self, world_ids=None, num_steps=10, sites=None):
        """
        Create plots showing role category counts over time.

        For each world, plots the number of roles in each category
        (possible, viable, adjacent, occupiable, occupied) at evenly spaced
        time steps.

        Args:
            world_ids: List of specific world IDs to plot, or None for all worlds
            num_steps: Number of evenly-spaced time steps to analyze
            sites: Specific sites to analyze, or None for all sites
        """

        # If world_ids is None, use all worlds
        if world_ids is None:
            world_ids = list(self.world_dict.keys())
        elif isinstance(world_ids, int):
            world_ids = [world_ids]

        # For each world, create feature calculator and analyzer
        for world_id in world_ids:
            logging.info(f"Trying world {world_id}...")
            network_id = self.world_dict[world_id]

            # Create utility calculator and analyzer
            calculator = FeatureUtilityCalculator(self.db_path)
            analyzer = RoleAnalyzer(calculator)

            # Get evenly spaced steps for analysis
            steps = self.get_evenly_spaced_steps(num_steps)

            # Get sites for this world if not specified
            world_sites = sites
            if world_sites is None:
                sites_dict = db.get_sites_dict(self.db_path)
                if world_id in sites_dict:
                    world_sites = list(sites_dict[world_id].keys())
                else:
                    continue  # Skip this world if no sites found

            # Evaluate rolesets for this world
            roleset_data = analyzer.evaluate_rolesets(world_id, steps)

            if roleset_data.empty:
                logging.info(f"No roleset data available for World {world_id}")
                continue

            # Create plots
            self.plot_roleset_categories(roleset_data, world_id, network_id)
            self.plot_roleset_ratios(roleset_data, world_id, network_id)

            logging.info(f"World {world_id} complete")

    def get_evenly_spaced_steps(self, num_steps: int) -> List[int]:
        """Get evenly spaced step numbers from the database."""
        # Reusing the method from NetworkPlot
        all_steps = db.get_steps_list(self.db_path)
        if num_steps >= len(all_steps):
            return all_steps
        interval = max(len(all_steps) // num_steps, 1)
        selected_steps = all_steps[::interval]
        selected_steps.append(all_steps[-1])  # Always include the last step
        return sorted(list(set(selected_steps)))  # Remove duplicates and sort

    def plot_roleset_categories(self, roleset_data, world_id, network_id):
        """
        Plot the absolute counts of roles in each category over time.

        Args:
            roleset_data: DataFrame containing roleset evaluation results
            world_id: ID of the world being plotted
            network_id: ID of the network the world belongs to
        """
        # Aggregate data by step (mean across sites)
        step_data = roleset_data.groupby('step').agg({
            'possible': 'mean',
            'sustainable': 'mean',
            'adjacent': 'mean',
            'occupiable': 'mean',
            'occupied': 'mean'
        }).reset_index()

        plt.figure(figsize=(6.5, 4.875))

        # Plot each category
        plt.plot(step_data['step'], step_data['possible'], label='Possible', color='lightgray')
        plt.plot(step_data['step'], step_data['sustainable'], label='Sustainable', color='blue')
        plt.plot(step_data['step'], step_data['adjacent'], label='Adjacent', color='green')
        plt.plot(step_data['step'], step_data['occupiable'], label='Occupiable', color='orange')
        plt.plot(step_data['step'], step_data['occupied'], label='Occupied', color='red')

        plt.title(f"Role Categories Over Time\nNetwork: {network_id}, World: {world_id}")
        plt.xlabel('Step')
        plt.ylabel('Number of Roles')
        plt.yscale('log')  # Log scale often works well for these numbers
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save the figure
        filename = f"{self.plot_dir}/roleset_categories_n{network_id}w{world_id}.png"
        plt.savefig(filename, dpi=300)
        plt.close()

    def plot_roleset_ratios(self, roleset_data, world_id, network_id):
        """
        Plot the ratios between different role categories over time.

        Args:
            roleset_data: DataFrame containing roleset evaluation results
            world_id: ID of the world being plotted
            network_id: ID of the network the world belongs to
        """
        # Aggregate data by step (mean across sites)
        step_data = roleset_data.groupby('step').agg({
            'possible': 'mean',
            'sustainable': 'mean',
            'adjacent': 'mean',
            'occupiable': 'mean',
            'occupied': 'mean'
        }).reset_index()

        # Calculate ratios
        step_data['sustainable_ratio'] = step_data['sustainable'] / step_data['possible']
        step_data['adjacent_ratio'] = step_data['adjacent'] / step_data['possible']
        step_data['adjacent2_ratio'] = step_data['adjacent'] / step_data['sustainable']
        step_data['occupiable_ratio'] = step_data['occupiable'] / step_data['adjacent']
        step_data['occupation_ratio'] = step_data['occupied'] / step_data['occupiable']

        plt.figure(figsize=(6.5, 4.875))

        # Plot each ratio
        plt.plot(step_data['step'], step_data['sustainable_ratio'], label='Sustainable/Possible', color='blue')  # Updated label
        plt.plot(step_data['step'], step_data['adjacent_ratio'], label='Adjacent/Possible', color='green')
        plt.plot(step_data['step'], step_data['adjacent2_ratio'], label='Adjacent/Sustainable', color='lightgreen')
        plt.plot(step_data['step'], step_data['occupiable_ratio'], label='Occupiable/Adjacent', color='orange')
        plt.plot(step_data['step'], step_data['occupation_ratio'], label='Occupied/Occupiable', color='red')

        plt.title(f"Role Category Ratios Over Time\nNetwork: {network_id}, World: {world_id}")
        plt.xlabel('Step')
        plt.ylabel('Ratio')
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save the figure
        filename = f"{self.plot_dir}/roleset_ratios_n{network_id}w{world_id}.png"
        plt.savefig(filename, dpi=300)
        plt.close()

def database_cleanup(db_loc, db_name):
    db_path = os.path.join(db_loc, db_name)
    db.migrate_broken_tables(db_path)

def validate_and_convert_floats(G):
    for _, node_data in G.nodes(data=True):
        for key, value in node_data.items():
            if isinstance(value, float):  # Check for float type (includes float32)
                node_data[key] = float(value)  # Convert to standard Python float

    for _, _, edge_data in G.edges(data=True):
        for key, value in edge_data.items():
            if isinstance(value, float):
                edge_data[key] = float(value)

def setup_logging(db_loc, db_name, job_name):
    process_id = os.getpid()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(db_loc, f"{job_name}_log_{timestamp}_{process_id}.log")

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_filename)
        ]
    )
    logging.info(f"Logging set up for process {process_id} and database {db_name}, log file: {log_filename}")

def check_for_binary_data(db_loc, db_name):
    db_path = os.path.join(db_loc, db_name)
    conn = db.get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    for table in tables:
        table_name = table[0]
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()

        for column in columns:
            col_name = column[1]
            col_type = column[2]
            cursor.execute(f"SELECT {col_name} FROM {table_name} WHERE typeof({col_name}) = 'blob' LIMIT 1;")
            if cursor.fetchone():
                logging.info(f"Binary data found in {db_name}, Table: {table_name}, Column: {col_name}, Expected Type: {col_type}")
    conn.close()

def find_db_files(directory):
    dbs_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.db'):
                dbs_list.append((root, file))
    return dbs_list

def run_analysis_from_config(args):
    config, db_loc, db_name = args
    job_name = config.get('job_name', '')
    setup_logging(db_loc, db_name, job_name)
    classes = config.get('classes', [])
    functions = config.get('functions', [])
    for class_entry in classes:
        class_name = class_entry['class_name']
        class_obj = globals()[class_name]
        output_directory = class_entry.get('output_directory', '')
        instance = class_obj(db_loc, db_name, output_directory)
        methods = class_entry.get('methods')
        for method_entry in methods:
            method_name = method_entry.get('method_name')
            method_kwargs = method_entry.get('kwargs', {})
            method = getattr(instance, method_name)
            message = "{class_name}.{method_name} with {method_kwargs} for {db_name} in {db_loc}"
            logging.info(message.format(
                class_name=class_name,
                method_name=method_name,
                method_kwargs=method_kwargs,
                db_name=db_name,
                db_loc=db_loc
            ))
            method(**method_kwargs)
            logging.info("Finished method: {method_name} for {class_name}".format(method_name=method_name, class_name=class_name))
    for function in functions:
        function_name = function['function_name']
        function_kwargs = function.get('kwargs', {})
        function_kwargs.update({'db_loc': db_loc, 'db_name': db_name})
        function_obj = globals()[function_name]
        message = "{function_name} with {kwargs} for {db_name} in {db_loc}"
        logging.info(message.format(function_name=function_name, kwargs=function_kwargs, db_name=db_name, db_loc=db_loc))
        function_obj(**function_kwargs)
        logging.info("Finished function: {function_name}".format(function_name=function_name))

def main():
    """
    Main entry point for the analysis script.

    Reads the configuration file specified as command-line argument,
    finds all databases in the target directory, and executes analysis
    tasks on each database, potentially in parallel.
    """
    if len(sys.argv) < 2:
        print("Usage: python script.py <config_file>")
        sys.exit(1)
    with open(sys.argv[1], 'r') as f:
        data = f.read()
    config = json.loads(data)
    target_directory = config['target_directory']
    dbs_list = [(config, l, n) for l,n in find_db_files(target_directory)]
    num_processes = os.environ.get('SLURM_CPUS_PER_TASK', None)
    if num_processes:
        num_processes = int(num_processes)
    with tqdm(total=len(dbs_list), desc="Databases Pool") as progress:
        with Pool(num_processes) as pool:
            for result in pool.imap_unordered(run_analysis_from_config, dbs_list):
                progress.update()

if __name__ == "__main__":
    main()