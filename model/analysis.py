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
from datetime import datetime
from multiprocessing import Pool
from typing import Any, Dict, List, Set, Tuple, Iterator, Callable, Optional
from tqdm import tqdm

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
        self.params = db.get_params_df(self.db_path)

    def get_or_create_output_directory(self):
        output_path = os.path.join(self.db_loc, self.output_directory)
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        return output_path


class PopulationPlot(BasePlot):

    def plot(self, plot_type = 'area', num_types = 1000, world=None):
        worlds_to_plot = [(world, self.world_dict[world])] if world is not None else self.world_dict.items()
        for shadow in [False, True]:
            s = ["", "s_"][shadow]
            for w, n in worlds_to_plot:
                logging.info(f"Trying world {w}")
                ph_df = db.get_phenotypes_df(self.db_path, shadow, worlds=w)
                if not ph_df.empty:
                    self.save_pivot_fig(ph_df, self.role_pivot, plot_type, shadow, w, n)
                    m = ph_df.groupby('phenotype')['pop'].sum()
                    phenotypes = tuple(m.nlargest(num_types).index)
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

    def plot(self, num_roles=30, num_bins=20, world=None):
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


class PayoffPlot(BasePlot):

    def __init__(
        self,
        db_loc: str,
        db_name: str,
        output_directory: str,
    ) -> None:
        super().__init__(db_loc, db_name, output_directory)
        self.interactions_df = db.get_interactions_df(self.db_path)

    def plot(self, plot_type, network_ids=None):
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
    proportions = np.array(proportions)
    nonzero_proportions = proportions[proportions != 0]
    if len(nonzero_proportions) == 0:
        return 0.0
    entropy = -np.sum(nonzero_proportions * np.log2(nonzero_proportions))
    return entropy

def add_evolutionary_activity_columns(df, snap_interval=None):
    df['phenotype_activity'] = df.groupby(['world_id', 'phenotype']).cumcount() + 1
    df['phenotype_cum_pop'] = df.groupby(['world_id', 'phenotype'])['pop'].cumsum()

    df['phenotype_expected_prop'] = df.groupby(['world_id', 'phenotype'])['pop'].shift(1) / df.groupby(['world_id', 'step_num'])['pop'].transform('sum').groupby(df['world_id']).shift(1)
    df['phenotype_observed_prop'] = df['pop'] / df.groupby(['world_id', 'step_num'])['pop'].transform('sum')
    df['phenotype_delta_N'] = np.where(df['phenotype_observed_prop'] > df['phenotype_expected_prop'],
                                    df.groupby(['world_id', 'step_num'])['pop'].transform('sum') *
                                    (df['phenotype_observed_prop'] - df['phenotype_expected_prop'])**2,
                                    0)
    df = df.drop(['phenotype_expected_prop', 'phenotype_observed_prop'], axis=1)

    df['role_step'] = df.groupby(['world_id', 'role', 'step_num']).ngroup()
    df['role_activity'] = df.groupby(['world_id', 'role'])['role_step'].transform(lambda x: pd.factorize(x)[0] + 1)
    df = df.drop('role_step', axis=1)

    df['role_pop'] = df.groupby(['world_id', 'step_num', 'role'])['pop'].transform('sum')
    df['role_cum_pop'] = df.groupby(['world_id', 'role'])['pop'].transform('cumsum')

    df['role_expected_prop'] = df.groupby(['world_id', 'role'])['role_pop'].shift(1) / df.groupby(['world_id', 'step_num'])['pop'].transform('sum').groupby(df['world_id']).shift(1)
    df['role_observed_prop'] = df['role_pop'] / df.groupby(['world_id', 'step_num'])['pop'].transform('sum')
    df['role_delta_N'] = np.where(df['role_observed_prop'] > df['role_expected_prop'],
                               df.groupby(['world_id', 'step_num'])['pop'].transform('sum') *
                               (df['role_observed_prop'] - df['role_expected_prop'])**2,
                               0)
    df = df.drop(['role_expected_prop', 'role_observed_prop', 'role_pop'], axis=1)
    if snap_interval is not None:
        df.loc[df['step_num'] % snap_interval == 0, 'role_delta_N'] = 0
        df.loc[df['step_num'] % snap_interval == 0, 'phenotype_delta_N'] = 0

    return df

def calculate_evolutionary_activity_stats(df):
    grouped = df.groupby(['world_id', 'step_num'])
    phenotype_acum = grouped['phenotype_activity'].sum()
    phenotype_pcum = grouped['phenotype_cum_pop'].sum()
    phenotype_nunique = grouped['phenotype'].nunique()
    role_acum = grouped['role_activity'].sum()
    role_pcum = grouped['role_cum_pop'].sum()
    role_nunique = grouped['role'].nunique()
    phenotype_delta_N = grouped['phenotype_delta_N'].sum()
    role_delta_N = grouped['role_delta_N'].sum()

    phenotype_entropy = []
    role_entropy = []

    for _, group in grouped:
        pop_sum = group['pop'].sum()
        proportions = group['pop'] / pop_sum
        phenotype_entropy.append(shannon_entropy(proportions))
        role_pop = group.groupby('role')['pop'].sum()
        role_proportions = role_pop / pop_sum
        role_entropy.append(shannon_entropy(role_proportions))

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

def get_CAD(df: pd.DataFrame, activity_col: str) -> pd.DataFrame:
    cad_df = df.groupby(['step_num', activity_col]).size().unstack(fill_value=0)
    return cad_df

def CAD_plot(
    df: pd.DataFrame,
    s_df: pd.DataFrame,
    activity_col: str,
    title: str,
) -> None:
    CAD = get_CAD(df, activity_col).sum()
    sCAD = get_CAD(s_df, activity_col).sum()

    total = CAD.sum() + sCAD.sum()

    CAD = CAD / total
    sCAD = sCAD / total

    CAD.plot(loglog=True, title=title)
    sCAD.plot(loglog=True, title=title)
    plt.show()

class EvolutionaryActivityStatsPlot(BasePlot):
    def __init__(self, db_loc: str, db_name: str, output_directory: str) -> None:
        super().__init__(db_loc, db_name, output_directory)

    def plot(self, columns_to_plot=None):
        if columns_to_plot is None:
            columns_to_plot = [
                'phenotype_acum', 'phenotype_acum_mean', 'phenotype_pcum', 'phenotype_pcum_mean',
                'role_acum', 'role_acum_mean', 'role_pcum', 'role_pcum_mean',
                'phenotype_diversity', 'role_diversity', 'phenotype_entropy', 'role_entropy',
                'phenotype_delta_N', 'role_delta_N'
            ]

        for world_id, network_id in self.world_dict.items():
            df = db.get_phenotypes_df(self.db_path, shadow=False, worlds=world_id)
            s_df = db.get_phenotypes_df(self.db_path, shadow=True, worlds=world_id)

            if df.empty or s_df.empty:
                continue

            df = add_evolutionary_activity_columns(df)
            s_df = add_evolutionary_activity_columns(s_df)

            stats_df = calculate_evolutionary_activity_stats(df)
            s_stats_df = calculate_evolutionary_activity_stats(s_df)

            self.plot_stats(stats_df, s_stats_df, world_id, network_id, columns_to_plot)

    def plot_stats(self, stats_df: pd.DataFrame, s_stats_df: pd.DataFrame, world_id: int, network_id: int, columns_to_plot: List[str]):
        for column in columns_to_plot:
            plt.figure(figsize=(6.5, 4.875))
            plt.plot(stats_df['step_num'], stats_df[column], label='Selection', color='blue')
            plt.plot(s_stats_df['step_num'], s_stats_df[column], label='Neutral Shadow', color='gray')
            plt.title(f"{column.replace('_', ' ').title()}\nNetwork: {network_id}, World: {world_id}")
            plt.xlabel("Step")
            plt.ylabel(column.replace('_', ' ').title())
            plt.legend()

            filename = f"{self.plot_dir}/{column}_n{network_id}w{world_id}.png"
            plt.savefig(filename, dpi=300)
            plt.close()

class CADPlot(BasePlot):
    def __init__(self, db_loc: str, db_name: str, output_directory: str) -> None:
        super().__init__(db_loc, db_name, output_directory)
        self.world_dict = db.get_world_dict(self.db_path)

    def plot(self, activity_columns=None):
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

        CAD = get_CAD(df, activity_col).sum()
        sCAD = get_CAD(s_df, activity_col).sum()

        total = CAD.sum() + sCAD.sum()

        CAD = CAD / total
        sCAD = sCAD / total

        CAD.plot(loglog=True, label='Selection', color='blue')
        sCAD.plot(loglog=True, label='Neutral Shadow', color='gray')

        plt.title(f"CAD Plot: {activity_col.replace('_', ' ').title()}\nNetwork: {network_id}, World: {world_id}")
        plt.xlabel("Activity")
        plt.ylabel("Cumulative Frequency")
        plt.legend()

        filename = f"{self.plot_dir}/cad_{activity_col}_n{network_id}w{world_id}.png"
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