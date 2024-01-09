import copy
import os
import json
import logging
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import networkx as nx
import pandas as pd
import database as db
import output as out
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

    def get_or_create_output_directory(self):
        output_path = os.path.join(self.db_loc, self.output_directory)
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        return output_path


class PopulationPlot(BasePlot):

    def plot(self):
        for shadow in [False, True]:
            s = ["", "s_"][shadow]
            for w, n in self.world_dict.items():
                logging.info(f"Trying world {w}")
                ph_df = db.get_phenotypes_df(self.db_path, shadow, worlds=w)
                if not ph_df.empty:
                    self.save_pivot_fig(ph_df, self.role_pivot, shadow, w, n)
                    m = ph_df.groupby('phenotype')['pop'].sum()
                    phenotypes = tuple(m.nlargest(1000).index)
                    if len(phenotypes) == 1:
                        phenotypes = phenotypes[0]
                    ph_df = db.get_phenotypes_df(
                        self.db_path, shadow,
                        worlds=w,
                        phenotypes=phenotypes
                    )
                    self.save_pivot_fig(ph_df, self.phenotype_pivot, shadow, w, n)

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

    def pop_plot(self, df: pd.DataFrame, title: str, legend = False) -> None:
        df.ewm(span=20).mean().plot.area(
            legend=legend,
            title=title,
            xlabel="Step",
            ylabel="Population",
            figsize=(19.2, 9.66)
        )

class ModelVarsPlot(BasePlot):

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
                try:
                    df[nd[n]].ewm(span=100).mean().plot(
                        title=title,
                        xlabel="Step",
                        ylabel=name,
                        figsize=(19.2, 9.66),
                    )
                    plt.tight_layout()
                    plt.savefig(f"{self.plot_dir}/{column}_{n}.png")
                    plt.close()
                except KeyError as e:
                    logging.error(e)


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
        plt.figure(figsize=(10, 8)) 
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
        plt.savefig(file_path)
        plt.close()

    def plot_features_networks(
        self,
        network_ids: Optional[List[int]] = None,
        world_ids: Optional[List[int]] = None,
        steps: Optional[List[int]] = None
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
                plot_steps = steps if steps else G.graph.get('change_steps', [])
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
        networks_data = {}
        i_df = self.interactions_df
        fc_df = self.feature_changes_df
        all_sites = set(site for world_sites in self.sites_dict.values() for site in world_sites)

        for wid, nid in self.world_dict.items():
            if nid not in networks_data:
                networks_data[nid] = {}
            networks_data[nid][wid] = {site: {} for site in all_sites}

        for nid, worlds in networks_data.items():
            logging.info(f"Beginning network {nid}")
            for wid, sites in worlds.items():
                logging.info(f"Beginning world {wid}")
                for site in sites:
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
                    network_json = nx.node_link_data(role_network)
                    networks_data[nid][wid][site] = network_json
        logging.info("Preparing to write json file")
        output_file_path = os.path.join(self.db_loc, "role_networks_data.json")
        with open(output_file_path, 'w') as json_file:
            json.dump(networks_data, json_file, indent=4)

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
        plt.savefig(file_path)
        plt.close()

    def plot_role_networks(
        self,
        network_ids: Optional[List[int]] = None,
        world_ids: Optional[List[int]] = None,
        steps: Optional[List[int]] = None
        ) -> None:        
        json_file_path = os.path.join(self.db_loc, "role_networks_data.json")
        if not os.path.isfile(json_file_path):
            raise FileNotFoundError(f"File not found at {json_file_path}")

        role_networks_dir = os.path.join(self.plot_dir, 'role_networks')
        os.makedirs(role_networks_dir, exist_ok=True)

        with open(json_file_path, 'r') as file:
            networks_data = json.load(file)
        for nid, worlds in networks_data.items():
            if network_ids is not None and nid not in map(str, network_ids):
                continue
            logging.info(f"Beginning network {nid}")
            for wid, sites in worlds.items():
                if world_ids is not None and wid not in map(str, world_ids):
                    continue
                logging.info(f"Beginning world {wid}")
                for site_id, site_data in sites.items():
                    logging.info(f"Beginning site {site_id}")
                    G = nx.node_link_graph(site_data)
                    plot_steps = steps if steps else G.graph.get('change_steps', []) 
                    for step in plot_steps:
                        logging.info(f"Updating role network for world {wid}{site_id} at step {step}")
                        ph_df = db.get_phenotypes_df(
                            self.db_path,
                            shadow=False,
                            worlds=wid,
                            sites=site_id,
                            steps=step
                        )
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