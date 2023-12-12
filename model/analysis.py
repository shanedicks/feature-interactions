import copy
import os
import json
import logging
import sys
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import database as db
import output as out
from matplotlib.colors import ListedColormap
from multiprocessing import Pool
from typing import Any, Dict, List, Set, Tuple, Iterator, Callable, Optional
from tqdm import tqdm

def role_column(df: pd.DataFrame) -> None:
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


class PopulationPlot(BasePlot):

    def plot(self):
        for shadow in [False, True]:
            s = ["", "s_"][shadow]
            for w, n in self.world_dict.items():
                logging.info(f"Trying world {w}")
                ph_df = db.get_phenotypes_df(self.db_path, shadow, worlds=w)
                if not ph_df.empty:
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

    def set_node_postions(
        self,
        G: nx.DiGraph,
        env_nodes: List[str],
        other_nodes: List[str]
    ) -> None:
        env_positions = {node: (i/2, 0) for i, node in enumerate(sorted(env_nodes))}
        other_positions = nx.spring_layout(G.subgraph(other_nodes), k=2, seed=21)
        min_x = min(x for x, y in role_positions.values())
        min_y = min(y for x, y in role_positions.values())
        pos = {
            **{n: (x - min_x, y - min_y + 1) for n, (x,y) in other_positions.items()},
            **env_positions
        }
        nx.set_node_attributes(G, pos, "pos")

    def construct_features_network(
        self,
        features: pd.DataFrame,
        interactions: pd.DataFrame,
    ) -> nx.DiGraph:
        G = nx.DiGraph()
        env_features = features[features['env']==1]['name']
        agent_features = features[features['env']==0]['name']
        G.add_nodes_from(
            env_features,
            type='agent',
            node_color='skyblue',
        )
        G.add_nodes_from(
            agent_features,
            type='env',
            node_color='lightgreen',
            node_shape='s',
        )
        G.add_edges_from(list(zip(interactions['i_name'], interactions['t_name'])))
        self.set_node_postions(G, env_features, agent_features)
        return G

    def export_features_networks_data(self) -> None:
        fc_df = db.get_feature_changes_df(self.db_path)
        i_df = db.get_interactions_df(self.db_path)

        networks_data = {}

        for network_id in set(self.world_dict.values()):
            interactions_subset = i_df[i_df['network_id'] == network_id]
            world_ids = [wid for wid, nid in self.world_dict.items() if nid == network_id]
            features_subset = fc_df[fc_df['world_id'].isin(world_ids)]
            features_network = self.construct_features_network(
                features_subset,
                interactions_subset
            )
            network_json = nx.node_link_data(features_network)
            networks_data[network_id] = network_json

        output_file_path = os.path.join(self.db_loc, "features_network_data.json")
        with open(output_file_path, 'w') as json_file:
            json.dump(networks_data, json_file, indent=4)

    def draw_feature_interactions(
        self,
        G: nx.DiGraph,
        file_path: str
        ) -> None:
        pos = {node: data['pos'] for node, data in G.nodes(data=True)}
        labels = {n: n for n in pos.keys()}
        for node, data in G.nodes(data=True):
            node_color, node_shape = data['node_color'], data['node_shape']
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=[node],
                node_color=node_color,
                node_shape=node_shape
            )
        nx.draw_networkx_edges(G, pos)
        nx.draw_networkx_labels(G, pos, labels)
        plt.savefig(file_path)
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
        role_relations = []
        role_nodes = ph_df['role'].unique()
        for role1 in role_nodes:
            G.add_node(role1, type='role')
            role1_chars = role1.split(":")
            initiator_matches = i_df[i_df['initiator'].isin(role1_chars)]
            targets = initiator_matches['target'].unique()
            for env_node in env_nodes:
                if env_node in target_features:
                    role_relations.append((role1, env_node))
            targets_df =  ph_df[ph_df['role'].str.contains('|'.join(targets))]
            role_targets = targets_df['role'].unique()
            for role2 in role_targets:
                role_relations.append((role1, role2))
        weighted_relations = []
        for relation in role_relations:
            role1, role2 = relation
            role1_components = role1.split(":")
            role2_components = role2.split(":")
            num_interactions = len(
                i_df[
                    (i_df['initiator'].isin(role1_components)) &
                    (i_df['target'].isin(role2_components))
                ]
            )
            weighted_relations.append((role1, role2, num_interactions))
        G.add_weighted_edges_from(weighted_relations)
        self.set_node_postions(G, env_nodes, role_nodes)
        return G

    def export_role_networks_to_json(self) -> None:
        networks_data = {}
        i_df = get_interactions_df(self.db_path)
        fc_df = get_feature_changes_df(self.db_path)

        for wid, nid in self.world_dict.items():
            if nid not in networks_data:
                networks_data[nid] = {}
            ph_df = get_phenotypes_df(self.db_path, shadow=False, worlds=wid)
            interactions_subset = i_df[i_df['network_id'] == nid]
            features_subset = fc_df[
                (fc_df['world_id'] == wid) &
                (fc_df['env'] == 1)
            ]
            env_nodes = features_subset['name'].unique().tolist()
            role_network = self.construct_role_network(
                env_nodes,
                phenotypes_df,
                interactions_subset
            )
            network_json = nx.node_link_data(role_network)
            networks_data[nid][wid] = network_json

        output_file_path = os.path.join(self.db_loc, "role_network_data.json")
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
        pos = {}
        for node, data in G.nodes(data=True):
            node_color, node_shape = data['node_color'], data['node_shape']
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=[node],
                node_color=node_color,
                node_shape=node_shape,
                node_size=500
            )
        nx.draw_networkx_labels(G, pos, font_size=8)
        nx.draw_networkx_edges(G, pos, width=edge_widths, node_size=500)
        sm_pop = plt.cm.ScalarMappable(cmap=cmap, norm=normalize)
        sm_pop.set_array([])
        plt.colorbar(sm_pop, label='Population')
        plt.savefig(file_path)
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

def setup_logging(db_loc, db_name):
    process_id = os.getpid()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(db_loc, f"log_{db_name}_{process_id}_{timestamp}.log")

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

def test_analysis_function(db_loc, db_name):
    db_path = os.path.join(db_loc, db_name)
    file_path = os.path.join(db_loc, 'world_dict.json')
    try:
        world_dict = db.get_world_dict(db_path)
        print(world_dict)
    except Exception as e:
        print(e)
    with open(file_path, 'w') as json_file:
        json.dump(world_dict, json_file, indent=2)

def find_db_files(directory):
    dbs_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.db'):
                dbs_list.append((root, file))
    return dbs_list

def run_analysis_from_config(args):
    config, db_loc, db_name = args
    setup_logging(db_loc, db_name)
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
            method = getattr(instance, method_name)
            message_template = method_entry.get('message', "{class_name}.{method_name} for {db_name} in {db_loc}")
            logging.info(message_template.format(class_name=class_name, method=method_name, db_name=db_name, db_loc=db_loc))
            method()
            logging.info("Finished method: {method_name} for {class_name}".format(method_name=method_name, class_name=class_name))
    for function in functions:
        function_name = function['function_name']
        function_obj = globals()[function_name]
        message_template = function.get('message', "{function_name} for {db_name} in {db_loc}")
        loggin.info(message_template.format(function_name=function_name, db_name=db_name, db_loc=db_loc))
        function_obj(db_loc, db_name)
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
    with tqdm(total=len(dbs_list), desc="Databases Pool") as progress:
        with Pool(num_processes) as pool:
            for result in pool.imap_unordered(run_analysis_from_config, dbs_list):
                progress.update()

if __name__ == "__main__":
    main()