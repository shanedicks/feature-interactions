import itertools
import json
import os
import shutil
import sys
from datetime import datetime
from functools import partial, reduce
from multiprocessing import Pool
from typing import Any, Dict, Iterable, List, Mapping, Union
from tqdm import tqdm
from model import World
from database import Manager

def get_param_set_list(
    params_dict: Mapping[str, Union[Any, Iterable[Any]]],
) -> List[Dict[str, Any]]:
    params_list = []
    for param, values in params_dict.items():
        if isinstance(values, str):
            all_values = [(param, values)]
        else:
            try:
                all_values = [(param, value) for value in values]
            except TypeError:
                all_values = [(param, values)]
        params_list.append(all_values)
    params_set = itertools.product(*params_list)
    return [dict(params) for params in params_set]

class Controller():

    def __init__(
        self,
        experiment_name: str,
        path_to_db: str,
        db_interval = None
    ) -> None:
        self.experiment_name = experiment_name
        self.db_manager = self.get_db_manager(path_to_db)
        self.db_interval = 1 if not db_interval else db_interval
        self.default_network_params = {
            "init_env_features": 5,
            "init_agent_features": 3,
            "max_feature_interactions": 5,
            "trait_payoff_mod": 0.5,
            "anchor_bias": 0.0,
            "payoff_bias": 0.0
        }
        self.default_world_params = {
            "trait_mutate_chance": 0.01,
            "trait_create_chance": 0.001,
            "feature_mutate_chance": 0.001,
            "feature_create_chance": 0.001,
            "feature_gain_chance": 0.5,
            "feature_timeout":  100,
            "trait_timeout":  100,
            "init_agents":  900,
            "base_agent_utils": 0.0,
            "base_env_utils": 100.0,
            "total_pop_limit": 6000,
            "pop_cost_exp": 2,
            "feature_cost_exp": .75,
            "grid_size":  3,
            "repr_multi":  1,
            "mortality": 0.01,
            "move_chance": 0.01,
            "snap_interval": 500,
            "target_sample": 1,
        }

    def get_db_manager(self, path_to_db) -> "Manager":
        db_name = f"{self.experiment_name}.db"
        return Manager(path_to_db, db_name)

    def run(
        self,
        num_networks: int,
        num_iterations: int,
        max_steps: int,
        network_params_dict: Mapping[str, Union[Any, Iterable[Any]]] = None,
        world_params_dict: Mapping[str, Union[Any, Iterable[Any]]] = None,
    ) -> None:
        self.db_manager.initialize_db()
        if network_params_dict is None:
            network_params_dict = self.default_network_params
        if world_params_dict is None:
            world_params_dict = self.default_world_params
        self.total_networks = num_networks * reduce(
            lambda x, y: x * y,
            [len(v) for v in network_params_dict.values() if isinstance(v, list)],
            1
        )
        self.network_worlds = num_iterations * reduce(
            lambda x, y: x * y,
            [len(v) for v in world_params_dict.values() if isinstance(v, list)],
            1
        )
        self.max_steps = max_steps
        total = self.total_networks * self.network_worlds
        orig_stdout = sys.stdout
        filename = self.db_manager.db_string.replace(".db", ".txt")
        progress = tqdm(total=total)
        with open(filename, 'w') as f:
            sys.stdout = f
            network_paramset = get_param_set_list(network_params_dict)
            for network_params in network_paramset:
                network_row = (
                    network_params['init_env_features'],
                    network_params['init_agent_features'],
                    network_params['max_feature_interactions'],
                    network_params['trait_payoff_mod'],
                    network_params['anchor_bias'],
                    network_params['payoff_bias']
                )
                for network in range(num_networks):
                    network_id = self.db_manager.write_row('networks', network_row)
                    world_param_set = get_param_set_list(world_params_dict)
                    self.world_num = 0
                    for world_params in world_param_set:
                        for i in range(num_iterations):
                            self.world_num += 1
                            world_row = (
                                world_params['trait_mutate_chance'],
                                world_params['trait_create_chance'],
                                world_params['feature_mutate_chance'],
                                world_params['feature_create_chance'],
                                world_params['feature_gain_chance'],
                                world_params['feature_timeout'],
                                world_params['trait_timeout'],
                                world_params['init_agents'],
                                world_params['base_agent_utils'],
                                world_params['base_env_utils'],
                                world_params['total_pop_limit'],
                                world_params['pop_cost_exp'],
                                world_params['feature_cost_exp'],
                                world_params['grid_size'],
                                world_params['repr_multi'],
                                world_params['mortality'],
                                world_params['move_chance'],
                                world_params['snap_interval'],
                                world_params['target_sample'],
                                network_id
                            )
                            world_id = self.db_manager.write_row("worlds", world_row)
                            world = World(self, world_id, network_id,**network_params, **world_params)
                            self.world = world
                            while world.running and world.schedule.time < self.max_steps:
                                world.step()
                            progress.update()
        sys.stdout = orig_stdout

    def run_mp(
        self,
        num_networks: int,
        num_iterations: int,
        max_steps: int,
        num_processes,
        network_params_dict: Mapping[str, Union[Any, Iterable[Any]]] = None,
        world_params_dict: Mapping[str, Union[Any, Iterable[Any]]] = None,
    ) -> None:
        self.db_manager.initialize_db()
        if network_params_dict is None:
            network_params_dict = self.default_network_params
        if world_params_dict is None:
            world_params_dict = self.default_world_params
        self.total_networks = num_networks * reduce(
            lambda x, y: x * y,
            [len(v) for v in network_params_dict.values() if isinstance(v, list)],
            1
        )
        self.max_steps = max_steps
        func = partial(
            self.run_network,
            num_iterations=num_iterations,
            world_params_dict=world_params_dict,
        )
        networks = []
        network_paramset = get_param_set_list(network_params_dict)
        for network_params in network_paramset:
            network_row = (
                network_params['init_env_features'],
                network_params['init_agent_features'],
                network_params['max_feature_interactions'],
                network_params['trait_payoff_mod'],
                network_params['anchor_bias'],
                network_params['payoff_bias']
            )
            for i in range(num_networks):
                network_id = self.db_manager.write_row('networks', network_row)
                networks.append((network_id, network_params))
        with tqdm(total=len(networks), desc="Network Pool") as progress:
            with Pool(num_processes) as p:
                for n in p.imap_unordered(func, networks):
                    progress.update()

    def run_network(
        self,
        network,
        num_iterations,
        world_params_dict,
    ) -> None:
        network_id, network_params = network
        world_params_list = get_param_set_list(world_params_dict)
        self.network_worlds = num_iterations * reduce(
            lambda x, y: x * y,
            [len(v) for v in world_params_dict.values() if isinstance(v, list)],
            1
        )
        orig_stdout = sys.stdout
        outfile = self.db_manager.db_string.replace(".db", f"_{network_id}.txt")
        errfile = self.db_manager.db_string.replace(".db", f"_{network_id}_err.txt")
        self.world_num = 0
        with open(outfile, 'w') as out:
            sys.stdout = out
            print(f"Network {network_id}\n{network_params}")
            for world_params in world_params_list:
                for i in range(num_iterations):
                    self.world_num +=1
                    world_row = (
                        world_params['trait_mutate_chance'],
                        world_params['trait_create_chance'],
                        world_params['feature_mutate_chance'],
                        world_params['feature_create_chance'],
                        world_params['feature_gain_chance'],
                        world_params['feature_timeout'],
                        world_params['trait_timeout'],
                        world_params['init_agents'],
                        world_params['base_agent_utils'],
                        world_params['base_env_utils'],
                        world_params['total_pop_limit'],
                        world_params['pop_cost_exp'],
                        world_params['feature_cost_exp'],
                        world_params['grid_size'],
                        world_params['repr_multi'],
                        world_params['mortality'],
                        world_params['move_chance'],
                        world_params['snap_interval'],
                        world_params['target_sample'],
                        network_id
                    )
                    world_id = self.db_manager.write_row("worlds", world_row)
                    world = World(self, world_id, network_id,**network_params, **world_params)
                    while world.running and world.schedule.time < self.max_steps:
                        world.step()
            sys.stdout = orig_stdout
        return

def main():
    with open(sys.argv[1], 'r') as f:
        data = f.read()
    obj = json.loads(data)
    timestamp = datetime.today().strftime("%Y%m%d_%H%M%S")
    title = obj['title']
    experiment_name = f"{title}_{timestamp}"
    output_path = obj['output_path']
    path_to_db = f"{output_path}/{experiment_name}"
    os.mkdir(path_to_db)
    shutil.copy2(sys.argv[1], path_to_db)
    db_interval = obj.get('db_interval')
    controller = Controller(
        experiment_name, 
        path_to_db,
        db_interval
    )
    num_processes = os.environ.get(
        'SLURM_CPUS_PER_TASK',
        obj.get('num_processes')
    )
    if num_processes == 1:
        controller.run(
            obj['num_networks'],
            obj['num_iterations'],
            obj['max_steps'],
            network_params_dict=obj['network_params'],
            world_params_dict=obj['world_params']
        )
    else:
        num_processes = int(num_processes) if num_processes else None
        controller.run_mp(
            obj['num_networks'],
            obj['num_iterations'],
            obj['max_steps'],
            num_processes = num_processes,
            network_params_dict=obj['network_params'],
            world_params_dict=obj['world_params']
        )

if __name__ == "__main__":
    main()
