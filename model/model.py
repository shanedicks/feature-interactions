import itertools
from datetime import datetime
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Tuple, Type, Union
from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from agents import Agent, Site
from database import Manager
from features import Feature, Interaction
from output import *


class Controller():

    def __init__(
        self,
        experiment_name: str,
    ) -> None:
        self.experiment_name = experiment_name
        self.db_name = self.get_db_name()
        self.db_manager = self.get_db_manager()
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
            "feature_timeout":  50,
            "trait_timeout":  50,
            "init_agents":  100,
            "base_agent_utils": 0.0,
            "base_env_utils": 100.0,
            "total_pop_limit": 6000,
            "pop_cost_exp": 2,
            "feature_cost_exp": .75,
            "grid_size":  3,
            "repr_multi":  1,
            "mortality": 0.02,
            "move_chance": 0.01,
            "snap_interval": 50,
            "target_sample": 1,
        }

    def get_db_name(self) -> str:
        timestamp = datetime.today().strftime("%Y%m%d_%H%M")
        db_name = "{0}_{1}.db".format(self.experiment_name, timestamp)
        return db_name

    def get_db_manager(self) -> "Manager":
        path_to_db = "../data/"
        db_name = self.db_name
        manager = Manager(path_to_db, db_name)
        return manager

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
        network_paramset = self.param_set_generator(network_params_dict)
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
                world_param_set = self.param_set_generator(world_params_dict)
                for world_params in world_param_set:
                    for i in range(num_iterations):
                        self.last_feature_id, self.last_trait_id = 0, 0
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
                        while world.running and world.schedule.time < max_steps:
                            world.step()

    def param_set_generator(
        self,
        params_dict: Mapping[str, Union[Any, Iterable[Any]]],
    ) -> Iterator[Dict[str, Any]]:
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
        for params in params_set:
            yield dict(params)



class Shadow(Model):

    def __init__(
        self,
        model: "World"
    ) -> None:
        super().__init__()
        self.model = model
        self.random = model.random
        self.feature_interactions = self.model.feature_interactions
        self.base_agent_utils = self.model.base_agent_utils
        self.base_env_utils = self.model.base_env_utils
        self.repr_multi = self.model.repr_multi
        self.mortality = self.model.mortality
        self.move_chance = self.model.move_chance
        self.trait_mutate_chance = self.model.trait_mutate_chance
        self.feature_mutate_chance = self.model.feature_mutate_chance
        self.trait_create_chance = self.model.trait_create_chance
        self.feature_create_chance = self.model.feature_create_chance
        self.feature_gain_chance = self.model.feature_gain_chance
        self.site_pop_limit = self.model.site_pop_limit
        self.pop_cost_exp = self.model.pop_cost_exp
        self.grid = MultiGrid(self.model.grid_size, self.model.grid_size, True)
        self.sites = {}
        self.roles_dict = {}
        for pos, site in self.model.sites.items():
            new_site = Site(
                    model = self,
                    pos = pos,
                    traits = site.traits
                )
            self.sites[pos] = new_site
            for agent in site.agents():
                new_agent = Agent(
                    unique_id = self.next_id(),
                    model = self,
                    shadow = True,
                    utils = agent.utils,
                    traits = agent.traits 
                )
                self.grid.place_agent(new_agent, pos)
                new_agent.site = new_agent.get_site()

    def get_features_list(self, env: bool = False):
        return self.model.get_features_list(env=env)

    def agents(self) -> Iterator['Agent']:
        return self.grid.iter_cell_list_contents([pos for pos in self.sites])

class World(Model):

    def __init__(
        self,
        controller: "Controller",
        world_id,
        network_id,
        # features network variables
        init_env_features: int,
        init_agent_features: int,
        max_feature_interactions: int,
        trait_payoff_mod: float,
        anchor_bias: float,
        payoff_bias: float,
        # mutation variables
        trait_mutate_chance: float,
        trait_create_chance: float,
        feature_mutate_chance: float,
        feature_create_chance: float,
        feature_gain_chance: float,
        feature_timeout: int,
        trait_timeout: int,
        #other variables
        init_agents: int,
        base_agent_utils: float,
        base_env_utils: float,
        total_pop_limit: int,
        pop_cost_exp: int,
        feature_cost_exp: float,
        grid_size: int,
        repr_multi: int,
        mortality: float,
        move_chance: float,
        snap_interval: int,
        target_sample: int,
    ) -> None:
        super().__init__()
        assert trait_payoff_mod <= 1 and trait_payoff_mod >= 0
        assert anchor_bias <= 1 and anchor_bias >= -1
        self.controller = controller
        self.db = controller.db_manager
        self.world_id = world_id
        self.network_id = network_id
        self.base_agent_utils = base_agent_utils
        self.base_env_utils = base_env_utils
        self.max_feature_interactions = max_feature_interactions
        self.trait_payoff_mod = trait_payoff_mod
        self.anchor_bias = anchor_bias
        self.payoff_bias = payoff_bias
        self.trait_mutate_chance = trait_mutate_chance
        self.feature_mutate_chance = feature_mutate_chance
        self.trait_create_chance = trait_create_chance
        self.feature_create_chance = feature_create_chance
        self.feature_gain_chance = feature_gain_chance
        self.grid_size = grid_size
        self.repr_multi = repr_multi
        self.mortality = mortality
        self.move_chance = move_chance
        self.site_pop_limit = total_pop_limit / (grid_size ** 2)
        self.pop_cost_exp = pop_cost_exp
        self.feature_cost_exp = feature_cost_exp
        self.grid = MultiGrid(grid_size, grid_size, True)
        self.schedule = RandomActivation(self)
        self.cached_payoffs = {}
        self.roles_dict = {}
        self.site_roles_dict = {}
        self.sites = {}
        self.feature_timeout = feature_timeout
        self.trait_timeout = trait_timeout
        self.target_sample = target_sample
        self.spacetime_dict = {}
        row = (self.world_id, self.schedule.time, "world")
        self.spacetime_dict["world"] = self.db.write_row("spacetime", row)
        self.get_or_create_init_features_network(
            init_env_features,
            init_agent_features
        )
        for _, x, y in self.grid.coord_iter():
            pos = (x,y)
            site = Site(model = self, pos = pos)
            self.sites[pos] = site
            self.site_roles_dict[pos] = {}
        t_list = list(range(1, init_agent_features+1))
        for i in range(init_agents):
            num_traits = self.random.choices(t_list, t_list[::-1])[0]
            agent = self.create_agent(num_traits = num_traits)
            self.schedule.add(agent)
            self.grid.place_agent(agent, agent.pos)
        for agent in self.schedule.agents:
            agent.site = agent.get_site()
        self.shadow = Shadow(model=self)
        self.running = True
        print("Environment -------------------")
        env_report(self)
        print("Roles Distribution ------------")
        print(role_dist(self))
        print("Interaction Report ------------")
        interaction_report(self)

    def next_feature_id(self) -> int:
        self.current_feature_id += 1
        return self.current_feature_id

    def get_features_list(self, env: bool = False) -> List[Feature]:
        return [f for f in self.feature_interactions.nodes if f.env is env]

    def next_feature(self, env: bool = False) -> Feature:
        restored = self.db.get_next_feature(
            self.network_id,
            self.current_feature_db_id
        )
        if restored:
            feature = self.restore_feature(restored)
        else:
            feature = self.create_feature(env=env)
        feature_changes_row = (
            self.spacetime_dict["world"],
            feature.db_id,
            "added"
        )
        self.db.write_row('feature_changes', feature_changes_row)
        return feature

    def get_or_create_init_features_network(
        self,
        num_env: int,
        num_agent:int
    ) -> None:
        self.current_feature_id = 0
        self.current_feature_db_id = 0
        self.feature_interactions = nx.DiGraph()
        for i in range(num_env):
            self.next_feature(env = True)
        for i in range(num_agent):
            self.next_feature()

    def restore_feature(self, feature_dict: Dict[str, Any]):
        feature_db_id = feature_dict['feature_id']
        self.current_feature_db_id = feature_db_id
        feature_id = self.next_feature_id()
        feature = Feature(
            feature_id = feature_id,
            db_id = feature_db_id,
            model = self,
            env = feature_dict['env']
        )
        self.feature_interactions.add_node(feature)
        print("restored feature {0}".format(feature))
        self.restore_interactions(feature)
        return feature

    def restore_interactions(self, initiator: Feature) -> None:
        interactions = self.db.get_feature_interactions(initiator.db_id)
        for i in interactions:
            target = get_feature_by_name(self, i['target'])
            if target:
                interaction = Interaction(
                    model = self,
                    initiator = initiator,
                    target = target,
                    db_id = i['db_id'],
                    restored = True,
                    anchors = {"i": i['i_anchor'], "t": i['t_anchor']}
                )
                self.feature_interactions.add_edge(
                    initiator, target, interaction = interaction
                )
                print("restored interaction {0}".format(interaction))
                affected_roles = [
                    role for role in self.roles_dict.values()
                    if any(f in role.features for f in [initiator, target])
                ]
                for role in affected_roles:
                    role.interactions = role.get_interactions()

    def create_interaction(self, initiator: Feature) -> None:
        extant_targets = list(self.feature_interactions.neighbors(initiator))
        target_choices = [
            x for x in self.feature_interactions.nodes if x not in extant_targets
        ]
        if len(target_choices) > 0:
            target = self.random.choice(target_choices)
            interaction = Interaction(
                model = self,
                initiator = initiator,
                target = target,
            )
            self.feature_interactions.add_edge(
                initiator, target, interaction = interaction
            )
            affected_roles = [
                role for role in self.roles_dict.values()
                if any(f in role.features for f in [initiator, target])
            ]
            for role in affected_roles:
                role.interactions = role.get_interactions()

    def create_feature(self, env: bool = False) -> Feature:
        feature_id = self.next_feature_id()
        feature = Feature(
            feature_id = feature_id,
            model = self,
            env = env
        )
        self.feature_interactions.add_node(feature)
        if feature.env is False:
            num_features = self.feature_interactions.number_of_nodes()
            max_ints = min(self.max_feature_interactions, num_features)
            num_ints = self.random.randrange(1, max_ints)
            for i in range(num_ints):
                self.create_interaction(feature)
        print('New feature ', feature)
        return feature

    def remove_feature(self, feature: Feature) -> None:
        print("Removing feature", feature)
        feature_changes_row = (
            self.spacetime_dict["world"],
            feature.db_id,
            "removed"
        )
        self.db.write_row('feature_changes', feature_changes_row)
        in_edges = feature.in_edges()
        out_edges = feature.out_edges()
        self.feature_interactions.remove_node(feature)
        affected_features = [x.initiator for x in in_edges]
        affected_features.extend([x.target for x in out_edges])
        affected_roles = [
            role for role in self.roles_dict.values()
            if any(f in role.features for f in affected_features)
        ]
        roles_to_remove = [
            role for role in self.roles_dict.values()
            if feature in role.features
        ]
        sd = self.sites
        cache = self.cached_payoffs
        for role in roles_to_remove:
            pl = [p for s in sd for p in role.types[s].keys()]
            for phenotype in pl:
                keys = [k for k, v in cache.items() if v == phenotype]
                if phenotype in cache:
                    del cache[phenotype]
                for k in keys:
                    del cache[k][phenotype]
            del self.roles_dict[role.features]
        for role in affected_roles:
            role.update()

    def prune_features(self):
        for feature in self.get_features_list():
            feature.check_empty()
            feature.prune_traits()
        pruneable = [
            f for f in self.get_features_list()
            if f.empty_steps >= self.feature_timeout
        ]
        for feature in pruneable:
            self.remove_feature(feature)

    def create_agent(self, num_traits: int, utils:float = None) -> Agent:
        if utils is None:
            utils = self.base_agent_utils
        traits = {}
        agent_features = self.get_features_list()
        features = self.random.sample(agent_features, num_traits)
        for feature in features:
            value = self.random.choice(feature.values)
            traits[feature] = value
        x = self.random.randrange(self.grid_size)
        y = self.random.randrange(self.grid_size)
        agent = Agent(
            unique_id = self.next_id(),
            model = self,
            utils = utils,
            traits = traits,
        )
        agent.pos = (x,y)
        return agent

    def verify_shadow(self):
        shadow = self.shadow
        for pos in shadow.sites:
            shadow_site = shadow.sites[pos]
            live = self.sites[pos]
            assert shadow_site.died == live.died
            assert shadow_site.moved_in == live.moved_in
            assert shadow_site.moved_out == live.moved_out
            assert shadow_site.born == live.born

    def step(self):
        self.new = 0
        self.cached = 0
        step = self.schedule.time
        row_dict = {"spacetime": []}
        if step > 0:
            row = (self.world_id, step, "world")
            row_dict['spacetime'].append(row)
        for pos, site in self.sites.items():
            row = (self.world_id, step, str(pos))
            row_dict['spacetime'].append(row)
            site.reset()
        self.db.write_rows(row_dict)
        spacetimes = self.db.get_spacetimes(self.world_id, step)
        for s in spacetimes:
            if s['pos'] == "world":
                self.spacetime_dict["world"] = s["id"]
            else:
                self.spacetime_dict[eval(s['pos'])] = s["id"]
        for site in self.shadow.sites.values():
            site.reset()
        self.schedule.step()
        self.verify_shadow()
        self.prune_features()
        print(
            "Step{0}: {1} agents, {2} roles, {3} types, "
            "{6} features | {4} new and {5} cached interactions".format(
                self.schedule.time,
                self.schedule.get_agent_count(),
                len(set(occupied_roles_list(self))),
                len(set(occupied_phenotypes_list(self))),
                self.new,
                self.cached,
                get_num_agent_features(self)
            )
        )
        tables_update(self)
        env_report(self)
        print(role_dist(self))
        if self.schedule.get_agent_count == 0:
            self.running = False
