
from typing import Dict, List, Tuple, Iterator
from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from agents import Agent, Site
from features import Feature, Interaction
from output import *


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
        self.datacollector = self.model.datacollector
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

    def step(self):
        for pos, site in self.sites.items():
            live_site = self.model.sites[pos]
            born, died = live_site.born, live_site.died
            for agent in site.shuffled_sample(born):
                agent.reproduce()
            for agent in site.shuffled_sample(died):
                agent.die()

class World(Model):

    def __init__(
        self,
        feature_interactions: nx.digraph.DiGraph = None,
        init_env_features: int = 5,
        init_agent_features: int = 3,
        max_feature_interactions: int = 5,
        init_agents: int = 50,
        base_agent_utils: float = 0.0,
        base_env_utils: float = 50.0,
        total_pop_limit = 6000,
        pop_cost_exp = 2,
        grid_size: int = 3,
        repr_multi: int = 1,
        mortality: float = 0.02,
        move_chance: float = 0.01,
        trait_mutate_chance: float = 0.01,
        trait_create_chance: float = 0.001,
        feature_mutate_chance: float = 0.001,
        feature_create_chance: float = 0.001,
        feature_gain_chance: float = 0.5,
        snap_interval: int = 50,
        feature_timeout: int = 10,
        trait_timeout: int = 10,
    ) -> None:
        super().__init__()
        self.feature_interactions = feature_interactions
        self.base_agent_utils = base_agent_utils
        self.base_env_utils = base_env_utils
        self.max_feature_interactions = min(
            max_feature_interactions,
            (init_env_features + init_agent_features)
        )
        self.grid_size = grid_size
        self.repr_multi = repr_multi
        self.mortality = mortality
        self.move_chance = move_chance
        self.trait_mutate_chance = trait_mutate_chance
        self.feature_mutate_chance = feature_mutate_chance
        self.trait_create_chance = trait_create_chance
        self.feature_create_chance = feature_create_chance
        self.feature_gain_chance = feature_gain_chance
        self.site_pop_limit = total_pop_limit / (grid_size ** 2)
        self.pop_cost_exp = pop_cost_exp
        self.grid = MultiGrid(grid_size, grid_size, True)
        self.schedule = RandomActivation(self)
        self.cached_payoffs = {}
        self.roles_dict = {}
        self.site_roles_dict = {}
        self.sites = {}
        self.snap_interval = snap_interval
        self.feature_timeout = feature_timeout
        self.trait_timeout = trait_timeout
        self.datacollector = DataCollector(
            model_reporters = {
                "Pop": get_population,
                "Total Utility": get_total_utility,
                "Mean Utility": get_mean_utility,
                "Med Utility": get_median_utility,
                "Phenotypes": get_num_phenotypes,
                "Features": get_num_agent_features,
                "Roles": get_num_roles,
            },
            tables = {
                "Roles": ['Step', 'Site', 'Shadow', 'Role', 'Pop'],
                "Phenotypes": ['Step', 'Site', 'Shadow', 'Phenotype', 'Pop'],
                "Rolesets": ['Step', 'Site', 'Viable', 'Adjacent', 'Occupiable', "Occupied"],
                "Traits": ['Step', 'Site', 'Shadow', 'Feature', 'Trait', 'Count'],
                "Sites": ['Step', 'Site', 'Born', 'Died', 'Moved In', 'Moved Out']
            }
        )
        if self.feature_interactions is None:
            self.current_feature_id = 0
            self.feature_interactions = nx.DiGraph()
            for i in range(init_env_features):
                self.create_feature(env = True)
            for i in range(init_agent_features):
                self.create_feature()
        else:
            self.current_feature_id = len(self.feature_interactions.number_of_nodes())
        self.roles_network = nx.DiGraph()
        self.roles_network.add_nodes_from(self.get_features_list(env=True))
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
        print("Environment -------------------")
        env_report(self)
        print("Roles Distribution ------------")
        print(role_dist(self))
        print("Interaction Report ------------")
        interaction_report(self)
        draw_feature_interactions(self)

    def next_feature_id(self) -> int:
        self.current_feature_id += 1
        return self.current_feature_id

    def get_features_list(self, env: bool = False) -> List[Feature]:
        return [f for f in self.feature_interactions.nodes if f.env is env]

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
        return

    def create_feature(self, env: bool = False) -> Feature:
        feature_id = self.next_feature_id()
        feature = Feature(
            feature_id = feature_id,
            model = self,
            env = env
        )
        self.feature_interactions.add_node(feature)
        if feature.env is False:
            num_ints = self.random.randrange(1, self.max_feature_interactions)
            for i in range(num_ints):
                self.create_interaction(feature)
        print('New feature ', feature)
        return feature

    def remove_feature(self, feature: Feature) -> None:
        print("Removing feature ", feature)
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
            pl = [p for s in sd for p in role.phenotypes[s].keys()]
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
        for site in self.sites.values():
            site.reset()
        for site in self.shadow.sites.values():
            site.reset()
        self.schedule.step()
        self.shadow.step()
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
        if self.schedule.time % self.snap_interval == 0:
            evaluate_rolesets(self)
        self.datacollector.collect(self)
        tables_update(self)
        env_report(self)
        print(role_dist(self))
