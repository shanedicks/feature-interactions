from itertools import chain, combinations
from typing import Dict, List, Tuple
import random

import matplotlib.pyplot as plt
import numpy as np
from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import MultiGrid
from mesa.time import RandomActivation

from agents import Agent, Role, Site
from output import *

Payoff = Tuple[float, float]
PayoffDict = Dict[str, Dict[str, Payoff]]


def name_from_number(num: int, lower: bool = True):
    if lower:
        char = 97
    else:
        char = 65
    letters = ''
    while num:
        mod = (num - 1) % 26
        letters += chr(mod + char)
        num = (num - 1) // 26
    return ''.join(reversed(letters))


class Feature:

    def __init__(
        self,
        model: "World",
        feature_id: int,
        env: bool,
        num_values: int = 5,
    ) -> None:
        self.id = feature_id
        self.model = model
        self.name = name_from_number(feature_id, lower=False)
        self.env = env
        self.values = []
        for i in range(num_values):
            self.values.append(self.new_value())

    def new_value(self):
        return name_from_number(len(self.values) + 1)

    def agent_feature_eu_dict(self, site):
        # Get the expected utility for each trait of Feature for an Agent at a given site
        lfd = site.get_local_features_dict()
        pop = len(self.model.grid.get_cell_list_contents(site.pos))
        in_edges = [
            x[2] for x in self.model.feature_interactions.in_edges(
                nbunch=self,
                data='interaction'
            )
        ]
        out_edges = [
            x[2] for x in self.model.feature_interactions.edges(
                nbunch=self,
                data='interaction'
            )
        ]
        scores = {}
        for v in self.values:
            eu = 0
            for edge in in_edges:
                weight = lfd[edge.initiator]['total'] / pop if pop > 0 else 0
                opp_dist = lfd[edge.initiator]['dist']
                eu += weight * sum([edge.payoffs[i][v][1]*opp_dist[i] for i in opp_dist])
            for edge in out_edges:
                if edge.target in site.traits.keys():
                    weight = edge.target.env_feature_weight(site)
                    eu += weight * edge.payoffs[v][site.traits[edge.target]][0]
                elif edge.target.env:
                    continue
                else:
                    weight = lfd[edge.target]['total'] / pop if pop > 0 else 0
                    opp_dist = lfd[edge.target]['dist']
                    eu += weight * sum([edge.payoffs[v][i][0]*opp_dist[i] for i in opp_dist])
            scores[v] = eu
        return scores

    def env_feature_weight(self, site):
        # Get the chance of initiating an interaction against Feature for an agent at a given site
        lfd = site.get_local_features_dict()
        lt = site.traits[self]
        edges = [
            x[2] for x in self.model.feature_interactions.in_edges(
                nbunch=self,
                data='interaction'
            )
        ]
        pop = sum([
            lfd[f]['total'] for f in lfd.keys()
            if f in [e.initiator for e in edges]
        ])
        avg_impact = 0
        for edge in edges:
            weight = lfd[edge.initiator]['total'] / pop if pop > 0 else 0
            opp_dist = lfd[edge.initiator]['dist']
            avg_impact += weight * sum([edge.payoffs[i][lt][1]*opp_dist[i] for i in opp_dist])
        if avg_impact >= 0:
            return 1
        else:
            num_ints = site.utils[self]/-avg_impact
            return min([num_ints/pop, 1])

    def __repr__(self) -> str:
        return self.name


class Interaction:

    def __init__(
        self,
        model: "World",
        initiator: Feature,
        target: Feature,
        trait_payoff_mod: float = 0.5,
        payoffs: PayoffDict = None
    ) -> None:
        self.model = model
        self.random = model.random
        self.initiator = initiator
        self.target = target
        self.trait_payoff_mod = trait_payoff_mod
        self.anchors = self.set_anchors()
        assert trait_payoff_mod <= 1.0
        if payoffs is None:
            self.payoffs = self.construct_payoffs()
        else:
            self.payoffs = payoffs

    def set_anchors(self):
        anchor = 1 - self.trait_payoff_mod
        i_anchor = round(self.random.uniform(-anchor, anchor), 2)
        t_anchor = round(self.random.uniform(-anchor, anchor), 2)
        return {"i": i_anchor, "t": t_anchor}

    def construct_payoffs(self) -> PayoffDict:
        payoffs = {}
        for i_value in self.initiator.values:
            payoffs[i_value] = {}
            for t_value in self.target.values:
                payoffs[i_value][t_value] = self.new_payoff(i_value, t_value)
        return payoffs

    def new_payoff(self, i_value, t_value):
        mod = self.trait_payoff_mod
        i = round(self.anchors["i"] + self.random.uniform(-mod, mod), 2)
        assert i <= 1.0 and i >= -1.0
        t = round(self.anchors["t"] + self.random.uniform(-mod, mod), 2)
        assert t <= 1.0 and t >= -1.0
        return (i, t)

    def __repr__(self) -> str:
        return "{0}→{1}".format(
            self.initiator.name, 
            self.target.name
        )


class World(Model):

    def __init__(
        self,
        feature_interactions: nx.digraph.DiGraph = None,
        init_env_features: int = 5,
        init_agent_features: int = 3,
        max_feature_interactions: int = 4,
        init_agents: int = 100,
        base_agent_utils: float = 0.0,
        base_env_utils: float = 10.0,
        total_pop_limit = 10000,
        pop_cost_exp = 2,
        grid_size: int = 2,
        repr_multi: int = 1,
        mortality: float = 0.02,
        move_chance: float = 0.01
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
        self.site_pop_limit = total_pop_limit / (grid_size ** 2)
        self.pop_cost_exp = pop_cost_exp
        self.grid = MultiGrid(grid_size, grid_size, True)
        self.schedule = RandomActivation(self)
        self.cached_payoffs = {}
        self.roles_dict = {}
        self.site_roles_dict = {}
        self.sites = {}
        self.born = 0
        self.died = 0
        self.moved = 0
        self.datacollector = DataCollector(
            model_reporters = {
                "Pop": get_population,
                "Total Utility": get_total_utility,
                "Mean Utility": get_mean_utility,
                "Med Utility": get_median_utility,
                "Phenotypes": get_num_phenotypes,
                "Features": get_num_agent_features,
                "Roles": get_num_roles,
                "Born": "born",
                "Died": "died",
                "Moved": "moved"
            },
            tables = {
                "Occupied Roles": ['Step', 'Site', 'Role', 'Pop'],
                "Other Roles": ['Step', 'Site', 'Viable', 'Adjacent', 'Occupiable']
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
            site = Site(model=self, pos=pos)
            self.sites[pos] = site
            self.site_roles_dict[pos] = {}
        t_list = list(range(1, init_agent_features+1))
        for i in range(init_agents):
            num_traits = self.random.choices(t_list, t_list[::-1])[0]
            agent = self.create_agent(num_traits = num_traits)
            self.schedule.add(agent)
            x = self.random.randrange(self.grid_size)
            y = self.random.randrange(self.grid_size)
            self.grid.place_agent(agent, (x, y))
        for agent in self.schedule.agents:
            agent.site = agent.get_site()
            agent.role = agent.get_role()
        print("Environment -------------------")
        print(env_report(self))
        print("Roles Distribution ------------")
        print(role_dist(self))
        print("Interaction Report ------------")
        interaction_report(self)

    def next_feature_id(self) -> int:
        self.current_feature_id += 1
        return self.current_feature_id

    def get_features_list(self, env: bool = False) -> List[Feature]:
        return [f for f in self.feature_interactions.nodes if f.env is env]

    def role_generator(self):
        f_list = self.get_features_list()
        feature_sets = chain.from_iterable(combinations(f_list, n) for n in range(len(f_list) + 1))
        for feature_set in feature_sets:
            yield feature_set

    def evaluate_rolesets(self):
        dc = self.datacollector
        srd = self.site_roles_dict
        for site in self.sites.values():
            site.pop_cost = site.get_pop_cost()
            site.fud = site.get_feature_utility_dict()
            sd = srd[site.pos]
            sd['occupied'] = get_occupied(site)
            sd['possible'] = get_num_possible(self)
            sd['viable'] = 0
            sd['adjacent'] = set()
            sd['occupiable'] = set()
        for role in self.role_generator():
            for site in self.sites.values():
                sd = srd[site.pos]
                v = check_viable(site, role)
                a = check_adjacent(site, role)
                o = a and v
                if o:
                    sd['viable'] += 1
                    sd['adjacent'].add(role)
                    sd['occupiable'].add(role)
                elif v:
                    sd['viable'] += 1
                elif a:
                    sd['adjacent'].add(role)
        for site in self.sites.values():
            keys = ['Step', 'Site', 'Viable', 'Adjacent', 'Occupiable']
            site.update_role_network()
            sd = srd[site.pos]
            values = [
                self.schedule.time,
                site.pos, 
                sd['viable'],
                len(sd['adjacent']),
                len(sd['occupiable']),
            ]
            dc.add_table_row('Other Roles', {k:v for k,v in zip(keys, values)})
            keys = ['Step', 'Site', 'Role', 'Pop']
            for role, pop in sd['occupied'].items():
                values = [self.schedule.time, site.pos, role.rolename, pop]
                dc.add_table_row('Occupied Roles', {k:v for k,v in zip(keys, values)})

    def create_interaction(
        self,
        initiator: Feature,
    ) -> None:
        extant_targets = list(self.feature_interactions.neighbors(initiator))
        target_choices = [
            x for x in self.feature_interactions.nodes
            if x not in extant_targets
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

    def create_trait(
            self,
            feature: Feature
    ) -> str:
        value = feature.new_value()
        feature.values.append(value)
        initated = [
            x[2] for x
            in self.feature_interactions.edges(
                nbunch=feature,
                data='interaction'
            )
        ]
        targeted = [
            x[2] for x
            in self.feature_interactions.in_edges(
                nbunch=feature,
                data='interaction'
            )
        ]
        for i in initated:
            i.payoffs[value] = {}
            for t_value in i.target.values:
                i.payoffs[value][t_value] = i.new_payoff(value, t_value)
        for t in targeted:
            for i_value in t.initiator.values:
                t.payoffs[i_value][value] = t.new_payoff(i_value, value)
        print("New trait {0} added to feature {1}".format(value, feature))
        return value


    def create_feature(self,
        env: bool = False
    ) -> Feature:
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

    def create_agent(
        self,
        num_traits: int,
        utils:float = None,
    ) -> Agent:
        if utils is None:
            utils = self.base_agent_utils
        traits = {}
        agent_features = self.get_features_list()
        features = self.random.sample(agent_features, num_traits)
        for feature in features:
            value = self.random.choice(feature.values)
            traits[feature] = value
        agent = Agent(
            unique_id = self.next_id(),
            model = self,
            utils = utils,
            traits = traits,
        )
        return agent

    def step(self):
        self.new = 0
        self.cached = 0
        self.born = 0
        self.died = 0
        self.moved = 0
        for site in self.sites.values():
            site.reset()
        self.schedule.step()
        print(
            "Step{0}: {1} agents, {2} roles, {3} types, {4} new and {5} cached interactions".format(
                self.schedule.time,
                self.schedule.get_agent_count(),
                len(set(occupied_roles_list(self))),
                len(set(occupied_phenotypes_list(self))),
                self.new,
                self.cached
            )
        )
        print("{0} born and {1} died".format(self.born, self.died))
        self.evaluate_rolesets()
        env_report(self)
        self.datacollector.collect(self)
        print(role_dist(self))
