from typing import Dict, List, Tuple
import random

import matplotlib.pyplot as plt
from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import MultiGrid
from mesa.time import RandomActivation

from agents import Agent, Site
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
        feature_id: int,
        env: bool,
        num_values: int = 5
    ) -> None:
        self.name = name_from_number(feature_id, lower=False)
        self.env = env
        self.values = []
        for i in range(num_values):
            self.values.append(self.new_value())

    def new_value(self):
        return name_from_number(len(self.values) + 1)

    def __repr__(self) -> str:
        return self.name


class Interaction:

    def __init__(
        self,
        model: "World",
        initiator: Feature,
        target: Feature,
        trait_utility_random: bool = False,
        trait_payoff_mod: float = 0.25,
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
            self.payoffs = self.construct_payoffs(random=trait_utility_random)
        else:
            self.payoffs = payoffs

    def set_anchors(self):
        anchor = 1 - self.trait_payoff_mod
        i_anchor = round(self.random.uniform(-anchor, anchor), 2)
        t_anchor = round(self.random.uniform(-anchor, anchor), 2)
        return {"i": i_anchor, "t": t_anchor}

    def construct_payoffs(self, random:bool) -> PayoffDict:
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
        init_env_features: int = 3,
        init_agent_features: int = 3,
        max_feature_interactions: int = 4,
        init_agents: int = 10,
        base_agent_utils: float = 0.0,
        base_env_utils: float = 2.0,
        grid_size: int = 1,
        snap_interval: int = 20
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
        self.snap_interval = snap_interval
        self.grid = MultiGrid(grid_size, grid_size, True)
        self.schedule = RandomActivation(self)
        self.datacollector = DataCollector(
            model_reporters = {
                "Pop": get_population,
                "Total Utility": get_total_utility,
                "Avg Utility": get_avg_utility
            },
            agent_reporters = {
                "Utils": lambda agent: agent.utils,
                "Age": lambda agent: agent.age,
                "Traits": lambda agent: len(agent.traits)
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
        self.sites = {}
        for _, x, y in self.grid.coord_iter():
            pos = (x,y)
            site = Site(model=self, pos=pos)
            self.sites[pos] = site
        for i in range(init_agents):
            num_traits = self.random.randrange(1, init_agent_features + 1)
            agent = self.create_agent(num_traits = num_traits)
            self.schedule.add(agent)
            x = self.random.randrange(self.grid_size)
            y = self.random.randrange(self.grid_size)
            self.grid.place_agent(agent, (x, y))

    def next_feature_id(self) -> int:
        self.current_feature_id += 1
        return self.current_feature_id

    def get_features_list(self, env: bool = False) -> List[Feature]:
        return [f for f in self.feature_interactions.nodes if f.env is env]

    def create_interaction(
        self,
        initiator: Feature,
    ) -> None:
        extant_targets = list(self.feature_interactions.neighbors(initiator))
        target_choices = [
            x for x
            in self.feature_interactions.nodes
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
                interaction.initiator,
                interaction.target,
                interaction = interaction
            )
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
            env = env
        )
        self.feature_interactions.add_node(feature)
        if feature.env is False:
            num_ints = self.random.randrange(1, self.max_feature_interactions)
            for i in range(num_ints):
                self.create_interaction(feature)
        return feature

    def create_agent(
        self,
        num_traits: int,
        utils:float = None,
    ) -> Agent:
        if utils is None:
            utils = self.base_agent_utils
        traits = {}
        agent_features = self.get_features_list(env=False)
        features = self.random.sample(agent_features, num_traits)
        for feature in features:
            value = self.random.choice(feature.values)
            traits[feature] = value
        agent = Agent(
            unique_id = self.next_id(),
            model = self,
            utils = utils,
            traits = traits
        )
        return agent

    def step(self):
        for site in self.sites.values():
            site.reset()
        self.schedule.step()
        print(
            "Step{0}: {1} agents, {2} roles, {3} types".format(
                self.schedule.time,
                len(self.schedule.agents),
                len(set(occupied_roles_list(self))),
                len(set(occupied_trait_set_list(self))),
            )
        )
        self.datacollector.collect(self)



