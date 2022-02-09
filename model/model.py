from typing import Dict, List, Tuple
import random
import networkx as nx
import matplotlib.pyplot as plt
from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import MultiGrid
from mesa.time import RandomActivation

from agents import Agent, Feature, Interaction


class World(Model):

    def __init__(
        self,
        feature_interactions: nx.digraph.DiGraph = None,
        num_env_features: int = 2,
        num_agent_features: int = 4,
        num_feature_interactions: int = 5,
        initial_agents: int = 100,
        num_agent_traits: int = 2,
        base_utils: float = 0.0
    ) -> None:
        super().__init__()
        self.feature_interactions = feature_interactions
        self.base_utils = base_utils
        self.num_agent_traits = num_agent_traits
        self.num_feature_interactions = num_feature_interactions
        self.schedule = RandomActivation(self)
        self.datacollecter = DataCollector()
        if self.feature_interactions is None:
            self.current_feature_id = 0
            self.feature_interactions = nx.DiGraph()
            for i in range(num_env_features):
                self.create_feature(env = True)
            for i in range(num_agent_features):
                self.create_feature()
        else:
            self.current_feature_id = len(self.feature_interactions.number_of_nodes())
        for i in range(initial_agents):
            agent = self.create_agent(num_traits = num_agent_traits)
            self.schedule.add(agent)
        return

    def next_feature_id(self) -> int:
        self.current_feature_id += 1
        return self.current_feature_id

    def get_features_list(self, env: bool = False) -> List[Feature]:
        return [f for f in self.feature_interactions.nodes if f.env is env]

    def create_interaction(
        self,
        initiator: Feature,
    ) -> None:
        extant_targets = self.feature_interactions.neighbors(initiator)
        target_choices = [
            x
            for x
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
                object = interaction
            )
        return

    def create_feature(self,
        num_values: int = 5,
        env: bool = False
    ) -> Feature:
        feature_id = self.next_feature_id()
        feature = Feature(
            feature_id = feature_id, 
            num_values = num_values, 
            env = env
        )
        self.feature_interactions.add_node(feature)
        if feature.env is False:
            num_ints = self.random.randrange(1, self.num_feature_interactions)
            for i in range(num_ints):
                self.create_interaction(feature)
        return feature

    def create_agent(
        self,
        num_traits: int,
        utils:float = None,
    ) -> Agent:
        if utils is None:
            utils = self.base_utils
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

    def draw_feature_interactions(self):
        g = self.feature_interactions
        pos = nx.circular_layout(g)
        labels = {n: n.name for n in pos.keys()}
        nx.draw_networkx_nodes(g, pos, nodelist=self.get_features_list(env=False), node_color="tab:blue")
        nx.draw_networkx_nodes(g, pos, nodelist=self.get_features_list(env=True), node_color="tab:green")
        nx.draw_networkx_edges(g, pos)
        nx.draw_networkx_labels(g, pos, labels)
        plt.show()

    def draw_utility_hist(self):
        agent_fitness = [a.utils for a in self.schedule.agents]
        plt.hist(agent_fitness)
        plt.show()

    def draw_features_hist(self):
        num_features = [len(a.traits) for a in self.schedule.agents]
        plt.hist(num_features)
        plt.show()

    def draw_age_hist(self):
        ages = [a.age for a in self.schedule.agents]
        plt.hist(ages)
        plt.show()

    def feature_distribution(self):
        for feature in self.get_features_list():
            agents = [a for a in self.schedule.agents if feature in a.traits]
            traits = {v: len([a for a in agents if a.traits[feature] == v]) for v in feature.values}
            print(feature, len(agents))
            print(traits)

    def step(self):
        self.schedule.step()
        print("Step{0} - {1} agents".format(self.schedule.time,len(self.schedule.agents)))
        self.datacollecter.collect(self)


