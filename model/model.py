from typing import Dict, List, Tuple
import random
import networkx as nx
import matplotlib.pyplot as plt
from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import MultiGrid
from mesa.time import RandomActivation

from agents import Agent, Feature, Interaction

def get_population(world):
    return len(world.schedule.agents)

def get_total_utility(world):
    return sum([a.utils for a in world.schedule.agents])

def get_avg_utility(world):
    total = get_total_utility(world)
    population = get_population(world)
    avg = total / population if population > 0 else 0
    return avg


class World(Model):

    def __init__(
        self,
        feature_interactions: nx.digraph.DiGraph = None,
        num_env_features: int = 2,
        num_agent_features: int = 4,
        num_feature_interactions: int = 5,
        initial_agents: int = 100,
        num_agent_traits: int = 2,
        base_utils: float = 0.0,
        grid_size: int = 2,
    ) -> None:
        super().__init__()
        self.feature_interactions = feature_interactions
        self.base_utils = base_utils
        self.num_agent_traits = num_agent_traits
        self.num_feature_interactions = num_feature_interactions
        self.grid_size = grid_size
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
                "Age": lambda agent: agent.age
            }
        )
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
            x = self.random.randrange(self.grid_size)
            y = self.random.randrange(self.grid_size)
            self.grid.place_agent(agent, (x, y))
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
        extant_targets = list(self.feature_interactions.neighbors(initiator))
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

    def step(self):
        self.schedule.step()
        print("Step{0} - {1} agents".format(self.schedule.time,len(self.schedule.agents)))
        self.datacollector.collect(self)

def draw_feature_interactions(world):
    g = world.feature_interactions
    pos = nx.circular_layout(g)
    labels = {n: n.name for n in pos.keys()}
    nx.draw_networkx_nodes(g, pos, nodelist=world.get_features_list(env=False), node_color="tab:blue")
    nx.draw_networkx_nodes(g, pos, nodelist=world.get_features_list(env=True), node_color="tab:green")
    nx.draw_networkx_edges(g, pos)
    nx.draw_networkx_labels(g, pos, labels)
    plt.show()

def draw_utility_hist(world):
    agent_fitness = [a.utils for a in world.schedule.agents]
    plt.hist(agent_fitness)
    plt.show()

def draw_features_hist(world):
    num_features = [len(a.traits) for a in world.schedule.agents]
    plt.hist(num_features)
    plt.show()

def draw_age_hist(world):
    ages = [a.age for a in world.schedule.agents]
    plt.hist(ages)
    plt.show()

def feature_distribution(world):
    for feature in world.get_features_list():
        agents = [a for a in world.schedule.agents if feature in a.traits]
        traits = {v: len([a for a in agents if a.traits[feature] == v]) for v in feature.values}
        print(feature, len(agents))
        print(traits)
