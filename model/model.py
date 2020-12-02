from typing import Dict, List, Tuple
import random

from mesa import Model
from mesa.datacollection import DataCollector
from mesa.time import RandomActivation

from .agents import Agent, Feature, Interaction


class World(Model):

    def __init__(
        self,
        features: Dict[int, Feature] = None,
        interactions: Dict[int, Interaction] = None,
        num_env_features: int = 1,
        num_agent_features: int = 2,
        num_env_interactions: int = 1,
        num_agent_interactions: int = 2,
        initial_agents: int = 10,
        num_agent_traits: int = 2,
        base_utils: float = 3.0
    ) -> None:
        super().__init__()
        self.features = features
        self.interactions = interactions
        self.base_utils = base_utils
        self.num_agent_traits = num_agent_traits
        self.schedule = RandomActivation(self)
        self.datacollecter = DataCollector()
        if self.features is None:
            self.features = {}
            self.current_feature_id = 0
            for i in range(num_env_features):
                self.create_feature(env = True)
            for i in range(num_agent_features):
                self.create_feature()
        else:
            self.current_feature_id = len(self.features)
        if self.interactions is None:
            self.interactions = {}
            self.current_interaction_id = 0
            for i in range(num_env_interactions):
                self.create_interaction(env = True)
            for i in range(num_agent_interactions):
                self.create_interaction()
        else:
            self.current_interaction_id = len(self.interactions)
        for i in range(initial_agents):
            self.create_agent(num_traits = num_agent_traits)
        return

    def next_feature_id(self) -> int:
        self.current_feature_id += 1
        return self.current_feature_id

    def next_interaction_id(self) -> int:
        self.current_interaction_id += 1
        return self.current_interaction_id

    def create_agent(
        self,
        num_traits: int,
        utils:float = None,
    ) -> None:
        if utils is None:
            utils = self.base_utils
        traits = {}
        agent_features = [f for f in self.features.values() if f.env is False]
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
        self.schedule.add(agent)
        return

    def create_feature(self, num_values: int = 5, env: bool = False) -> None:
        feature_id = self.next_feature_id()
        feature = Feature(
            model = self,
            feature_id = feature_id, 
            num_values = num_values, 
            env = env
        )
        self.features[feature_id] = feature
        return

    def get_interaction_features(self, env: bool) -> Tuple[Feature, Feature]:
        features = [f for f in self.features.values() if f.env is False]
        initiator = self.random.choice(features)
        if env is False:
            target = self.random.choice(features)
        else:
            env_features = [f for f in self.features.values() if f.env is True]
            target = self.random.choice(env_features)
        return (initiator, target)

    def create_interaction(
        self,
        env: bool = False
    ) -> None:
        interaction_id = self.next_interaction_id()
        features = self.get_interaction_features(env = env)
        initiator, target = features
        key = (initiator.name, target.name)
        while key in self.interactions:
            features = self.get_interaction_features(env = env)
            initiator, target = features
            key = (initiator.name, target.name)
        interaction = Interaction(
            model = self,
            initiator = initiator,
            target = target,
        )
        self.interactions[key] = interaction
        return

    def step(self):
        self.schedule.step()
        self.datacollecter.collect(self)

