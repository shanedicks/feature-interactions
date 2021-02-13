from typing import Dict, List, Tuple
import random

from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import MultiGrid
from mesa.time import RandomActivation

from .agents import Agent, Feature, Interaction


class World(Model):

    def __init__(
        self,
        features: Dict[int, Feature] = None,
        interactions: Dict[int, Interaction] = None,
        num_env_features: int = 1,
        num_agent_features: int = 2,
        num_feature_interactions: int = 2,
        initial_agents: int = 100,
        num_agent_traits: int = 2,
        base_utils: float = 0.0
    ) -> None:
        super().__init__()
        self.features = features
        self.interactions = interactions
        self.base_utils = base_utils
        self.num_agent_traits = num_agent_traits
        self.num_feature_interactions = num_feature_interactions
        self.schedule = RandomActivation(self)
        self.datacollecter = DataCollector()
        if self.interactions is None:
            self.interactions = {}
        if self.features is None:
            self.features = {}
            self.current_feature_id = 0
            for i in range(num_env_features):
                self.create_feature(env = True)
            for i in range(num_agent_features):
                self.create_feature()
        else:
            self.current_feature_id = len(self.features)
        for i in range(initial_agents):
            agent = self.create_agent(num_traits = num_agent_traits)
            self.schedule.add(agent)
        return

    def next_feature_id(self) -> int:
        self.current_feature_id += 1
        return self.current_feature_id

    def get_target_feature(self) -> Feature:
        features = self.features.values()
        target = self.random.choice(features)
        return target

    def create_interaction(
        self,
        initiator: Feature,
    ) -> None:
        initiator = initiator
        extant_targets = [
            x.target
            for x
            in self.interactions.values()
            if x.initiator == initiator
        ]
        target_choices = [
            x
            for x
            in self.features.values()
            if x not in extant_targets
        ]
        if len(target_choices) > 0:
            target = self.random.choice(target_choices)
            key = (initiator.name, target.name)
            interaction = Interaction(
                model = self,
                initiator = initiator,
                target = target,
            )
            self.interactions[key] = interaction
        return

    def create_feature(self,
        num_values: int = 5,
        env: bool = False
    ) -> Feature:
        feature_id = self.next_feature_id()
        feature = Feature(
            model = self,
            feature_id = feature_id, 
            num_values = num_values, 
            env = env
        )
        self.features[feature_id] = feature
        if feature.env is False:
            for i in range(self.num_feature_interactions):
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
        return agent

    def step(self):
        self.schedule.step()
        self.datacollecter.collect(self)

