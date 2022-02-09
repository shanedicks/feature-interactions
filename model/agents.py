from typing import Dict, List, Tuple
from random import Random
from mesa import Agent

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
        num_values: int,
        env: bool,
    ) -> None:
        self.name = name_from_number(feature_id, lower=False)
        self.env = env
        self.values = []
        for i in range(num_values):
            value = name_from_number(i + 1)
            self.values.append(value)

    def __repr__(self) -> str:
        return self.name


class Interaction:

    def __init__(
        self,
        model: "World",
        initiator: Feature,
        target: Feature,
        payoffs: Dict[str, Dict[str, Tuple[float, float]]] = None
    ) -> None:
        self.model = model
        self.random = model.random
        self.initiator = initiator
        self.target = target
        if payoffs is None:
            self.payoffs = {}
            for i_value in self.initiator.values:
                self.payoffs[i_value] = {}
                for t_value in self.target.values:
                    i, t = round(self.random.uniform(-1, 1), 2), round(self.random.uniform(-1, 1), 2)
                    self.payoffs[i_value][t_value] = (i, t)
        else:
            self.payoffs = payoffs

    def get_interactor(
        self,
        feature: Feature,
        agent: Agent = None
    ) -> Agent:
        agents = self.model.schedule.agents
        if agent is None:
            choices = [x for x in agents if feature in x.traits]
        else:
            choices = [
                x
                for x
                in agents
                if x is not agent and feature in x.traits
        ]
        return self.random.choice(choices)

    def agent_interaction(
        self,
        init_agent: Agent = None,
        target_agent: Agent = None
    ) -> None:
        if init_agent is None:
            init_agent = self.get_interactor(feature = self.initiator)
        else:
            init_agent = init_agent
        if target_agent is None:
            target_agent = self.get_interactor(
                feature = self.target,
                agent = init_agent
            )
        else:
            target_agent = target_agent
        i_value = init_agent.traits[self.initiator]
        t_value = target_agent.traits[self.target]
        payoff = self.payoffs[i_value][t_value]
        init_agent.utils += payoff[0]
        target_agent.utils += payoff[1]

    def env_interaction(self, init_agent: Agent) -> None:
        if init_agent is None:
            init_agent = self.get_interactor(self.initiator)
        i_value = init_agent.traits[self.initiator]
        t_value = self.random.choice(self.target.values)
        init_agent.utils += self.payoffs[i_value][t_value][0]

    def do_interaction(
        self,
        init_agent: Agent = None,
        target_agent: Agent = None
    ) -> None:
        if self.target.env == True:
            self.env_interaction(init_agent)
        else:
            self.agent_interaction(init_agent, target_agent)

    def __repr__(self) -> str:
        return "{0}>{1}".format(
            self.initiator.name, 
            self.target.name
        )


class Agent(Agent):
    def __init__(
        self,
        unique_id: int,
        model: "World",
        traits: Dict["Feature", str],
        utils: float,
        trait_mutation_chance: float = 0.1,
        new_trait_chance: float = 0.02
    ) -> None:
        super().__init__(unique_id, model)
        self.utils = utils
        self.traits = traits
        self.trait_mutation_chance = trait_mutation_chance
        self.new_trait_chance = new_trait_chance
        self.interactions = self.get_interactions()
        self.age = 0

    def get_interactions(self) -> List["Interaction"]:
        features = [f for f in self.traits.keys()]
        return [
            x[2]
            for x
            in self.model.feature_interactions.edges(nbunch=features, data='object')
        ]

    def interact(self) -> None:
        for i in self.interactions:
            i.do_interaction(init_agent=self)

    def reproduce(self) -> None:
        model = self.model
        new_traits = self.traits.copy()
        for feature in new_traits:
            if model.random.random() <= self.trait_mutation_chance:
                new_values = [
                    x
                    for x
                    in feature.values
                    if x != new_traits[feature]
                ]
                new_traits[feature] = model.random.choice(new_values)
                print("Trait Mutation in {0}: {1}->{2}".format(feature, self.traits[feature], new_traits[feature]))
        if model.random.random() <= self.new_trait_chance:
            features = [
                f
                for f
                in model.feature_interactions.nodes
                if f.env is False and f not in new_traits
            ]
            features.append('new')
            feature = model.random.choice(features)
            if feature == 'new':
                feature = model.create_feature()
                print("Feature aquired - new feature {}".format(feature))
            else:
                print("Feature aquired- existing feature {}".format(feature))
            new_traits[feature] = model.random.choice(feature.values)
        new_agent = Agent(
            unique_id = model.next_id(),
            model = model,
            utils = model.base_utils,
            traits = new_traits,
        )
        #print("Reproduction! Welcome {new_agent}".format(new_agent=new_agent.unique_id))
        model.schedule.add(new_agent)
        return

    def die(self) -> None:
        self.model.schedule.remove(self)

    def step(self):
        if self.utils >= 0:
            self.interact()
            if self.utils > len(self.traits):
                self.reproduce()
            self.age += 1
        else:
            self.die()

    def __repr__(self) -> str:
        return "Agent {}".format(self.unique_id)
