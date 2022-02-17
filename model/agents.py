from typing import Dict, List, Tuple, Optional
from random import Random
from mesa import Agent

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
        trait_utility_random: bool = False,
        trait_payoff_mod: float = 0.25,
        payoffs: PayoffDict = None
    ) -> None:
        self.model = model
        self.random = model.random
        self.initiator = initiator
        self.target = target
        self.trait_payoff_mod = trait_payoff_mod
        assert trait_payoff_mod <= 1.0
        if payoffs is None:
            self.payoffs = self.construct_payoffs(random=trait_utility_random)
        else:
            self.payoffs = payoffs

    def construct_payoffs(self, random:bool) -> PayoffDict:
        payoffs = {}
        mod = self.trait_payoff_mod
        anchor = 1 - mod
        i_anchor = round(self.random.uniform(-anchor, anchor), 2)
        t_anchor = round(self.random.uniform(-anchor, anchor), 2)
        print(self, i_anchor, t_anchor)
        for i_value in self.initiator.values:
            payoffs[i_value] = {}
            for t_value in self.target.values:
                i = round(i_anchor + self.random.uniform(-mod, mod), 2)
                assert i <= 1.0 and i >= -1.0
                t = round(t_anchor + self.random.uniform(-mod, mod), 2)
                assert t <= 1.0 and t >= -1.0
                payoffs[i_value][t_value] = (i, t)
        return payoffs

    def get_interactor(
        self,
        feature: Feature,
        agent: Agent
    ) -> Optional[Agent]:
        choices = [
            x
            for x
            in self.model.grid.get_cell_list_contents([agent.pos])
            if x is not agent and feature in x.traits and x.utils >= 0
        ]
        if len(choices) > 0:
            return self.random.choice(choices)
        else:
            return None

    def agent_interaction(self, init_agent: Agent) -> None:
        target_agent = self.get_interactor(
            feature = self.target,
            agent = init_agent
        )
        if target_agent is not None:
            i_value = init_agent.traits[self.initiator]
            t_value = target_agent.traits[self.target]
            payoff = self.payoffs[i_value][t_value]
            init_agent.utils += payoff[0]
            target_agent.utils += payoff[1]

    def env_interaction(self, init_agent: Agent) -> None:
        site = self.model.sites[init_agent.pos]
        i_value = init_agent.traits[self.initiator]
        if self.target in site.traits and site.utils[self.target] > 0:
            t_value = site.traits[self.target]
            payoff = self.payoffs[i_value][t_value]
            init_agent.utils += payoff[0]
            site.utils[self.target] += payoff[1]

    def do_interaction(
        self,
        init_agent: Agent,
    ) -> None:
        if self.target.env == True:
            self.env_interaction(init_agent)
        else:
            self.agent_interaction(init_agent)

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
        traits: Dict[Feature, str],
        utils: float,
        trait_mutation_chance: float = 0.1,
        new_trait_chance: float = 0.01
    ) -> None:
        super().__init__(unique_id, model)
        self.utils = utils
        self.traits = traits
        self.trait_mutation_chance = trait_mutation_chance
        self.new_trait_chance = new_trait_chance
        self.interactions = self.get_interactions()
        self.age = 0

    def get_interactions(self) -> List[Interaction]:
        features = [f for f in self.traits.keys()]
        return [
            x[2]
            for x
            in self.model.feature_interactions.edges(nbunch=features, data='object')
        ]

    def interact(self) -> None:
        for i in self.interactions:
            if self.utils >= 0:
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
            new_traits[feature] = model.random.choice(feature.values)
        new_agent = Agent(
            unique_id = model.next_id(),
            model = model,
            utils = model.base_agent_utils,
            traits = new_traits,
        )
        #print("Reproduction! Welcome {new_agent}".format(new_agent=new_agent.unique_id))
        model.schedule.add(new_agent)
        model.grid.place_agent(new_agent, self.pos)
        return

    def die(self) -> None:
        self.model.schedule.remove(self)

    def step(self):
        self.interact()
        if self.utils > len(self.traits):
            self.reproduce()
        if self.utils < 0:
            self.die()
        else:
            self.age += 1

    def __repr__(self) -> str:
        return "Agent {}".format(self.unique_id)
