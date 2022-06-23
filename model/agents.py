from typing import Dict, List, Tuple, Optional, Set
from random import Random
from mesa import Agent

Payoff = Tuple[float, float]
PayoffDict = Dict[str, Dict[str, Payoff]]
Position = Tuple["Interaction", str]

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

    def get_agent_target(
        self,
        feature: Feature,
        agent: Agent
    ) -> Optional[Agent]:
        choices = [
            x
            for x
            in self.model.grid.get_cell_list_contents(agent.pos)
            if x is not agent and feature in x.traits and x.utils >= 0
        ]
        if len(choices) > 0:
            return self.random.choice(choices)
        else:
            return None

    def do_interaction(self, init_agent: Agent) -> None:
        if self.target.env == True:
            site = self.model.sites[init_agent.pos]
            i_value = init_agent.traits[self.initiator]
            if self.target in site.traits and site.utils[self.target] > 0:
                t_value = site.traits[self.target]
                payoff = self.payoffs[i_value][t_value]
                init_agent.idle = False
                init_agent.utils += payoff[0]
                site.utils[self.target] += payoff[1]
        else:
            target_agent = self.get_agent_target(
                feature = self.target,
                agent = init_agent
            )
            if target_agent is not None:
                i_value = init_agent.traits[self.initiator]
                t_value = target_agent.traits[self.target]
                payoff = self.payoffs[i_value][t_value]
                init_agent.idle = False
                init_agent.utils += payoff[0]
                target_agent.utils += payoff[1]

    def __repr__(self) -> str:
        return "{0}→{1}".format(
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
        feature_mutation_chance: float = 0.01,
        feature_gain_chance: float = 0.5
    ) -> None:
        super().__init__(unique_id, model)
        self.utils = utils
        self.traits = traits
        self.trait_mutation_chance = trait_mutation_chance
        self.feature_mutation_chance = feature_mutation_chance
        self.feature_gain_chance = feature_gain_chance
        self.interactions = self.get_interactions()
        self.age = 0
        self.idle = True
        self.role = self.get_role()

    def get_interactions(self, out: bool = True) -> List[Interaction]:
        features = [f for f in self.traits.keys()]
        if out:
            return [
                x[2]
                for x
                in self.model.feature_interactions.edges(
                    nbunch=features,
                    data='interaction'
                )
            ]
        else:
            return [
                x[2]
                for x
                in self.model.feature_interactions.in_edges(
                    nbunch=features,
                    data='interaction'
                )
            ]

    def get_role(self) -> Set[Position]:
        i_positions = self.get_interactions()
        t_positions = self.get_interactions(out=False)
        role = {(i, 'i') for i in i_positions}
        for t in t_positions:
            role.add((t, 't'))
        return role

    @property
    def rolename(self) -> str:
        features = sorted([feature.name for feature in self.traits.keys()])
        return "".join(features)


    def interact(self) -> None:
        for i in self.interactions:
            if self.utils >= 0:
                i.do_interaction(init_agent=self)

    def reproduce(self) -> None:
        child_traits = self.traits.copy()
        for feature in child_traits:
            if self.random.random() <= self.trait_mutation_chance:
                new_traits = [
                    x
                    for x
                    in feature.values
                    if x != child_traits[feature]
                ]
                new_traits.append('new')
                child_traits[feature] = self.random.choice(new_traits)
                if child_traits[feature] == 'new':
                    print("Novel Trait Mutation!")
                    child_traits[feature] = self.model.create_trait(feature) 
        if self.random.random() <= self.feature_mutation_chance:
            if self.random.random() < self.feature_gain_chance:
                features = [
                    f
                    for f
                    in self.model.feature_interactions.nodes
                    if f.env is False and f not in child_traits
                ]
                features.append('new')
                feature = self.random.choice(features)
                if feature == 'new':
                    feature = self.model.create_feature()
                child_traits[feature] = self.random.choice(feature.values)
            else:
                child_traits.popitem()
        if len(child_traits) > 0:
            new_agent = Agent(
                unique_id = self.model.next_id(),
                model = self.model,
                utils = self.model.base_agent_utils,
                traits = child_traits,
            )
            self.model.schedule.add(new_agent)
            self.model.grid.place_agent(new_agent, self.pos)

    def move(self) -> None:
        neighborhood = self.model.grid.get_neighborhood(
                self.pos,
                moore=True,
                include_center=False
            )
        new_position = self.random.choice(neighborhood)
        self.model.grid.move_agent(self, new_position)

    def die(self) -> None:
        self.model.schedule.remove(self)
        self.model.grid.remove_agent(self)

    def step(self):
        self.interact()
        if self.utils > len(self.traits):
            self.reproduce()
        if self.utils <= 0:
            self.die()
        else:
            self.age += 1
            if self.idle:
                self.move()

    def __repr__(self) -> str:
        return "Agent {}".format(self.unique_id)
