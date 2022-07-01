from typing import Dict, List, Tuple, Optional, Set, Union
from random import Random
from mesa import Agent

Position = Tuple["Interaction", str]

class Site:

    def __init__(
        self,
        model: "World",
        pos: Tuple[int, int]
    ) -> None:
        self.model = model
        self.random = model.random
        self.pos = pos
        self.traits = {}
        self.utils = {}
        env_features = self.model.get_features_list(env=True)
        num_traits = self.random.randrange(len(env_features) + 1)
        features = self.random.sample(env_features, num_traits)
        for feature in features:
            self.traits[feature] = self.random.choice(feature.values)
            self.utils[feature] = self.model.base_env_utils

    def reset(self):
        for feature in self.utils:
            self.utils[feature] = self.model.base_env_utils

    def __repr__(self) -> str:
        return "Site {0}".format(self.pos)


class Agent(Agent):
    def __init__(
        self,
        unique_id: int,
        model: "World",
        traits: Dict["Feature", str],
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

    @property
    def rolename(self) -> str:
        features = sorted([feature.name for feature in self.traits.keys()])
        return "".join(features)

    @property
    def phenotype(self) -> str:
        traits = sorted(["{0}{1}".format(f, v) for f, v in self.traits.items()])
        return "".join(traits)

    def get_interactions(self, out: bool = True) -> List["Interaction"]:
        features = [f for f in self.traits.keys()]
        if out:
            return [
                x[2] for x
                in self.model.feature_interactions.edges(
                    nbunch=features,
                    data='interaction'
                )
            ]
        else:
            return [
                x[2] for x
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

    def get_agent_target(self) -> Optional[Agent]:
        target_features = [
            x.target for x
            in self.interactions
            if x.target.env == False
        ]
        choices = [
            x for x
            in self.model.grid.get_cell_list_contents(self.pos)
            if x is not self and any(f in x.traits for f in target_features ) and x.utils >= 0
        ]
        if len(choices) > 0:
            return self.random.choice(choices)
        else:
            return None

    def do_interactions(
            self,
            interactions: List["Interaction"],
            initiator: Agent,
            target: Union[Agent, Site]
        ) -> None:
        if isinstance(target, Agent):
            for i in interactions:
                if target.utils >= 0 and initiator.utils >= 0:
                    print(i, initiator, target)
                    print(initiator.utils, target.utils)
                    i_value = initiator.traits[i.initiator]
                    t_value = target.traits[i.target]
                    payoff = i.payoffs[i_value][t_value]
                    initiator.utils += payoff[0]
                    target.utils += payoff[1]
                    initiator.idle = False
                    print(payoff)
                else:
                    break

        else:
            for i in interactions:
                if target.utils[i.target] >= 0 and initiator.utils >= 0:
                    i_value = initiator.traits[i.initiator]
                    t_value = target.traits[i.target]
                    payoff = i.payoffs[i_value][t_value]
                    initiator.utils += payoff[0]
                    target.utils[i.target] += payoff[1]
                    initiator.idle = False
                else:
                    break

    def interact(self) -> None:
        site = self.model.sites[self.pos]
        env_ints = [
            x for x
            in self.interactions
            if x.target in site.traits.keys()
        ]
        self.do_interactions(
            interactions = env_ints,
            initiator = self,
            target = site
        )
        agent = self.get_agent_target()
        if agent is not None and self.utils >= 0:
            print("Agent Interaction")
            init_ints = [
                x for x
                in self.interactions
                if x.target in agent.traits.keys()
            ]
            self.do_interactions(
                interactions = init_ints,
                initiator = self,
                target = agent
            )
            if agent.utils < 0:
                print(agent, "died")
                agent.die()
            else:
                target_ints = [
                    x for x
                    in self.get_interactions(out = False)
                    if x.initiator in agent.traits.keys()
                ]
                self.do_interactions(
                    interactions = target_ints,
                    initiator = agent,
                    target = self
                )
                if agent.utils < 0:
                    print(agent, "died")
                    agent.die()

    def reproduce(self) -> None:
        child_traits = self.traits.copy()
        for feature in child_traits:
            if self.random.random() <= self.trait_mutation_chance:
                new_traits = [
                    x for x
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
                    f for f
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
        print("Agent Step: {0}-------------------------------------------------".format(self))
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
