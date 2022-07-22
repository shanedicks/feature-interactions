from typing import Dict, List, Tuple, Optional, Set, Union, FrozenSet
from random import Random
from mesa import Agent
from output import draw_feature_interactions

class Role:
    def __init__(
        self,
        model: "World",
        features: FrozenSet["Feature"],
    ):
        self.model = model
        self.features = features
        self.rolename = ".".join([f.name for f in features])
        self.interactions = self.get_interactions()

    def get_interactions(self):
        features = [f for f in self.features]
        interactions = {}
        interactions['initiator'] = [
            x[2] for x
            in self.model.feature_interactions.edges(
                nbunch=features,
                data='interaction'
            )
        ]
        interactions['target'] = [
            x[2] for x
            in self.model.feature_interactions.in_edges(
                nbunch=features,
                data='interaction'
            )
        ]
        return interactions

    def __repr__(self) -> str:
        return self.rolename


class Agent(Agent):
    def __init__(
        self,
        unique_id: int,
        model: "World",
        traits: Dict["Feature", str],
        utils: float,
        trait_mutation_chance: float = 0.05,
        feature_mutation_chance: float = 0.001,
        feature_gain_chance: float = 0.5
    ) -> None:
        super().__init__(unique_id, model)
        self.utils = utils
        self.traits = traits
        self.trait_mutation_chance = trait_mutation_chance
        self.feature_mutation_chance = feature_mutation_chance
        self.feature_gain_chance = feature_gain_chance
        self.age = 0
        self.role = self.get_role()

    @property
    def phenotype(self) -> str:
        traits = sorted(["{0}{1}".format(f, v) for f, v in self.traits.items()])
        return "".join(traits)

    def get_role(self) -> Role:
        features = frozenset([f for f in self.traits.keys()])
        if features not in self.model.roles:
            role = Role(
                model = self.model,
                features = features
            )
            self.model.roles[features] = role
        return self.model.roles[features]

    def get_agent_target(self) -> Optional[Agent]:
        target_features = [
            x.target for x
            in self.role.interactions['initiator']
            if x.target.env == False
        ]
        def targetable(target):
            if target.utils >= 0 \
            and any(f in target.traits for f in target_features) \
            and target is not self:
                return True
            else:
                return False
        shuffled = self.model.sites[self.pos].local_agents_iter()
        return next(filter(targetable, shuffled), None)

    def do_env_interactions(
            self,
        ) -> None:
        site = self.model.sites[self.pos]
        interactions = [
            x for x
            in self.role.interactions['initiator']
            if x.target in site.traits.keys()
        ]
        if len(interactions) > 1:
            self.random.shuffle(interactions)
        for i in interactions:
            print(i, self, site)
            if site.utils[i.target] > 0 and self.utils >= 0:
                print("Env Int Starting:", self.utils, site.utils[i.target])
                i_value = self.traits[i.initiator]
                t_value = site.traits[i.target]
                payoff = i.payoffs[i_value][t_value]
                print(payoff)
                self.utils += payoff[0]
                site.utils[i.target] += payoff[1]
                print("Env Int Ending:", self.utils, site.utils[i.target])
            elif self.utils < 0:
                break

    def do_agent_interactions(
            self,
            initiator: Agent,
            target: Agent
        ) -> Tuple[float, float]:
        if initiator is self:
            interactions = [
                x for x
                in self.role.interactions['initiator']
                if x.target in target.traits.keys()
            ]
        else:
            interactions = [
                x for x
                in self.role.interactions['target']
                if x.initiator in initiator.traits.keys()
            ]
        i_payoff, t_payoff = 0, 0
        for i in interactions:
            print(i, initiator, target)
            i_value = initiator.traits[i.initiator]
            t_value = target.traits[i.target]
            payoff = i.payoffs[i_value][t_value]
            i_payoff += payoff[0]
            t_payoff += payoff[1]
            print(payoff)
        return(i_payoff, t_payoff)

    def interact(self) -> None:
        self.do_env_interactions()
        agent = self.get_agent_target() if self.utils >= 0 else None
        if agent is not None:
            print("{0}:{1} | {2}:{3}".format(self, self.phenotype, agent, agent.phenotype))
            print("Agent Int Starting: ", self.utils, agent.utils)
            try:
                payoffs = self.model.cached_payoffs[self.phenotype][agent.phenotype]
                print("Cached...", payoffs)
                self.model.cached += 1
                self.utils += payoffs[0]
                agent.utils += payoffs[1]
                print("Agent Int Ending: ", self.utils, agent.utils)
                if agent.utils < 0:
                    agent.die()
            except KeyError:
                self_payoff, agent_payoff = 0, 0
                results = self.do_agent_interactions(self, agent)
                self_payoff += results[0]
                agent_payoff += results[1]
                results = self.do_agent_interactions(agent, self)
                self_payoff += results[1]
                agent_payoff += results[0]
                self.utils += self_payoff
                agent.utils += agent_payoff
                payoffs = (self_payoff, agent_payoff)
                print("New...", payoffs)
                print("Agent Int Ending: ", self.utils, agent.utils)
                self.model.new += 1
                if self.phenotype not in self.model.cached_payoffs:
                    self.model.cached_payoffs[self.phenotype] = {}
                if agent.phenotype not in self.model.cached_payoffs:
                    self.model.cached_payoffs[agent.phenotype] = {}
                self.model.cached_payoffs[self.phenotype][agent.phenotype] = payoffs
                self.model.cached_payoffs[agent.phenotype][self.phenotype] = tuple(reversed(payoffs))
                if agent.utils < 0:
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
                    draw_feature_interactions(self.model)
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
            self.model.born += 1

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
        self.model.died += 1
        print("{0} died".format(self))

    def step(self):
        print(self, "step ----------------------------------")
        self.start = self.utils
        if self.utils >= 0:
            self.interact()
        pop_cost = self.model.sites[self.pos].pop_cost
        self.utils -= pop_cost
        if self.utils < 0 or self.random.random() < self.model.mortality:
            self.die()
        else:
            self.age += 1
            if self.utils > len(self.traits) * self.model.repr_multi:
                self.reproduce()
            if self.start == self.utils - pop_cost:
                print(self, "moving from", self.pos)
                self.move()
                print(self, " moved to ", self.pos)

    def __repr__(self) -> str:
        return "Agent {}".format(self.unique_id)
