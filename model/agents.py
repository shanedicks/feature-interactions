from typing import Dict, List, Tuple, Optional, Set, Union, FrozenSet
from random import Random
from mesa import Agent
from output import *

class Role:
    def __init__(
        self,
        model: "World",
        features: FrozenSet["Feature"],
    ):
        self.model = model
        self.features = features
        self.rolename = ".".join(sorted([f.name for f in features]))
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

    def add_to_role_network(self, site):
        roles_network = site.roles_network
        roles_network.add_node(self)
        env_targets = [x for x in self.interactions['initiator'] if x.target.env == True and x.target in site.traits.keys()]
        roles_network.add_edges_from([(self, x.target) for x in env_targets])
        targeted_by_roles = [
            r['role'] for r
            in self.model.roles_dict.values()
            if any(
                f in r['role'].features for f in [x.initiator for x in self.interactions['target']]
            )
            and site in r['sites']
        ]
        roles_network.add_edges_from([(initiator, self) for initiator in targeted_by_roles])
        targeting_roles = [
            r['role'] for r
            in self.model.roles_dict.values()
            if any(
                f in r['role'].features for f in [x.target for x in self.interactions['initiator']]
            )
            and site in r['sites']
        ]
        roles_network.add_edges_from([(self, target) for target in targeting_roles])

    def __repr__(self) -> str:
        return self.rolename


class Site:

    def __init__(
        self,
        model: "World",
        pos: Tuple[int, int]
    ) -> None:
        self.model = model
        self.random = model.random
        self.pos = pos
        self.pop_cost = 0
        self.traits = {}
        self.utils = {}
        self.roles_network = nx.DiGraph()
        env_features = self.model.get_features_list(env=True)
        num_traits = self.random.randrange(len(env_features) + 1)
        features = self.random.sample(env_features, num_traits)
        for feature in features:
            self.traits[feature] = self.random.choice(feature.values)
            self.utils[feature] = self.model.base_env_utils
        self.roles_network.add_nodes_from(features)

    def local_agents_iter(self, shuffled = True):
        agents = self.model.grid.get_cell_list_contents(self.pos)
        if shuffled:
            self.random.shuffle(agents)
        for agent in agents:
            yield agent

    def prune_roles(self):
        role_nodes = [n for n in self.roles_network.nodes if type(n) is Role]
        roles = set([a.role for a in self.local_agents_iter()])
        to_remove = [n for n in role_nodes if n not in roles]
        for node in to_remove:
            self.roles_network.remove_node(node)
            print("Removed {0} from {1}".format(node, self))
            self.model.roles_dict[frozenset(node.features)]['sites'].remove(self)

    def reset(self):
        for feature in self.utils:
            self.utils[feature] = self.model.base_env_utils
        pop = len(self.model.grid.get_cell_list_contents(self.pos))
        self.pop_cost = (pop / self.model.site_pop_limit)

    def __repr__(self) -> str:
        return "Site {0}".format(self.pos)


class Agent(Agent):
    def __init__(
        self,
        unique_id: int,
        model: "World",
        traits: Dict["Feature", str],
        utils: float,
        trait_mutation_chance: float = 0.01,
        trait_creation_chance: float = 0.01,
        feature_mutation_chance: float = 0.001,
        feature_gain_chance: float = 0.5,
        feature_creation_chance: float = 0.05
    ) -> None:
        super().__init__(unique_id, model)
        self.utils = utils
        self.traits = traits
        self.trait_mutation_chance = trait_mutation_chance
        self.trait_creation_chance = trait_creation_chance
        self.feature_mutation_chance = feature_mutation_chance
        self.feature_gain_chance = feature_gain_chance
        self.feature_creation_chance = feature_creation_chance
        self.age = 0
        self.site = None
        self.role = None

    @property
    def phenotype(self) -> str:
        traits = sorted(["{0}{1}".format(f, v) for f, v in self.traits.items()])
        return "".join(traits)

    def get_site(self):
        return self.model.sites[self.pos] 

    def get_role(self) -> Role:
        features = frozenset([f for f in self.traits.keys()])
        if features not in self.model.roles_dict:
            new_role = Role(
                model = self.model,
                features = features
            )
            self.model.roles_dict[features] = {
                "role": new_role,
                'sites': []
            }
        role = self.model.roles_dict[features]['role']
        if self.site not in self.model.roles_dict[features]['sites']:
            self.model.roles_dict[features]['sites'].append(self.site)
        if role not in self.site.roles_network.nodes:
            role.add_to_role_network(site=self.site)
            print("{0} added to {1}".format(role, self.site))
        return role

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
        shuffled = self.site.local_agents_iter()
        return next(filter(targetable, shuffled), None)

    def do_env_interactions(self) -> None:
        interactions = [
            x for x
            in self.role.interactions['initiator']
            if x.target in self.site.traits.keys()
        ]
        if len(interactions) > 1:
            self.random.shuffle(interactions)
        for i in interactions:
            if self.site.utils[i.target] > 0 and self.utils >= 0:
                i_value = self.traits[i.initiator]
                t_value = self.site.traits[i.target]
                payoff = i.payoffs[i_value][t_value]
                self.utils += payoff[0]
                self.site.utils[i.target] += payoff[1]
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
            i_value = initiator.traits[i.initiator]
            t_value = target.traits[i.target]
            payoff = i.payoffs[i_value][t_value]
            i_payoff += payoff[0]
            t_payoff += payoff[1]
        return(i_payoff, t_payoff)

    def interact(self) -> None:
        self.do_env_interactions()
        agent = self.get_agent_target() if self.utils >= 0 else None
        if agent is not None:
            try:
                payoffs = self.model.cached_payoffs[self.phenotype][agent.phenotype]
                self.model.cached += 1
                self.utils += payoffs[0]
                agent.utils += payoffs[1]
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
                if self.random.random() <= self.trait_creation_chance:
                    child_traits[feature] = self.model.create_trait(feature)
                else:
                    child_traits[feature] = self.random.choice(new_traits)
        if self.random.random() <= self.feature_mutation_chance:
            if self.random.random() < self.feature_gain_chance:
                features = [
                    f for f
                    in self.model.feature_interactions.nodes
                    if f.env is False and f not in child_traits
                ]
                if self.random.random() <= self.feature_creation_chance:
                    feature = self.model.create_feature()
                else:
                    feature = self.random.choice(features)
                child_traits[feature] = self.random.choice(feature.values)
            else:
                key = self.random.choice(list(child_traits.keys()))
                del child_traits[key]
        if len(child_traits) > 0:
            new_agent = Agent(
                unique_id = self.model.next_id(),
                model = self.model,
                utils = self.model.base_agent_utils,
                traits = child_traits,
            )
            self.model.schedule.add(new_agent)
            self.model.grid.place_agent(new_agent, self.pos)
            new_agent.site = new_agent.get_site()
            new_agent.role = new_agent.get_role()
            self.model.born += 1

    def move(self) -> None:
        neighborhood = self.model.grid.get_neighborhood(
                self.pos,
                moore=True,
                include_center=False
            )
        new_position = self.random.choice(neighborhood)
        self.model.grid.move_agent(self, new_position)
        self.site = self.get_site()
        self.get_role()
        self.model.moved += 1

    def die(self) -> None:
        self.model.schedule.remove(self)
        self.model.grid.remove_agent(self)
        self.model.died += 1

    def step(self):
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
            if self.start == self.utils - pop_cost or\
               self.random.random() < self.model.move_chance:
                self.move()

    def __repr__(self) -> str:
        return "Agent {}".format(self.unique_id)
