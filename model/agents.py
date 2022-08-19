from typing import Dict, List, Tuple, Set, Union, Iterator
from random import Random
from mesa import Agent
from features import Role
from output import *

Lfd = Dict["Feature", Dict[str, Union[int, Dict[str, int], Dict[str, float]]]]

class Site:

    def __init__(
        self,
        model: "World",
        pos: Tuple[int, int],
        traits: Dict["Feature", str] = None
    ) -> None:
        self.model = model
        self.random = model.random
        self.pos = pos
        self.pop_cost = 0
        self.born = 0
        self.died = 0
        self.moved_in = 0
        self.moved_out = 0
        self.utils = {}
        self.fud = {}
        self.roles_network = nx.DiGraph()
        if traits is None:
            self.traits = {}
            env_features = self.model.get_features_list(env=True)
            num_traits = self.random.randrange(len(env_features) + 1)
            features = self.random.sample(env_features, num_traits)
            for feature in features:
                self.traits[feature] = self.random.choice(feature.values)
                self.utils[feature] = self.model.base_env_utils
            self.roles_network.add_nodes_from(features)
        else:
            self.traits = traits

    def agents(self) -> List['Agent']:
        return self.model.grid.get_cell_list_contents(self.pos)

    def get_pop(self) -> int:
        return len(self.agents())

    def get_pop_cost(self) -> float:
        m = self.model
        return float((self.get_pop() / m.site_pop_limit) ** m.pop_cost_exp)

    def reset(self):
        self.born = 0
        self.died = 0
        self.moved_in = 0
        self.moved_out = 0
        for feature in self.utils:
            self.utils[feature] = self.model.base_env_utils
        self.pop_cost = self.get_pop_cost()

    def shuffled_sample(self, num: int):
        if self.get_pop() < num:
            agents = self.agents()
            self.random.shuffle(agents)
        else:
            agents = self.random.sample(self.agents(), num)
        return agents

    def get_local_features_dict(self):
        lfd = {}
        for f in self.model.get_features_list():
            lfd[f] = {}
            lfd[f]['traits'] = f.traits_dict[self.pos]
            total = sum(lfd[f]['traits'].values())
            lfd[f]['total'] = total
            if total > 0:
                lfd[f]['dist'] = {i: c/total for i, c in lfd[f]['traits'].items()}
            else:
                lfd[f]['dist'] = {i: 0 for i in lfd[f]['traits']}
        return lfd

    def env_feature_weight(
            self,
            feature: "Feature",
            lfd: Lfd = None
        ) -> Union[int, float]:
        # Get the chance of initiating an interaction against Feature for an agent at a given site
        if lfd is None:
            lfd = self.get_local_features_dict()
        lt = self.traits[feature]
        edges = feature.in_edges()
        pop = sum([
            lfd[f]['total'] for f in lfd.keys()
            if f in [e.initiator for e in edges]
        ])
        avg_impact = 0
        for edge in edges:
            weight = lfd[edge.initiator]['total'] / pop if pop > 0 else 0
            dist = lfd[edge.initiator]['dist']
            avg_impact += weight * sum([edge.payoffs[i][lt][1]*dist[i] for i in dist])
        if avg_impact >= 0:
            return 1
        else:
            num_ints = self.utils[feature]/-avg_impact
            return min([num_ints/pop, 1])

    def agent_feature_eu_dict(
            self,
            feature
        ) -> Dict[str, float]:
        # Get the expected utility for each trait of Feature for an Agent at a given site
        lfd = self.get_local_features_dict()
        pop = len(self.agents())
        in_edges = feature.in_edges()
        out_edges = feature.out_edges()
        scores = {}
        for v in feature.values:
            eu = 0
            for edge in in_edges:
                weight = lfd[edge.initiator]['total'] / pop if pop > 0 else 0
                dist = lfd[edge.initiator]['dist']
                eu += weight * sum([edge.payoffs[i][v][1]*dist[i] for i in dist])
            for edge in out_edges:
                if edge.target in self.traits.keys():
                    weight = self.env_feature_weight(edge.target, lfd)
                    eu += weight * edge.payoffs[v][self.traits[edge.target]][0]
                elif edge.target.env:
                    continue
                else:
                    weight = lfd[edge.target]['total'] / pop if pop > 0 else 0
                    dist = lfd[edge.target]['dist']
                    eu += weight * sum([edge.payoffs[v][i][0]*dist[i] for i in dist])
            scores[v] = eu
        return scores

    def get_feature_utility_dict(self) -> Dict["Feature", float]:
        fud = {}
        pop = len(self.agents())
        for f in self.model.get_features_list():
            scores = self.agent_feature_eu_dict(f)
            fud[f] = scores
        return fud

    def update_role_network(self):
        rn = self.roles_network
        role_nodes = [n for n in rn.nodes() if type(n) is Role]
        occ_roles = self.model.site_roles_dict[self.pos]['occupied']
        nodes_to_remove = [n for n in role_nodes if n not in occ_roles]
        rn.remove_nodes_from(nodes_to_remove)
        nodes_to_add = [n for n in occ_roles if n not in role_nodes]
        rn.add_nodes_from(nodes_to_add)
        role_nodes.extend(nodes_to_add)
        for role in nodes_to_add:
            env_targets = [
                i.target for i in role.interactions['initiator'] 
                if i.target.env == True and i.target in self.traits.keys()
            ]
            rn.add_edges_from([(role, target) for target in env_targets])
            role_targets = [
                target for target in role_nodes
                if any(
                    f in target.features for f in [
                        i.target for i in role.interactions['initiator']
                    ]
                )
            ]
            rn.add_edges_from([(role, target) for target in role_targets])

    def __repr__(self) -> str:
        return "Site {0}".format(self.pos)


class Agent(Agent):
    def __init__(
        self,
        unique_id: int,
        model: "World",
        traits: Dict["Feature", str],
        utils: float,
        shadow: bool = False
    ) -> None:
        super().__init__(unique_id, model)
        self.shadow = shadow
        self.utils = utils
        self.traits = traits
        self.age = 0
        self.site = None
        self.role = self.get_role()

    @property
    def phenotype(self) -> str:
        traits = sorted(["{0}.{1}".format(f, v) for f, v in self.traits.items()])
        return ":".join(traits)

    def increment_phenotype(self):
        pd = self.role.types[self.pos]
        try:
            pd[self.phenotype] += 1
        except KeyError:
            pd[self.phenotype] = 1
        if not self.shadow:
            for feature in self.traits:
                trait = self.traits[feature]
                td = feature.traits_dict[self.pos]
                if trait not in td:
                    td[trait] = 0
                try:
                    td[trait] += 1
                except KeyError:
                    td[trait] = 1

    def decrement_phenotype(self):
        self.role.types[self.pos][self.phenotype] -= 1
        if not self.shadow:
            for feature in self.traits:
                trait = self.traits[feature]
                feature.traits_dict[self.pos][trait] -= 1

    def get_site(self):
        self.increment_phenotype()
        return self.model.sites[self.pos] 

    def get_role(self) -> Role:
        features = frozenset([f for f in self.traits.keys()])
        try:
            role = self.model.roles_dict[features]
        except KeyError:
            new_role = Role(
                model = self.model,
                features = features
            )
            self.model.roles_dict[features] = new_role
            role = self.model.roles_dict[features]
        return role

    def get_shadow_agent(self) -> Agent:
        agents = self.model.shadow.sites[self.pos].agents()
        return self.random.choice(agents)

    def get_agent_target(self) -> Union[Agent, None]:
        initiating = self.role.interactions['initiator']
        target_features = [x.target for x in initiating if x.target.env == False]
        n = self.model.target_sample
        def targetable(target):
            if target.utils >= 0 \
            and any(f in target.traits for f in target_features) \
            and target is not self:
                return True
            else:
                return False
        return next(filter(targetable, self.site.shuffled_sample(n)), None)

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
        cache = self.model.cached_payoffs
        if agent is not None:
            try:
                payoffs = cache[self.phenotype][agent.phenotype]
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
                if self.phenotype not in cache:
                    cache[self.phenotype] = {}
                cache[self.phenotype][agent.phenotype] = payoffs
                if agent.utils < 0:
                    agent.die()

    def reproduce(self) -> None:
        child_traits = self.traits.copy()
        for feature in child_traits:
            if self.random.random() <= self.model.trait_mutate_chance:
                new_traits = [
                    x for x in feature.values if x != child_traits[feature]
                ]
                if self.random.random() <= self.model.trait_create_chance \
                and not self.shadow:
                    child_traits[feature] = feature.create_trait()
                else:
                    child_traits[feature] = self.random.choice(new_traits)
        if self.random.random() <= self.model.feature_mutate_chance:
            if self.random.random() <= self.model.feature_create_chance \
            and not self.shadow:
                feature = self.model.create_feature()
                child_traits[feature] = self.random.choice(feature.values)
            else: 
                features = [
                    f for f in self.model.feature_interactions.nodes
                    if f.env is False and f not in child_traits
                ]
                if self.random.random() <= self.model.feature_gain_chance \
                and len(features) > 0:
                    feature = self.random.choice(features)
                    child_traits[feature] = self.random.choice(feature.values)
                elif len(child_traits) > 0:
                    key = self.random.choice(list(child_traits.keys()))
                    del child_traits[key]
        if len(child_traits) > 0 or self.shadow:
            new_agent = Agent(
                unique_id = self.model.next_id(),
                model = self.model,
                utils = self.model.base_agent_utils,
                traits = child_traits,
                shadow = self.shadow
            )
            if not self.shadow:
                self.model.schedule.add(new_agent)
            self.model.grid.place_agent(new_agent, self.pos)
            new_agent.site = new_agent.get_site()
            self.site.born += 1

    def move(self) -> None:
        self.site.moved_out += 1
        self.decrement_phenotype()
        neighborhood = self.model.grid.get_neighborhood(
                self.pos,
                moore=True,
                include_center=False
            )
        new_position = self.random.choice(neighborhood)
        if not self.shadow:
            sa = self.get_shadow_agent()
            sa.site.moved_out += 1
            sa.decrement_phenotype()
            sa.model.grid.move_agent(sa, new_position)
            sa.site = sa.get_site()
            sa.site.moved_in += 1
        self.model.grid.move_agent(self, new_position)
        self.site = self.get_site()
        self.site.moved_in += 1

    def die(self) -> None:
        if not self.shadow:
            self.model.schedule.remove(self)
        self.decrement_phenotype()
        self.model.grid.remove_agent(self)
        self.site.died += 1

    def step(self):
        self.start = self.utils
        if self.utils >= 0:
            self.interact()
        pop_cost = self.model.sites[self.pos].pop_cost * len(self.traits)
        self.utils -= pop_cost
        if self.utils < 0 or self.random.random() < self.model.mortality:
            self.die()
        else:
            self.age += 1
            if self.utils > len(self.traits) * self.model.repr_multi:
                self.reproduce()
            if self.start == self.utils - pop_cost \
            or self.random.random() < self.model.move_chance:
                self.move()

    def __repr__(self) -> str:
        return "Agent {}".format(self.unique_id)
