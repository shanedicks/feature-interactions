from typing import Dict, List, Tuple, FrozenSet

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
        model: "World",
        feature_id: int,
        env: bool,
        num_values: int = 5,
    ) -> None:
        self.id = feature_id
        self.next_value_id = 0
        self.model = model
        self.name = name_from_number(feature_id, lower=False)
        self.env = env
        self.empty_steps = 0
        self.empty_traits = {}
        self.values = []
        for i in range(num_values):
            self.values.append(self.new_value())
        self.traits_dict = {
            (x, y): {
            } for _, x, y in self.model.grid.coord_iter()
        }

    def new_value(self):
        self.next_value_id += 1
        return name_from_number(self.next_value_id)

    def in_edges(self) -> List['Interaction']:
        fi = self.model.feature_interactions
        return [x[2] for x in fi.in_edges(nbunch=self, data='interaction')]

    def out_edges(self) -> List['Interaction']:
        fi = self.model.feature_interactions
        return [x[2] for x in fi.edges(nbunch=self, data='interaction')]

    def create_trait(self) -> str:
        value = self.new_value()
        self.values.append(value)
        initated = self.out_edges()
        targeted = self.in_edges()
        for i in initated:
            i.payoffs[value] = {}
            for t_value in i.target.values:
                i.payoffs[value][t_value] = i.new_payoff(value, t_value)
        for t in targeted:
            for i_value in t.initiator.values:
                t.payoffs[i_value][value] = t.new_payoff(i_value, value)
        print("New trait {0} added to feature {1}".format(value, self))
        return value

    def remove_trait(self, value: str):
        initiated = self.out_edges()
        targeted = self.in_edges()
        for i in initiated:
            del i.payoffs[value]
        for t in targeted:
            for i_value in t.initiator.values:
                try:
                    del t.payoffs[i_value][value]
                except KeyError as e:
                    print(e)
        self.values.remove(value)
        del self.empty_traits[value]
        for s in self.model.sites:
            if value in self.traits_dict[s]:
                del self.traits_dict[s][value]
        print("Trait {0} removed from feature {1}".format(value, self))

    def check_empty(self):
        sd = self.model.sites
        if sum([v for s in sd for v in self.traits_dict[s].values()]) == 0:
            self.empty_steps += 1
        else:
            self.empty_steps = 0
        td = self.traits_dict
        for v in self.values:
            if sum([td[s][v] for s in sd if v in td[s]]) == 0:
                try:
                    self.empty_traits[v] += 1
                except KeyError:
                    self.empty_traits[v] = 1
            else:
                self.empty_traits[v] = 0

    def prune_traits(self):
        prunable = [
            t for t,c in self.empty_traits.items()
            if c >= self.model.trait_timeout
        ]
        for trait in prunable:
            self.remove_trait(trait)

    def __repr__(self) -> str:
        return self.name


class Interaction:

    def __init__(
        self,
        model: "World",
        initiator: Feature,
        target: Feature,
        trait_payoff_mod: float = 0.5,
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
            self.payoffs = self.construct_payoffs()
        else:
            self.payoffs = payoffs

    def set_anchors(self):
        anchor = 1 - self.trait_payoff_mod
        i_anchor = round(self.random.uniform(-anchor, anchor), 2)
        t_anchor = round(self.random.uniform(-anchor, anchor), 2)
        return {"i": i_anchor, "t": t_anchor}

    def construct_payoffs(self) -> PayoffDict:
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

    def __repr__(self) -> str:
        return "{0}→{1}".format(
            self.initiator.name, 
            self.target.name
        )


class Role:
    def __init__(
        self,
        model: "Model",
        features: FrozenSet["Feature"],
    ):
        self.model = model
        self.features = features
        self.rolename = ":".join(sorted([f.name for f in features]))
        self.interactions = self.get_interactions()
        self.neighbors = self.get_neighbors()
        self.types = {
            (x,y): {} for _,x,y in self.model.grid.coord_iter()
        }

    def get_interactions(self):
        fi = self.model.feature_interactions
        features = [f for f in self.features]
        interactions = {}
        interactions['initiator'] = [ 
            x[2] for x in fi.edges(nbunch=features, data='interaction')
        ]
        interactions['target'] = [
            x[2] for x in fi.in_edges(nbunch=features, data='interaction')
        ]
        return interactions

    def get_neighbors(self):
        roles = self.model.roles_dict.values()
        neighbors = {}
        neighbors["initiators"] = [
            r for r in roles
            if any(f in r.features for f in
                [i.initiator for i in self.interactions['target']]
            )
        ]
        neighbors["targets"] = [
            r for r in roles
            if any(f in r.features for f in
                [i.target for i in self.interactions['initiator']]
            )
        ]
        return neighbors

    def update(self):
        self.interactions = self.get_interactions()
        self.neighbors = self.get_neighbors()

    def __repr__(self) -> str:
        return self.rolename
