from typing import Dict, List, Tuple, FrozenSet
from itertools import product

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
        db_id: int = None
    ) -> None:
        self.feature_id = feature_id
        self.current_value_id = 0
        self.current_value_db_id = 0
        self.model = model
        self.name = name_from_number(feature_id, lower=False)
        self.env = env
        self.db_id = db_id
        if self.db_id is None:
            env = 1 if self.env else 0
            row = (self.model.network_id, self.name, env)
            self.db_id = self.model.db.write_row('features', row)
            self.model.current_feature_db_id = self.db_id
        self.empty_steps = 0
        self.empty_traits = {}
        self.trait_ids = {}
        self.values = []
        for i in range(num_values):
            self.next_trait()
        self.traits_dict = {
            (x, y): {
            } for _, x, y in self.model.grid.coord_iter()
        }

    def new_value(self):
        self.current_value_id += 1
        return name_from_number(self.current_value_id)

    def in_edges(self) -> List['Interaction']:
        fi = self.model.feature_interactions
        return [x[2] for x in fi.in_edges(nbunch=self, data='interaction')]

    def out_edges(self) -> List['Interaction']:
        fi = self.model.feature_interactions
        return [x[2] for x in fi.edges(nbunch=self, data='interaction')]

    def next_trait(self) -> str:
        value = self.new_value()
        self.values.append(value)
        restored = self.model.db.get_next_trait(
            self.db_id,
            value,
        )
        if restored:
            self.trait_ids[value] = restored['trait_id']
            print("restored trait {0} to feature {1}".format(value, self))
        else:
            row = (value, self.db_id)
            self.trait_ids[value] = self.model.db.write_row('traits', row)
            print("New trait {0} added to feature {1}".format(value, self))
        if self in self.model.feature_interactions.nodes():
            self.new_payoffs = {'payoffs': []}
            self.set_payoffs(value)
            if len(self.new_payoffs['payoffs']) > 0:
                self.model.db.write_rows(self.new_payoffs)
            del self.new_payoffs
        trait_changes_row = (
            self.model.spacetime_dict["world"],
            self.trait_ids[value],
            "added"
        )
        self.model.db.write_row('trait_changes', trait_changes_row)
        return value

    def set_payoffs(self, value: str) -> None:
        initated = self.out_edges()
        targeted = self.in_edges()
        rows_list = []
        for i in initated:
            i.payoffs[value] = {}
            i_d, t_d = i.initiator.trait_ids, i.target.trait_ids
            for t_value in i.target.values:
                np, new = i.next_payoff(value, t_value)
                i.payoffs[value][t_value] = np
                row = (i.db_id, i_d[value], t_d[t_value], np[0], np[1])
                if new:
                    rows_list.append(row)
        for t in targeted:
            i_d, t_d = t.initiator.trait_ids, t.target.trait_ids
            for i_value in t.initiator.values:
                np, new = t.next_payoff(i_value, value)
                t.payoffs[i_value][value] = np
                row = (t.db_id, i_d[i_value], t_d[value], np[0], np[1])
                if new:
                    rows_list.append(row)
        self.new_payoffs['payoffs'].extend(rows_list)

    def remove_trait(self, value: str) -> None:
        initiated = self.out_edges()
        targeted = self.in_edges()
        trait_changes_row = (
            self.model.spacetime_dict["world"],
            self.trait_ids[value],
            "removed"
        )
        self.model.db.write_row('trait_changes', trait_changes_row)
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
        del self.trait_ids[value]
        for s in self.model.sites:
            if value in self.traits_dict[s]:
                del self.traits_dict[s][value]
        print("Trait {0} removed from feature {1}".format(value, self))

    def check_empty(self) -> None:
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

    def prune_traits(self) -> None:
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
        db_id: int = None,
        restored: bool = False,
        payoffs: PayoffDict = None,
        anchors = None,
    ) -> None:
        self.model = model
        self.random = model.random
        self.initiator = initiator
        self.target = target
        self.anchors = self.set_anchors() if anchors is None else anchors
        self.db_id = db_id
        if self.db_id is None:
            row = (
                self.model.network_id,
                self.initiator.db_id,
                self.target.db_id,
                self.anchors["i"],
                self.anchors["t"]
            )
            self.db_id = self.model.db.write_row('interactions', row)
        if payoffs is None and restored is False:
            self.payoffs = self.construct_payoffs()
        elif restored is True:
            self.payoffs = self.restore_payoffs()
        else:
            self.payoffs = payoffs

    def set_anchors(self):
        anchor = 1 - self.model.trait_payoff_mod
        mode = self.model.anchor_bias * anchor
        i_anchor = round(self.random.triangular(-anchor, anchor, mode), 2)
        t_anchor = round(self.random.triangular(-anchor, anchor, mode), 2)
        return {"i": i_anchor, "t": t_anchor}

    def construct_payoffs(self) -> PayoffDict:
        payoff_dict = {i: {} for i in self.initiator.values}
        row_dict = {'payoffs': []}
        i_d, t_d = self.initiator.trait_ids, self.target.trait_ids
        for i_value in self.initiator.values:
            for t_value in self.target.values:
                np = self.new_payoff()
                payoff_dict[i_value][t_value] = np
                row = (self.db_id, i_d[i_value], t_d[t_value], np[0], np[1])
                row_dict['payoffs'].append(row)
        self.model.db.write_rows(row_dict)
        return payoff_dict

    def restore_payoffs(self) -> PayoffDict:
        payoff_dict = {i: {} for i in self.initiator.values}
        payoff_set = set(product(self.initiator.values, self.target.values))
        restored = self.model.db.get_interaction_payoffs(
            self.db_id,
            self.initiator.values,
            self.target.values
        )
        for p in restored:
            payoff_dict[p['initiator']][p['target']] = (p['i_utils'], p['t_utils'])
            payoff_set.remove((p['initiator'], p['target']))
        row_dict = {'payoffs': []}
        if len(payoff_set) > 0:
            i_d, t_d = self.initiator.trait_ids, self.target.trait_ids
            for i_value, t_value in payoff_set:
                np = self.new_payoff()
                payoff_dict[i_value][t_value] = np
                row = (self.db_id, i_d[i_value], t_d[t_value], np[0], np[1])
                row_dict['payoffs'].append(row)
            self.model.db.write_rows(row_dict)
        return payoff_dict

    def next_payoff(self, i_value, t_value) -> Tuple[Payoff, bool]:
        restored = self.model.db.get_next_payoff(self.db_id, i_value, t_value)
        if restored:
            payoff = ((restored['initiator_utils'], restored['target_utils']), False)
        else:
            payoff = (self.new_payoff(), True)
        return payoff

    def new_payoff(self) -> Payoff:
        mod = self.model.trait_payoff_mod
        mode = self.model.payoff_bias * mod
        i = round(self.anchors["i"] + self.random.triangular(-mod, mod, mode), 2)
        assert i <= 1.0 and i >= -1.0
        t = round(self.anchors["t"] + self.random.triangular(-mod, mod, mode), 2)
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
