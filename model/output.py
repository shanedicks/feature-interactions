import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import random
from collections import Counter
from itertools import chain, combinations
from math import comb, log2
from statistics import mean, median, quantiles
from typing import Any, Dict, List, Set, Tuple, Iterator, Callable, Union

SpacetimeDict = Dict[Tuple[int, int], int]
RowDict = Dict[str, List[Dict[str, Any]]]


# Control
def step(world, num_steps):
    for i in range(num_steps):
        world.step()

# Model Reporters
def get_population(world: "World") -> int:
    return world.schedule.get_agent_count()

def get_total_utility(world: "World", utils:np.ndarray  = None) -> float:
    if utils is None:
        utils = np.array([a.utils for a in world.schedule.agents])
    return np.sum(utils)

def get_mean_utility(world: "World", utils: np.ndarray = None) -> float:
    if utils is None:
        utils = np.array([a.utils for a in world.schedule.agents])
    avg = np.mean(utils) if world.schedule.get_agent_count() > 0 else 0
    return avg

def get_median_utility(world: "World", utils: np.ndarray = None) -> int:
    if utils is None:
        utils = np.array([a.utils for a in world.schedule.agents])
    med = np.median(utils) if world.schedule.get_agent_count() > 0 else 0
    return med

def get_num_agent_features(world: "World") -> int:
    return len(world.get_features_list())

def get_num_roles(world: "World") -> int:
    return len(occupied_roles_list(world))

def get_num_phenotypes(world: "World") -> int:
    return len(occupied_phenotypes_list(world))

def get_model_vars_row(
        world: "World",
        sd: SpacetimeDict,
        rd: RowDict
    ):
    spacetime_id = sd["world"]
    utils = np.array([a.utils for a in world.schedule.agents])
    row = (
        spacetime_id,
        get_population(world),
        get_total_utility(world, utils),
        get_mean_utility(world, utils),
        get_median_utility(world, utils),
        get_num_phenotypes(world),
        get_num_roles(world),
        get_num_agent_features(world),
        world.new + world.cached,
        world.env
    )
    rd['model_vars'].append(row)

def get_phenotypes_rows(
        model: "Model",
        sd: SpacetimeDict,
        rd: RowDict,
        shadow: bool = False,
    ) -> None:
    for site in model.sites:
        spacetime_id = sd[site]
        phenotypes = phenotype_dist(model, site)
        for phenotype, pop in phenotypes.items():
            row = (spacetime_id, shadow, phenotype, pop)
            rd['phenotypes'].append(row)

def get_sites_rows(
        world: "World",
        sd: SpacetimeDict,
        rd: RowDict
    ) -> None:
    for pos, site in world.sites.items():
        st_id = sd[pos]
        row = (st_id, site.get_pop(), site.born, site.died, site.moved_in, site.moved_out)
        rd['demographics'].append(row)
        for k, v in site.utils.items():
            row = (st_id, int(k.trait_ids[site.traits[k]]), round(v, 2))
            rd['environment'].append(row)
        for db_id, stats in site.interaction_stats.items():
            count, i_utils, t_utils = stats
            row = (st_id, db_id, count, i_utils, t_utils)
            rd['interaction_stats'].append(row)

# Descriptives
def env_features_dist(world: "World"):
    for site in world.sites.values():
        print(site, site.traits)

def traits_dist(
        model: "Model",
        site: Tuple[int, int] = None,
    ) -> Dict["Feature", Dict[str, int]]:
    if site is None:
        site_dict = model.sites
        return_dict = {}
        for feature in model.get_features_list():
            return_dict[feature] = {}
            for trait in feature.values:
                count = sum(
                    [feature.traits_dict[site][trait] for site in site_dict if trait in feature.traits_dict[site]]
                )
                if count > 0:
                    return_dict[feature][trait] = count
    else:
        return_dict = {}
        for feature in model.get_features_list():
            return_dict[feature] = {trait: count for trait, count in feature.traits_dict[site].items() if count > 0}
    return return_dict

def role_dist(model: "Model", site: Tuple[int, int] = None) -> Dict['Role', int]:
    d = {}
    if site is None:
        site_dict = model.sites
        for role in model.roles_dict.values():
            type_dist = {}
            for site in site_dict:
                for phenotype, count in role.types[site].items():
                    if count > 0:
                        type_dist[phenotype] = type_dist.get(phenotype, 0) + count
            total = sum(type_dist.values())
            shannon = round(-sum([(count/total)*log2(count/total) for count in type_dist.values()]), 2)
            d[role] = (shannon, len(type_dist.keys()), total)
    else:
        for role in model.roles_dict.values():
            type_dist = [count for count in role.types[site].values() if count > 0]
            total = sum(type_dist)
            shannon = round(sum([(count/total)*log2(count/total) for count in type_dist]),2)
            d[role] = (shannon, len(type_dist), total)
    return sorted(
            [[role, desc] for role, desc in d.items() if desc[2] > 0], 
            key=lambda x: x[1][2],
            reverse=True
        )

def phenotype_dist(model: "Model", site: Tuple[int, int] = None) -> Dict[str, int]:
    roles_dict = model.roles_dict.values()
    phenotype_dict = {}
    if site is None:
        site_dict = model.sites
        phenotype_set = {(phenotype, role) for role in roles_dict for site in site_dict for phenotype, count in role.types[site].items() if count > 0}
        for phenotype, role in phenotype_set:
            phenotype_dict[phenotype] = sum([role.types[site][phenotype] for site in site_dict if phenotype in role.types[site]])
    else:
        phenotype_set = {(phenotype, role) for role in roles_dict for phenotype, count in role.types[site].items() if count > 0}
        for phenotype, role in phenotype_set:
            phenotype_dict[phenotype] = role.types[site][phenotype]
    return {phenotype: count for phenotype, count in phenotype_dict.items() if count > 0}

def env_report(world: "World"):
    print("Pos, Pop, Pop change, Cost, Utils")
    for s in world.sites.values():
        utils = {feature: round(utils, 2) for feature, utils in s.utils.items()}
        print(s.pos, s.get_pop(), s.born - s.died, round(s.pop_cost, 2), utils)

# Utilities

def payoff_quantiles(interaction: "Interaction"):
    payoffs = [p for item in interaction.payoffs.values() for p in item.values()]
    count = len(payoffs)
    i = [p[0] for p in payoffs]
    t = [p[1] for p in payoffs]
    if count > 2:
        return(count, [round(q,2) for q in quantiles(i)], [round(q,2) for q in quantiles(t)])
    else:
        return(count, sorted(i), sorted(t))

def print_matrix(interaction: "Interaction"):
    print(interaction)
    for item in interaction.payoffs.values():
        print([p for p in item.values()])

def interaction_report(world: "World", full: bool = False):
    fi = world.feature_interactions
    interactions = [x[2] for x in fi.edges(data='interaction')]
    if full:
        for i in interactions:
            print_matrix(i)
    else:
        print("Interaction, Payoffs, ([initiator quantiles], [target quantiles])")
        for i in interactions:
            print(i, payoff_quantiles(i))

def occupied_roles_list(model: "Model"):
    rd = model.roles_dict.values()
    sd = model.sites
    return [r for r in rd if any([v > 0 for s in sd for v in r.types[s].values()])]

def occupied_phenotypes_list(model: "Model"):
    rd = model.roles_dict.values()
    sd = model.sites
    types = {p for r in rd for s in sd for p, n in r.types[s].items() if n > 0}
    return list(types)


# Graphics - based on World objects - need reworking to draw from db

def draw_utility_hist(world: "World") -> None:
    agent_fitness = [a.utils for a in world.schedule.agents]
    plt.hist(agent_fitness)
    plt.show()

def draw_features_hist(world: "World") -> None:
    num_features = [len(a.traits) for a in world.schedule.agents]
    plt.hist(num_features)
    plt.show()

def draw_age_hist(world: "World") -> None:
    ages = [a.age for a in world.schedule.agents]
    plt.hist(ages)
    plt.show()


def env_feature_weight(feature_trait: str, agents_df, payoffs_df, base_env_utils) -> float:
    """
    Calculate the probability that any agent can interact with a specific environmental feature
    trait at a given site, based on distributions of agent features and traits and the initial utility stock.

    :param feature_trait: The feature trait (e.g., "F.a") for which to calculate the weight.
    :param agents_df: DataFrame detailing agent traits and populations.
    :param payoffs_df: DataFrame detailing payoffs for interactions.
    :param base_env_utils: Initial utility stock for environmental features.
    :return: The probability of interaction with the feature trait.
    """
    feature, trait = feature_trait.split('.')

    pop = agents_df['pop'].sum()

    in_edges_df = payoffs_df[(payoffs_df['target_feature'] == feature) & (payoffs_df['target_trait'] == trait)]

    merged_df = in_edges_df.merge(agents_df, left_on=['initiator_feature', 'initiator_trait'], right_on=['feature', 'trait'])

    # Calculate weighted impact for each initiator feature's trait
    merged_df['weighted_impact'] = (merged_df['pop'] / pop) * merged_df['initiator_payoff']
    avg_impact = merged_df.groupby(['initiator_feature', 'initiator_trait'])['weighted_impact'].sum().sum()

    # Calculate interaction probability based on avg_impact and base utility stock
    if avg_impact >= 0:
        return 1
    else:
        num_ints = base_env_utils / -avg_impact  # Assuming base_env_utils is per feature trait
        return min([num_ints/pop, 1])



# Previously Site methods or based on Site objects - need reworking to handle input from Phenotypes db
class Site_Dummy:

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
            lfd: "Local Features Dict" = None
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

# Role Evaluation - based on Site or World objects - need reworking to handle input from Phenotypes db
def check_viable(site: "Site", features: Tuple["Feature"]) -> bool:
    eu = sum([max(site.fud[f].values()) for f in features])
    cost = site.pop_cost * (len(features) ** site.model.feature_cost_exp)
    if eu - cost < 0 or len(features)==0:
        return False
    else:
        return True

def check_adjacent(site: "Site", features: Tuple["Feature"]) -> bool:
    sd = site.model.site_roles_dict[site.pos]
    occupied = sd['occupied'].keys()
    for role in occupied:
        if len(set(role.features) ^ set(features)) <= 1 and len(features)>0:
            return True
    else:
        return False

def get_num_possible(world: "World") -> int:
    return 2 ** len(world.get_features_list())

def get_occupied(site: "Site") ->Dict['Role', int]:
    rd = site.model.roles_dict.values()
    d = {r: sum(r.types[site.pos].values()) for r in rd}
    return {k:v for k,v in d.items() if v > 0}

def role_generator(world: "World") -> Iterator[Tuple['Feature']]:
    f_list = world.get_features_list()
    feature_sets = chain.from_iterable(combinations(f_list, n) for n in range(len(f_list) + 1))
    for feature_set in feature_sets:
        yield feature_set

def evaluate_rolesets(world: "World") -> None:
    dc = world.datacollector
    srd = world.site_roles_dict
    feature_set = world.get_features_list()
    for site in world.sites.values():
        site.pop_cost = site.get_pop_cost()
        site.fud = site.get_feature_utility_dict()
        sd = srd[site.pos]
        sd['occupied'] = get_occupied(site)
        sd['possible'] = get_num_possible(world)
        sd['viable'] = count_viable(site)
        site.update_role_network()
        adjacent = set()
        occupiable = set()
        for role in sd['occupied']:
            features = [f for f in feature_set if f not in role.features]
            role = sorted([f for f in role.features], key=lambda x: x.name)
            role = tuple(role)
            adjacent.add(role)
            if check_viable(site, role):
                occupiable.add(role)
            def handle(feature, add):
                adj_role = [f for f in role]
                if add:
                    adj_role.append(feature)
                else:
                    adj_role.remove(feature)
                adj_role = sorted(adj_role, key=lambda x: x.name)
                adj_role = tuple(adj_role)
                if len(adj_role) > 0:
                    adjacent.add(adj_role)
                    if check_viable(site, adj_role):
                        occupiable.add(adj_role)
            for feature in features:
                handle(feature, True)
            for feature in role:
                handle(feature, False)
        sd['adjacent'] = len(adjacent)
        sd['occupiable'] = len(occupiable)
        keys = ['Step', 'Site', "Possible",'Viable', 'Adjacent', 'Occupiable', "Occupied"]
        values = [
            world.schedule.time,
            site.pos,
            2 ** len(feature_set),
            sd['viable'],
            sd['adjacent'],
            sd['occupiable'],
            len(sd['occupied'])
        ]
        dc.add_table_row('Rolesets', {k:v for k,v in zip(keys, values)})

def count_viable(site):
    fud = site.fud
    pop_cost = site.pop_cost
    f_exp = site.model.feature_cost_exp
    best = sorted([max(fud[f].values()) for f in fud])
    if best[::-1][0] == pop_cost and pop_cost == 0:
        zeros = [x for x in best if x == 0]
        return sum([comb(len(zeros),i) for i in range(1, len(zeros) + 1)])
    if sum([x for x in best if x > 0]) < pop_cost:
        return 0
    long = 0
    scores = []
    while sum(scores) <= (pop_cost * (long ** f_exp)) and long < len(best):
        scores.append(best[long])
        long += 1
    if long < len(best):
        combs = sum([comb(len(best),i) for i in range(long, len(best) + 1)])
    elif sum(scores) >= pop_cost:
        combs = 1
    else:
        combs = 0
    best.reverse()
    short = 0
    scores.clear()
    while sum(scores) <= (pop_cost * (short ** f_exp)) and short < len(best):
        scores.append(best[short])
        short += 1
    moves = [(i, i+1) for i in range(len(best)-1)]
    for length in range(short, long):
        history = {}
        roots = []
        indices = [i for i in range(length)]
        scores = [best[i] for i in indices]
        next_moves = list(filter(
            lambda x: x[0] in indices and x[1] not in indices,
            moves
        ))
        while (sum(scores) >= (pop_cost * (length ** f_exp)) and len(next_moves) > 0) or len(roots) > 0:
            if (sum(scores) < (pop_cost * (length ** f_exp)) or len(next_moves) == 0) and len(roots) > 0:
                indices = roots[-1]
                scores = [best[i] for i in indices]
                next_moves = list(filter(
                    lambda x: x[0] in indices 
                    and x[1] not in indices
                    and x not in history[tuple(indices)],
                    moves
                ))
                roots.pop()
            if tuple(indices) not in history:
                history[tuple(indices)] = []
                if sum(scores) >= (pop_cost * (length ** f_exp)):
                    combs += 1
            o,n = max(next_moves)
            history[tuple(indices)].append((o,n))
            indices[indices.index(o)] = n
            scores = [best[i] for i in indices]
            if tuple(indices) not in history:
                history[tuple(indices)] = []
                if sum(scores) >= (pop_cost * (length ** f_exp)):
                    combs += 1
            next_moves = list(filter(
                lambda x: x[0] in indices 
                and x[1] not in indices
                and x not in history[tuple(indices)],
                moves
            ))
            if len(next_moves) > 1 and sum(scores) > (pop_cost * (length ** f_exp)):
                roots.append(indices.copy())
    return combs
