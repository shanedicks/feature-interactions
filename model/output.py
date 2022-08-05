from collections import Counter
from itertools import chain, combinations
from typing import Dict, List, Tuple, Iterator
from statistics import mean, median
import matplotlib.pyplot as plt
import networkx as nx

# Control
def step_forward(world, num_steps):
    for i in range(num_steps):
        world.step()

# Model Reporters
def get_population(world: "World") -> int:
    return world.schedule.get_agent_count()

def get_total_utility(world: "World") -> float:
    return sum([a.utils for a in world.schedule.agents])

def get_mean_utility(world: "World") -> float:
    avg = mean([a.utils for a in world.schedule.agents]) if world.schedule.get_agent_count() > 0 else 0
    return avg

def get_median_utility(world: "World") -> int:
    med = median([a.utils for a in world.schedule.agents]) if world.schedule.get_agent_count() > 0 else 0
    return med

def get_num_agent_features(world: "World") -> int:
    return len(world.get_features_list())

def get_num_roles(world: "World") -> int:
    return len(occupied_roles_list(world))

def get_num_phenotypes(world: "World") -> int:
    return len(occupied_phenotypes_list(world))

def get_total_born(world: "World") -> int:
    return sum([site.born for site in world.sites.values()])

def get_total_died(world: "World") -> int:
    return sum([site.died for site in world.sites.values()])

def get_total_moved(world: "World") -> int:
    return sum([site.moved_in for site in world.sites.values()])

def write_roles_rows(
        model: "Model",
        step: int,
        dc: "datacollector",
        shadow: bool = False,
    ) -> None:
    keys = ['Step', 'Site', 'Shadow', 'Role', 'Pop']
    for site in model.sites:
        roles = role_dist(model, site)
        for role, pop in roles.items():
            values = [step, site, shadow, role.rolename, pop]
            dc.add_table_row('Roles', {k:v for k,v in zip(keys, values)})

def write_phenotypes_rows(
        model: "Model",
        step: int,
        dc: "datacollector",
        shadow: bool = False,
    ) -> None:
    keys = ['Step', 'Site', 'Shadow', 'Phenotype', 'Pop']
    for site in model.sites:
        phenotypes = phenotype_dist(model, site)
        for phenotype, pop in phenotypes.items():
            values = [step, site, shadow, phenotype, pop]
            dc.add_table_row('Phenotypes', {k:v for k,v in zip(keys, values)})

def write_traits_rows(
        model: "Model",
        step: int,
        dc: "datacollector",
        shadow: bool = False,
    ) -> None:
    keys = ['Step', 'Site', 'Shadow', 'Feature', 'Trait', 'Count']
    for site in model.sites:
        traits = traits_dist(model, site, shadow)
        for feature in traits:
            for trait, pop in traits[feature].items():
                values = [step, site, shadow, feature.name, trait, pop]
                dc.add_table_row('Traits', {k: v for k,v in zip(keys,values)})

def write_sites_rows(
        world: "World",
        step: int,
        dc = "datacollector"
    ) -> None:
    keys = ['Step', 'Site', 'Born', 'Died', 'Moved In', 'Moved Out']
    for pos, site in world.sites.items():
        values = [step, pos, site.born, site.died, site.moved_in, site.moved_out]
        dc.add_table_row('Sites', {k:v for k,v in zip(keys, values)})

def tables_update(world: "World") -> None:
    dc = world.datacollector
    step = world.schedule.time
    write_roles_rows(world, step, dc)
    write_roles_rows(world.shadow, step, dc, True)
    write_phenotypes_rows(world, step, dc)
    write_phenotypes_rows(world, step, dc, True)
    write_traits_rows(world, step, dc)
    write_traits_rows(world, step, dc, True)
    write_sites_rows(world, step, dc)


# Role Evaluation
def check_viable(site: "Site", features: Tuple["Feature"]) -> bool:
    eu = sum([max(site.fud[f].values()) for f in features])
    if eu - site.pop_cost < 0:
        return False
    else:
        return True

def check_adjacent(site: "Site", features: Tuple["Feature"]) -> bool:
    sd = site.model.site_roles_dict[site.pos]
    occupied = sd['occupied'].keys()
    for role in occupied:
        if len(set(role.features) ^ set(features)) <= 1:
            return True
    else:
        return False

def get_num_possible(world: "World") -> int:
    return 2 ** len(world.get_features_list())

def get_occupied(site: "Site") ->Dict['Role', int]:
    rd = site.model.roles_dict.values()
    d = {r: sum(r.phenotypes[site.pos].values()) for r in rd}
    return {k:v for k,v in d.items() if v > 0}

def role_generator(world: "World") -> Iterator[Tuple['Feature']]:
    f_list = world.get_features_list()
    feature_sets = chain.from_iterable(combinations(f_list, n) for n in range(len(f_list) + 1))
    for feature_set in feature_sets:
        yield feature_set

def evaluate_rolesets(world: "World") -> None:
    dc = world.datacollector
    srd = world.site_roles_dict
    for site in world.sites.values():
        site.pop_cost = site.get_pop_cost()
        site.fud = site.get_feature_utility_dict()
        sd = srd[site.pos]
        sd['occupied'] = get_occupied(site)
        sd['possible'] = get_num_possible(world)
        sd['viable'] = 0
        sd['adjacent'] = set()
        sd['occupiable'] = set()
    for role in role_generator(world):
        for site in world.sites.values():
            sd = srd[site.pos]
            v = check_viable(site, role)
            a = check_adjacent(site, role)
            o = a and v
            if o:
                sd['viable'] += 1
                sd['adjacent'].add(role)
                sd['occupiable'].add(role)
            elif v:
                sd['viable'] += 1
            elif a:
                sd['adjacent'].add(role)
    for site in world.sites.values():
        keys = ['Step', 'Site', 'Viable', 'Adjacent', 'Occupiable', "Occupied"]
        site.update_role_network()
        sd = srd[site.pos]
        values = [
            world.schedule.time,
            site.pos, 
            sd['viable'],
            len(sd['adjacent']),
            len(sd['occupiable']),
            len(sd['occupied'])
        ]
        dc.add_table_row('Rolesets', {k:v for k,v in zip(keys, values)})

# Graphics
def draw_feature_interactions(world: "World") -> None:
    g = world.feature_interactions
    pos = nx.circular_layout(g)
    labels = {n: n.name for n in pos.keys()}
    nx.draw_networkx_nodes(g, pos, nodelist=world.get_features_list(env=False), node_color="tab:blue")
    nx.draw_networkx_nodes(g, pos, nodelist=world.get_features_list(env=True), node_color="tab:green")
    nx.draw_networkx_edges(g, pos)
    nx.draw_networkx_labels(g, pos, labels)
    plt.show()

def draw_role_network(site: "Site") -> None:
    g = site.roles_network
    pos = nx.circular_layout(g)
    labels = {n: n.__repr__() for n in pos.keys()}
    env_nodes = [f for f in site.traits.keys()]
    role_nodes = [n for n in g.nodes() if n not in env_nodes]
    nx.draw_networkx_nodes(g, pos, nodelist=env_nodes, node_color="tab:green")
    nx.draw_networkx_nodes(g, pos, nodelist=role_nodes, node_color="tab:blue")
    nx.draw_networkx_edges(g, pos)
    nx.draw_networkx_labels(g, pos, labels)
    plt.show()

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

# Descriptives
def env_features_dist(world: "World"):
    for site in world.sites.values():
        print(site, site.traits)

def traits_dist(
        model: "Model",
        site: Tuple[int, int] = None,
        shadow: bool = False
    ) -> Dict["Feature", Dict[str, int]]:
    k = 'shadow' if shadow else 'live'
    if site is None:
        sd = model.sites
        d = {}
        for f in model.get_features_list():
            d[f] = {}
            for v in f.values:
                c = sum(
                    [f.traits_dict[s][k][v] for s in sd if v in f.traits_dict[s][k]]
                )
                if c > 0:
                    d[f][v] = c
    else:
        d = {}
        for f in model.get_features_list():
            d[f] = {v: c for v,c in f.traits_dict[site][k].items() if c > 0}
    return d

def role_dist(model: "Model", site: Tuple[int, int] = None) -> Dict['Role', int]:
    d = {}
    if site is None:
        sd = model.sites
        for r in model.roles_dict.values():
            d[r] = sum([v for s in sd for v in r.phenotypes[s].values()])
    else:
        for r in model.roles_dict.values():
            d[r] = sum([v for v in r.phenotypes[site].values()])
    return {role: count for role, count in d.items() if count > 0}

def phenotype_dist(model: "Model", site: Tuple[int, int] = None) -> Dict[str, int]:
    rd = model.roles_dict.values()
    l = sorted(occupied_phenotypes_list(model))
    d = {}
    if site is None:
        sd = model.sites
        for p in l:
            d[p] = sum(
                [r.phenotypes[s][p] for s in sd for r in rd if p in r.phenotypes[s]]
            )
    else:
        for p in l:
            d[p] = sum(
                [r.phenotypes[site][p] for r in rd if p in r.phenotypes[site]]
            )
    return {k:v for k,v in d.items() if v > 0}

def env_report(world: "World"):
    for site in world.sites.values():
        pop = len(world.grid.get_cell_list_contents(site.pos))
        utils = {k: round(v, 2) for k, v in site.utils.items()}
        print(site, pop, round(site.pop_cost,2), utils)

def get_feature_by_name(world: "World", name: str):
    f = [f for f in world.feature_interactions.nodes if f.name is name]
    f = f[0] if len(f) > 0 else None
    return f

def avg_payoff(interaction: "Interaction"):
    i = [p[0] for item in interaction.payoffs.values() for p in item.values()]
    t = [p[1] for item in interaction.payoffs.values() for p in item.values()]
    return(round(mean(i),2), round(mean(t),2))

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
        for i in interactions:
            print(i, avg_payoff(i))

def occupied_roles_list(model: "Model"):
    rd = model.roles_dict.values()
    sd = model.sites
    return [r for r in rd if any([v > 0 for s in sd for v in r.phenotypes[s].values()])]

def occupied_phenotypes_list(model: "Model"):
    rd = model.roles_dict.values()
    sd = model.sites
    types = {p for r in rd for s in sd for p, n in r.phenotypes[s].items() if n > 0}
    return list(types)

# Dataframes
def roles_dataframes(world: "World") -> Tuple['Dataframe', 'Dataframe']:
    dc = world.datacollector
    rdf = dc.get_table_dataframe('Roles')
    live = rdf.loc[rdf.Shadow==False].groupby(['Step', 'Role']).sum()
    live = live.unstack()['Pop']
    shadow = rdf.loc[rdf.Shadow==True].groupby(['Step', 'Role']).sum().unstack()
    shadow = shadow.unstack()['Pop']
    return (live, shadow)
