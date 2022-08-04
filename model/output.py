from collections import Counter
from itertools import chain, combinations
from typing import Dict, List, Tuple
from statistics import mean, median
import matplotlib.pyplot as plt
import networkx as nx

# Control
def step_forward(world, num_steps):
    for i in range(num_steps):
        world.step()

# Model Reporters
def get_population(world: "World"):
    return world.schedule.get_agent_count()

def get_total_utility(world: "World"):
    return sum([a.utils for a in world.schedule.agents])

def get_mean_utility(world: "World"):
    avg = mean([a.utils for a in world.schedule.agents]) if world.schedule.get_agent_count() > 0 else 0
    return avg

def get_median_utility(world: "World"):
    med = median([a.utils for a in world.schedule.agents]) if world.schedule.get_agent_count() > 0 else 0
    return med

def get_num_agent_features(world: "World"):
    return len(world.get_features_list())

def get_num_roles(world: "World"):
    return len(occupied_roles_list(world))

def get_num_phenotypes(world: "World"):
    return len(occupied_phenotypes_list(world))

def get_total_born(world):
    return sum([site.born for site in world.sites.values()])

def get_total_died(world):
    return sum([site.died for site in world.sites.values()])

def get_total_moved(world):
    return sum([site.moved_in for site in world.sites.values()])

# Role Evaluation
def check_viable(site: "Site", features: Tuple["Feature"]):
    eu = sum([max(site.fud[f].values()) for f in features])
    if eu - site.pop_cost < 0:
        return False
    else:
        return True

def check_adjacent(site: "Site", features: Tuple["Feature"]):
    sd = site.model.site_roles_dict[site.pos]
    occupied = sd['occupied'].keys()
    for role in occupied:
        if len(set(role.features) ^ set(features)) <= 1:
            return True
    else:
        return False

def get_num_possible(world: "World") -> int:
    return 2 ** len(world.get_features_list())

def get_occupied(site):
    rd = site.model.roles_dict.values()
    d = {r: sum(r.phenotypes[site.pos].values()) for r in rd}
#    c = Counter([a.role for a in site.agents()])
    return d

def role_generator(world):
    f_list = world.get_features_list()
    feature_sets = chain.from_iterable(combinations(f_list, n) for n in range(len(f_list) + 1))
    for feature_set in feature_sets:
        yield feature_set

def evaluate_rolesets(world):
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
        keys = ['Step', 'Site', 'Viable', 'Adjacent', 'Occupiable']
        site.update_role_network()
        sd = srd[site.pos]
        values = [
            world.schedule.time,
            site.pos, 
            sd['viable'],
            len(sd['adjacent']),
            len(sd['occupiable']),
        ]
        dc.add_table_row('Other Roles', {k:v for k,v in zip(keys, values)})
        keys = ['Step', 'Site', 'Role', 'Pop']
        for role, pop in sd['occupied'].items():
            values = [world.schedule.time, site.pos, role.rolename, pop]
            dc.add_table_row('Occupied Roles', {k:v for k,v in zip(keys, values)})

# Graphics
def draw_feature_interactions(world: "World"):
    g = world.feature_interactions
    pos = nx.circular_layout(g)
    labels = {n: n.name for n in pos.keys()}
    nx.draw_networkx_nodes(g, pos, nodelist=world.get_features_list(env=False), node_color="tab:blue")
    nx.draw_networkx_nodes(g, pos, nodelist=world.get_features_list(env=True), node_color="tab:green")
    nx.draw_networkx_edges(g, pos)
    nx.draw_networkx_labels(g, pos, labels)
    plt.show()

def draw_role_network(site: "Site"):
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

def draw_utility_hist(world: "World"):
    agent_fitness = [a.utils for a in world.schedule.agents]
    plt.hist(agent_fitness)
    plt.show()

def draw_features_hist(world: "World"):
    num_features = [len(a.traits) for a in world.schedule.agents]
    plt.hist(num_features)
    plt.show()

def draw_age_hist(world: "World"):
    ages = [a.age for a in world.schedule.agents]
    plt.hist(ages)
    plt.show()

# Descriptives
def agent_features_dist(world: "World"):
    for feature in world.get_features_list():
        agents = [a for a in world.schedule.agents if feature in a.traits]
        traits = {v: len([a for a in agents if a.traits[feature] == v]) for v in feature.values}
        print(feature, len(agents))
        print(traits)

def env_features_dist(world: "World"):
    for site in world.sites.values():
        print(site, site.traits)

def role_dist(world: "World"):
    rd = world.roles_dict.values()
    sd = world.sites
    d = {r: sum([v for s in sd for v in r.phenotypes[s].values()]) for r in rd}
    return d

def phenotype_dist(world: "World"):
    rd = world.roles_dict.values()
    sd = world.sites
    d = {v: sum([v for s in sd for r in rd for v in r.phenotypes[s].values()])}
    c = Counter([a.phenotype for a in world.schedule.agents])
    return c.most_common(len(c))

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

def occupied_roles_list(world: "World"):
    rd = world.roles_dict.values()
    sd = world.sites
    return [r for r in rd if any([v > 0 for s in sd for v in r.phenotypes[s].values()])]

def occupied_phenotypes_list(world: "World"):
    rd = world.roles_dict.values()
    sd = world.sites
    types = {p for r in rd for s in sd for p, n in r.phenotypes[s].items() if n > 0}
    return list(types)
