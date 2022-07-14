from collections import Counter
from typing import Dict, List, Tuple
from statistics import mean
import matplotlib.pyplot as plt
import networkx as nx

def get_population(world):
    return len(world.schedule.agents)

def get_total_utility(world):
    return sum([a.utils for a in world.schedule.agents])

def get_avg_utility(world):
    total = get_total_utility(world)
    population = get_population(world)
    avg = total / population if population > 0 else 0
    return avg

def draw_feature_interactions(world):
    g = world.feature_interactions
    pos = nx.circular_layout(g)
    labels = {n: n.name for n in pos.keys()}
    nx.draw_networkx_nodes(g, pos, nodelist=world.get_features_list(env=False), node_color="tab:blue")
    nx.draw_networkx_nodes(g, pos, nodelist=world.get_features_list(env=True), node_color="tab:green")
    nx.draw_networkx_edges(g, pos)
    nx.draw_networkx_labels(g, pos, labels)
    plt.show()

def draw_role_network(world):
    g = nx.Digraph()
    pos = nx.spring_layout(g)
    nodes = set(occupied_roles_list(world))
    nx.draw_networkx_nodes(g, pos, nodelist=nodes)

def draw_utility_hist(world):
    agent_fitness = [a.utils for a in world.schedule.agents]
    plt.hist(agent_fitness)
    plt.show()

def draw_features_hist(world):
    num_features = [len(a.traits) for a in world.schedule.agents]
    plt.hist(num_features)
    plt.show()

def draw_age_hist(world):
    ages = [a.age for a in world.schedule.agents]
    plt.hist(ages)
    plt.show()

def agent_features_dist(world):
    for feature in world.get_features_list():
        agents = [a for a in world.schedule.agents if feature in a.traits]
        traits = {v: len([a for a in agents if a.traits[feature] == v]) for v in feature.values}
        print(feature, len(agents))
        print(traits)

def env_features_dist(world):
    for site in world.sites.values():
        print(site, site.traits)

def role_dist(world):
    c = Counter([a.rolename for a in world.schedule.agents])
    return c.most_common(len(c))

def env_report(world):
	for site in world.sites.values():
		pop = len(world.grid.get_cell_list_contents(site.pos))
		print(site, pop, site.utils)

def get_feature_by_name(world, name):
    f = [f for f in world.feature_interactions.nodes if f.name is name]
    f = f[0] if len(f) > 0 else None
    return f

def avg_payoff(interaction):
    i = [p[0] for item in interaction.payoffs.values() for p in item.values()]
    t = [p[1] for item in interaction.payoffs.values() for p in item.values()]
    return(round(mean(i),2), round(mean(t),2))

def print_matrix(interaction):
    print(interaction)
    for item in interaction.payoffs.values():
        print([p for p in item.values()])

def interaction_report(world, full: bool = False):
	interactions = [
		x[2]
		for x
		in world.feature_interactions.edges(data='interaction')
	]
	if full:
		for i in interactions:
			print_matrix(i)
	else:
		for i in interactions:
			print(i, avg_payoff(i))

def occupied_roles_list(world):
	return list(set([frozenset(a.role) for a in world.schedule.agents]))

def occupied_rolenames_list(world):
    return list(set([a.rolename for a in world.schedule.agents]))

def occupied_trait_set_list(world):
	return [
		frozenset({(k,v) for k,v in a.traits.items()}) 
		for a 
		in world.schedule.agents
	]
