import os
import pandas as pd
import numpy as np
import logging
from math import comb
from itertools import combinations
from typing import Dict, List, Set, Tuple, Union, Optional, Any
import model.database as db

class FeatureUtilityCalculator:
    def __init__(self, db_path):
        self.db_path = db_path
        self.world_dict = db.get_world_dict(db_path)
        self.params_df = db.get_params_df(db_path)

        # Cache network-level data by network_id
        self.network_data = {}
        # Cache for feature data frames by (world_id, site)
        self.features_df_cache = {}
        # Cache for feature utility dictionaries by (world_id, site, step_num)
        self.fud_cache = {}
        # Cache for site environment features
        self.env_features_cache = {}
        
    def _ensure_network_data(self, network_id):
        if network_id not in self.network_data:
            self.network_data[network_id] = {
                'interactions_df': db.get_interactions_df(self.db_path)[
                    db.get_interactions_df(self.db_path)['network_id'] == network_id
                ],
                'payoffs_df': db.get_payoffs_df(self.db_path, network_id),
                'all_features': None,
                'feature_traits': None
            }

            interactions_df = self.network_data[network_id]['interactions_df']
            all_features = set(interactions_df['initiator'].unique()) | set(interactions_df['target'].unique())
            self.network_data[network_id]['all_features'] = all_features

            feature_traits = self._get_all_feature_traits(network_id)
            self.network_data[network_id]['feature_traits'] = feature_traits
    
    def _get_all_feature_traits(self, network_id):
        """Get all possible traits for each feature in the network."""
        feature_traits = {}
        payoffs_df = self.network_data[network_id]['payoffs_df']
        all_features = self.network_data[network_id]['all_features']

        for feature in all_features:
            # Get traits from initiator position
            initiator_traits = payoffs_df[payoffs_df['initiator_feature'] == feature]['initiator_trait'].unique()
            # Get traits from target position
            target_traits = payoffs_df[payoffs_df['target_feature'] == feature]['target_trait'].unique()
            # Combine all traits for this feature
            feature_traits[feature] = set(initiator_traits) | set(target_traits)
        return feature_traits

    def get_env_features(self, world_id, site):
        """Get environment features for a specific site in a world."""
        cache_key = (world_id, site)
        if cache_key in self.env_features_cache:
            return self.env_features_cache[cache_key]

        env_features = {}
        sites_dict = db.get_sites_dict(self.db_path)
        base_env_utils = self.params_df[self.params_df['world_id'] == world_id]['base_env_utils'].iloc[0]

        if world_id in sites_dict and site in sites_dict[world_id]:
            for feature_trait in sites_dict[world_id][site]:
                feature, trait = feature_trait.split('.')
                if feature not in env_features:
                    env_features[feature] = {}
                env_features[feature][trait] = base_env_utils

        self.env_features_cache[cache_key] = env_features
        return env_features

    def get_features_df(self, world_id, site, step_num=None):
        """Get feature distribution data for a specific site/step combination."""
        cache_key = (world_id, site)
        if cache_key not in self.features_df_cache:
            self.features_df_cache[cache_key] = db.get_local_features_df(
                self.db_path, shadow=False, world_id=world_id, site=site
            )

        if step_num is not None:
            return self.features_df_cache[cache_key][
                self.features_df_cache[cache_key]['step_num'] == step_num
            ]
        return self.features_df_cache[cache_key]

    def get_active_agent_features(self, world_id, step_num):
        """Get a dict of active agent features with their active traits at a specific step."""
        feature_changes_df = db.get_feature_changes_df(self.db_path)
        trait_changes_df = db.get_trait_changes_df(self.db_path)

        # Filter to relevant agent feature changes
        relevant_features = feature_changes_df[
            (feature_changes_df['world_id'] == world_id) &
            (feature_changes_df['step_num'] <= step_num) &
            (feature_changes_df['env'] == False)
        ]
        if relevant_features.empty:
            return {}

        # Get active feature names
        latest_feature_changes = (
            relevant_features
            .sort_values('step_num')
            .groupby('name')
            .last()
            .reset_index()
        )
        active_feature_names = latest_feature_changes[
            latest_feature_changes['change'] == 'added'
        ]['name'].tolist()

        # Filter to relevant trait changes for those features
        relevant_traits = trait_changes_df[
            (trait_changes_df['world_id'] == world_id) &
            (trait_changes_df['step_num'] <= step_num) &
            (trait_changes_df['feature_name'].isin(active_feature_names)) &
            (trait_changes_df['env'] == False)
        ]
        if relevant_traits.empty:
            return {name: [] for name in active_feature_names}

        # Determine which traits are active
        latest_trait_changes = (
            relevant_traits
            .sort_values('step_num')
            .groupby('trait_id')
            .last()
            .reset_index()
        )

        feature_traits = {name: [] for name in active_feature_names}
        for feature_name in active_feature_names:
            trait_names = latest_trait_changes[
                (latest_trait_changes['feature_name'] == feature_name) &
                (latest_trait_changes['change'] == 'added')
            ]['trait_name'].tolist()
            feature_traits[feature_name] = trait_names

        return feature_traits

    def build_lfd(self, features_df):
        """Build local feature distribution dictionary from features DataFrame."""
        lfd = {}
        for feature in features_df['feature'].unique():
            traits_df = features_df[features_df['feature'] == feature]
            trait_counts = traits_df.set_index('trait')['pop'].to_dict()
            total = sum(trait_counts.values())

            lfd[feature] = {
                'traits': trait_counts,
                'total': total,
                'dist': {t: (c/total if total > 0 else 0) for t, c in trait_counts.items()}
            }
        return lfd
    
    def calculate_env_feature_weight(self, network_id, env_feature, env_trait, lfd, base_env_utils):
        """Calculate the weight (probability) of an environment feature interaction."""
        interactions_df = self.network_data[network_id]['interactions_df']
        payoffs_df = self.network_data[network_id]['payoffs_df']

        in_edges = interactions_df[interactions_df['target'] == env_feature]

        if in_edges.empty:
            return 1  # No interactions means always available

        # Calculate population of initiators
        initiator_features = in_edges['initiator'].unique()
        pop = sum(lfd.get(f, {}).get('total', 0) for f in initiator_features)

        if pop == 0:
            return 1  # No initiators present

        # Calculate average impact
        avg_impact = 0
        for _, edge in in_edges.iterrows():
            initiator = edge['initiator']
            if initiator not in lfd:
                continue

            weight = lfd[initiator]['total'] / pop
            dist = lfd[initiator]['dist']

            # Find all payoffs for this interaction
            edge_payoffs = payoffs_df[
                (payoffs_df['interaction_id'] == edge['interaction_id']) & 
                (payoffs_df['target_trait'] == env_trait)
            ]

            for _, payoff in edge_payoffs.iterrows():
                i_trait = payoff['initiator_trait']
                if i_trait in dist:
                    avg_impact += weight * payoff['target_payoff'] * dist[i_trait]

        if avg_impact >= 0:
            return 1
        else:
            num_ints = base_env_utils / -avg_impact
            return min(num_ints/pop, 1)
    
    def calculate_trait_eu(self, world_id, feature, trait, lfd, env_features, population):
        """Calculate expected utility for a specific trait of a feature."""
        if population == 0:
            return 0

        network_id = self.world_dict[world_id]
        self._ensure_network_data(network_id)

        interactions_df = self.network_data[network_id]['interactions_df']
        payoffs_df = self.network_data[network_id]['payoffs_df']
        base_env_utils = self.params_df[self.params_df['world_id'] == world_id]['base_env_utils'].iloc[0]

        # Get incoming edges (where this feature is target)
        in_edges = interactions_df[interactions_df['target'] == feature]

        # Get outgoing edges (where this feature is initiator)
        out_edges = interactions_df[interactions_df['initiator'] == feature]

        eu = 0

        # Calculate utility from incoming edges
        for _, edge in in_edges.iterrows():
            initiator = edge['initiator']
            if initiator not in lfd:
                continue

            weight = lfd[initiator]['total'] / population
            dist = lfd[initiator]['dist']

            edge_payoffs = payoffs_df[
                (payoffs_df['interaction_id'] == edge['interaction_id']) & 
                (payoffs_df['target_trait'] == trait)
            ]

            for _, payoff in edge_payoffs.iterrows():
                i_trait = payoff['initiator_trait']
                if i_trait in dist:
                    eu += weight * payoff['target_payoff'] * dist[i_trait]

        # Calculate utility from outgoing edges
        for _, edge in out_edges.iterrows():
            target = edge['target']

            # Environmental target
            if target in env_features:
                for t_trait in env_features[target]:
                    # Calculate interaction weight for environmental feature
                    weight = self.calculate_env_feature_weight(
                        network_id, target, t_trait, lfd, base_env_utils
                    )

                    # Find payoff for this interaction
                    edge_payoffs = payoffs_df[
                        (payoffs_df['interaction_id'] == edge['interaction_id']) & 
                        (payoffs_df['initiator_trait'] == trait) &
                        (payoffs_df['target_trait'] == t_trait)
                    ]

                    if not edge_payoffs.empty:
                        eu += weight * edge_payoffs.iloc[0]['initiator_payoff']

            # Agent target
            elif target in lfd:
                weight = lfd[target]['total'] / population
                dist = lfd[target]['dist']

                edge_payoffs = payoffs_df[
                    (payoffs_df['interaction_id'] == edge['interaction_id']) & 
                    (payoffs_df['initiator_trait'] == trait)
                ]

                for _, payoff in edge_payoffs.iterrows():
                    t_trait = payoff['target_trait']
                    if t_trait in dist:
                        eu += weight * payoff['initiator_payoff'] * dist[t_trait]

        return eu
    
    def get_site_step_fud(self, world_id, site, step_num, full_traits=False):
        """
        Get feature utility dictionary for a specific site and step.
        
        Args:
            world_id: ID of the world
            site: Site position string (e.g., '(0, 0)')
            step_num: Simulation step number
            full_traits: If True, returns utilities for all traits; if False, only returns max utility
                        for each feature
        
        Returns:
            Dictionary mapping features to either:
            - max utility value (if full_traits=False)
            - dict of {trait: utility} (if full_traits=True)
        """
        cache_key = (world_id, site, step_num)
        if cache_key in self.fud_cache and not full_traits:
            return self.fud_cache[cache_key]

        network_id = self.world_dict[world_id]
        self._ensure_network_data(network_id)

        features_df = self.get_features_df(world_id, site, step_num)
        env_features = self.get_env_features(world_id, site)
        lfd = self.build_lfd(features_df)
        population = features_df['pop'].sum() if not features_df.empty else 0

        fud = {}
        # Process active agent features
        active_agent_features = self.get_active_agent_features(world_id, step_num)
        for feature in active_agent_features.keys():
            fud[feature] = {}

            # Calculate EU for each trait
            for trait in active_agent_features[feature]:
                fud[feature][trait] = self.calculate_trait_eu(
                    world_id, feature, trait, lfd, env_features, population
                )
            
            if not full_traits:
                max_trait_eu = max(fud[feature].values())
                fud[feature] = max_trait_eu
        if not full_traits:
            self.fud_cache[cache_key] = fud
        return fud
    
    def get_world_sites_steps_fud(self, world_id, steps=None):
        """Get feature utility dictionaries for all sites at specified steps in a world."""
        results = {}
        sites_dict = db.get_sites_dict(self.db_path)
        available_sites = sites_dict.get(world_id, {}).keys()

        for site in available_sites:
            site_results = {}
            features_df = self.get_features_df(world_id, site)
            available_steps = set(features_df['step_num'].unique())
            steps_to_process = available_steps.intersection(steps) if steps else available_steps

            for step in steps_to_process:
                site_results[step] = self.get_site_step_fud(world_id, site, step)

            if site_results:
                results[site] = site_results

        return results


class RoleAnalyzer:
    def __init__(self, feature_calculator):
        """Initialize with a FeatureUtilityCalculator instance."""
        self.calculator = feature_calculator
        self.db_path = feature_calculator.db_path
        self.world_dict = feature_calculator.world_dict
        self.params_df = feature_calculator.params_df
        # Cache for occupied roles by (world_id, site, step)
        self.occupied_roles_cache = {}
        
    def get_site_pop_cost(self, world_id, site, step_num):
        # Get site population limit and population cost exponent
        params = self.params_df[self.params_df['world_id'] == world_id].iloc[0]
        grid_size = params['grid_size']
        total_pop_limit = params['total_pop_limit']
        pop_cost_exp = params['pop_cost_exp']

        """Calculate population cost for a site at a given step."""
        if total_pop_limit > 0:
            # Get site population
            features_df = self.calculator.get_features_df(world_id, site, step_num)
            population = features_df['pop'].sum() if not features_df.empty else 0

            # Calculate site population limit
            site_pop_limit = total_pop_limit / (grid_size ** 2)

            # Calculate population cost
            return float((population / site_pop_limit) ** pop_cost_exp)
        else:
            return 0  # If using active_pop_limit and not total_pop_limit, return 0
    
    def check_sustainable(self, world_id, site, step_num, features):
        """Check if a role (set of features) is sustainable at a given site and step."""
        # Get feature utility dictionary
        fud = self.calculator.get_site_step_fud(world_id, site, step_num)

        # Get feature_cost_exp parameter for this world
        feature_cost_exp = self.params_df[self.params_df['world_id'] == world_id]['feature_cost_exp'].iloc[0]

        # Get population cost
        pop_cost = self.get_site_pop_cost(world_id, site, step_num)

        # Calculate cost of having these features
        cost = pop_cost * (len(features) ** feature_cost_exp)

        # Calculate expected utility (sum of max utilities for each feature)
        if len(features) == 0:
            return False

        eu = sum(fud[feature] for feature in features)

        # A role is sustainable if eu - cost >= 0
        return eu - cost >= 0
    
    def get_occupied_roles(self, world_id, site, step_num):
        """Get occupied roles at a site and step."""
        cache_key = (world_id, site, step_num)
        if cache_key in self.occupied_roles_cache:
            return self.occupied_roles_cache[cache_key]

        # Get phenotypes data
        ph_df = db.get_phenotypes_df(
            self.calculator.db_path, 
            shadow=False, 
            worlds=world_id,
            sites=site,
            steps=step_num
        )

        occupied_roles = {}
        for _, row in ph_df.iterrows():
            role = row['role']
            features = tuple(sorted(role.split(':')))
            if features:  # Skip empty roles
                occupied_roles[features] = row['pop']

        self.occupied_roles_cache[cache_key] = occupied_roles
        return occupied_roles

    def check_adjacent(self, world_id, site, step_num, features):
        """Check if a role is adjacent to any occupied role (differs by at most one feature)."""
        if not features:
            return False

        occupied_roles = self.get_occupied_roles(world_id, site, step_num)
        features_set = set(features)

        for occupied_role_features in occupied_roles:
            occupied_set = set(occupied_role_features)
            # Check if symmetric difference (features present in one set but not both) is <= 1
            if len(features_set.symmetric_difference(occupied_set)) <= 1:
                return True

        return False
    
    def count_sustainable(self, world_id, site, step_num):
        """
        Count sustainable roles
        """
        # Get all features in the network
        network_id = self.world_dict[world_id]
        self.calculator._ensure_network_data(network_id)
        active_agent_features = self.calculator.get_active_agent_features(world_id, step_num).keys()

        # Get FUD and params
        fud = self.calculator.get_site_step_fud(world_id, site, step_num)
        params = self.params_df[self.params_df['world_id'] == world_id].iloc[0]
        feature_cost_exp = params['feature_cost_exp']
        pop_cost = self.get_site_pop_cost(world_id, site, step_num)

        # Get best utility for each feature, sorted
        best = [fud[feature] for feature in active_agent_features]
        best.sort()  # Sort in ascending order

        # Handle boundary case - zero population cost
        if best[::-1][0] == pop_cost and pop_cost == 0:
            zeros = [x for x in best if x == 0]
            from math import comb
            return sum([comb(len(zeros), i) for i in range(1, len(zeros) + 1)])

        # Check if there's not enough positive utility to overcome any cost
        if sum([x for x in best if x > 0]) < pop_cost:
            return 0

        # Find the longest sustainable feature length
        longest = 0
        scores = []
        while sum(scores) <= (pop_cost * (longest ** feature_cost_exp)) and longest < len(best):
            scores.append(best[longest])
            longest += 1

        # Determine sustainable combinations for long features
        if longest < len(best):
            from math import comb
            combs = sum([comb(len(best), i) for i in range(longest, len(best) + 1)])
        elif sum(scores) >= pop_cost:
            combs = 1
        else:
            combs = 0

        # Find the shortest sustainable feature length
        best.reverse()
        short = 0
        scores.clear()
        while sum(scores) <= (pop_cost * (short ** feature_cost_exp)) and short < len(best):
            scores.append(best[short])
            short += 1

        # Create moves for the search
        moves = [(i, i+1) for i in range(len(best)-1)]

        # Explore intermediate lengths
        for length in range(short, longest):
            history = {}
            roots = []
            indices = [i for i in range(length)]
            scores = [best[i] for i in indices]
            next_moves = list(filter(
                lambda x: x[0] in indices and x[1] not in indices,
                moves
            ))

            while (sum(scores) >= (pop_cost * (length ** feature_cost_exp)) and len(next_moves) > 0) or len(roots) > 0:
                if (sum(scores) < (pop_cost * (length ** feature_cost_exp)) or len(next_moves) == 0) and len(roots) > 0:
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
                    if sum(scores) >= (pop_cost * (length ** feature_cost_exp)):
                        combs += 1

                o, n = max(next_moves)
                history[tuple(indices)].append((o, n))
                indices[indices.index(o)] = n
                scores = [best[i] for i in indices]

                if tuple(indices) not in history:
                    history[tuple(indices)] = []
                    if sum(scores) >= (pop_cost * (length ** feature_cost_exp)):
                        combs += 1

                next_moves = list(filter(
                    lambda x: x[0] in indices 
                    and x[1] not in indices
                    and x not in history[tuple(indices)],
                    moves
                ))

                if len(next_moves) > 1 and sum(scores) > (pop_cost * (length ** feature_cost_exp)):
                    roots.append(indices.copy())

        return combs

    def evaluate_rolesets(self, world_id, steps=None):
        """Evaluate viability of rolesets across all sites for specified steps."""
        results = []
        network_id = self.world_dict[world_id]

        # Get all sites for this world
        sites_dict = db.get_sites_dict(self.calculator.db_path)
        sites = sites_dict.get(world_id, {}).keys()

        # If steps not specified, get all steps with phenotype data
        if steps is None:
            phenotypes_df = db.get_phenotypes_df(
                self.calculator.db_path,
                shadow=False,
                worlds=world_id
            )
            steps = sorted(phenotypes_df['step_num'].unique())

        # Get all features for this network
        self.calculator._ensure_network_data(network_id)

        # Pre-populate the calculator's FUD cache to minimize database operations
        # This doesn't return any value we use directly, but ensures subsequent
        # calls to get_site_step_fud will retrieve from cache instead of database
        self.calculator.get_world_sites_steps_fud(world_id, steps)

        logging.info(f"World {world_id} sites: {sites}")
        for site in sites:
            logging.info(f"Evaluating world {world_id} site {site}")
            for step in steps:
                logging.info(f"Evaluating step {step}")
                active_agent_features = self.calculator.get_active_agent_features(world_id, step)
                # Get occupied roles
                occupied_roles = self.get_occupied_roles(world_id, site, step)

                # Count sustainable roles
                sustainable_count = self.count_sustainable(world_id, site, step)

                # Get the list of occupied features
                occupied_features = set()
                for role_features in occupied_roles:
                    occupied_features.update(role_features)

                # Generate all possible roles that are adjacent to occupied ones
                adjacent_roles = set(occupied_roles.keys())
                for role_features in occupied_roles:
                    role_set = set(role_features)

                    # Add roles with one feature removed
                    for feature in role_features:
                        new_role_set = role_set - {feature}
                        new_role = tuple(sorted(new_role_set))
                        if new_role:  # Skip empty roles
                            adjacent_roles.add(new_role)

                    # Add roles with one feature added
                    for feature in active_agent_features:
                        if feature not in role_set:
                            new_role_set = role_set | {feature}
                            new_role = tuple(sorted(new_role_set))
                            adjacent_roles.add(new_role)

                # Count how many adjacent roles are sustainable
                adjacent_count = len(adjacent_roles)
                occupiable_count = sum(1 for role in adjacent_roles 
                                     if self.check_sustainable(world_id, site, step, role))

                # Calculate total possible roles (2^n - 1, excluding empty set)
                possible_count = (2 ** len(active_agent_features)) - 1

                results.append({
                    'world_id': world_id,
                    'site': site,
                    'step': step,
                    'possible': possible_count,
                    'sustainable': sustainable_count,
                    'adjacent': adjacent_count,
                    'occupiable': occupiable_count,
                    'occupied': len(occupied_roles)
                })

        return pd.DataFrame(results)
