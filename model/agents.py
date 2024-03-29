from collections import defaultdict
from typing import Dict, List, Tuple, Set, Union, Iterator
from random import Random
from mesa import Agent
from model.features import Role
from model.output import *


class Site:

    def __init__(
        self,
        model: "World",
        pos: Tuple[int, int],
        traits: Dict["Feature", str] = None
    ) -> None:
        # Initialize the Site with model reference, position, and optional predefined traits.
        self.model = model
        self.random = model.random  # Random number generator.
        self.pos = pos  # Grid position of the site.

        # Set initial population dynamics and utility trackers.
        self.pop_cost = 0
        self.born = 0
        self.died = 0
        self.moved_in = 0
        self.moved_out = 0
        self.utils = {}
        self.interaction_stats = defaultdict(lambda: [0, 0, 0])

        # Access agents currently at this site.
        x, y = self.pos
        self.agents = self.model.grid[x][y].agents

        # If traits are not predefined, generate a random number of environmental traits.
        if traits is None:
            self.traits = {}
            env_features = self.model.get_features_list(env=True)
            num_traits = self.random.randrange(len(env_features) + 1)
            features = self.random.sample(env_features, num_traits)
            for feature in features:
                self.traits[feature] = self.random.choice(feature.values)
                self.utils[feature] = self.model.base_env_utils
        else:
            self.traits = traits  # Use provided traits if available.
            self.utils = {k: 0.0 for k in self.traits.keys()}

    def get_pop(self) -> int:
        return len(self.agents)

    def get_pop_cost(self) -> float:
        # Calculate the population cost for the site based on model parameters.
        m = self.model
        # If there's a population limit, calculate cost based on current population and limit.
        if m.site_pop_limit > 0:
            return float((self.get_pop() / m.site_pop_limit) ** m.pop_cost_exp)
        else:
            return 0  # Return zero if there's no population limit.

    def reset(self):
        # Reset population dynamics and utility values for the site.
        self.born = 0
        self.died = 0
        self.moved_in = 0
        self.moved_out = 0
        # Reset interaction_stats
        self.interaction_stats.clear()
        # Reset utility values for each feature to base environmental utilities.
        for feature in self.utils:
            self.utils[feature] = self.model.base_env_utils
        # Update population cost based on current population.
        self.pop_cost = self.get_pop_cost()

    def agent_sample(self, num: int):
        # Sample a specified number of agents from those present at the site.
        pop = self.get_pop()
        if pop < num:
            num = pop  # Adjust the number to sample if it exceeds current population.
        return self.random.sample(self.agents, num)  # Randomly sample agents.

    def cleanup(self):
        # Clear the list of agents at this site to break references.
        self.agents.clear()

        # Nullify references to model and other objects.
        self.model = None
        self.random = None
        self.pos = None
        self.traits = None
        self.utils = None

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
        # Initialize the Agent with a unique ID and model reference.
        super().__init__(unique_id, model)
        self.shadow = shadow
        self.utils = utils
        self.traits = traits
        self.age = 0
        self.site = None
        self.role = self.get_role()

        # Calculate the feature cost for non-shadow agents based on the number of traits and a model exponent.
        if not shadow:
            self.feature_cost = len(self.traits) ** self.model.feature_cost_exp

    @property
    def phenotype(self) -> str:
        traits = sorted(["{0}.{1}".format(f, v) for f, v in self.traits.items()])
        return ":".join(traits)

    def increment_phenotype(self):
        # Increment the count of the agent's phenotype at its current position.
        pd = self.role.types[self.pos]
        # Handle the addition of phenotype count, creating a new entry if it doesn't exist.
        pd[self.phenotype] = pd.get(self.phenotype, 0) + 1

        # For non-shadow agents, increment the count of each trait at the current position.
        if not self.shadow:
            for feature in self.traits:
                trait = self.traits[feature]
                td = feature.traits_dict[self.pos]
                # Update trait count, initializing if necessary.
                td[trait] = td.get(trait, 0) + 1

    def decrement_phenotype(self):
        # Decrement the count of the agent's phenotype at its current position.
        self.role.types[self.pos][self.phenotype] -= 1

        # For non-shadow agents, decrement the count of each trait at the current position.
        if not self.shadow:
            for feature in self.traits:
                trait = self.traits[feature]
                feature.traits_dict[self.pos][trait] -= 1

    def get_site(self):
        self.increment_phenotype()
        return self.model.sites[self.pos]

    def get_role(self) -> Role:
        # Determine the agent's role based on its traits.
        features = frozenset([f for f in self.traits.keys()])
        try:
            role = self.model.roles_dict[features]  # Retrieve existing role from the model's role dictionary.
        except KeyError:
            # Create a new role if it doesn't exist and add it to the model's role dictionary.
            new_role = Role(model=self.model, features=features)
            self.model.roles_dict[features] = new_role
            role = new_role
        return role

    def get_shadow_agent(self) -> Agent:
        # Select a random shadow agent from the same position in the shadow world.
        shadow_site = self.model.shadow.sites[self.pos]
        return self.random.choice(shadow_site.agents)  # Randomly choose and return a shadow agent.

    def get_agent_target(self) -> Union[Agent, None]:
        # Select a target agent based on role and utility considerations.
        target_roles = self.role.neighbors['targets']  # Retrieve targets for the agent's role.
        if len(target_roles) > 0:
            n = self.model.target_sample  # Number of agents to sample as potential targets.
            def targetable(target):
                # Define a function to identify if an agent is a suitable target.
                return target.utils >= 0 and target.role in target_roles and target is not self
            # Filter the sampled agents and return the first suitable target, if any.
            return next(filter(targetable, self.site.agent_sample(n)), None)
        else:
            return None  # Return None if there are no target roles.

    def do_env_interactions(self) -> None:
        # Gather interactions where the agent is the initiator and the target is a trait in the site.
        interactions = [
            x for x in self.role.interactions['initiator'] if x.target in self.site.traits.keys()
        ]

        # Randomize the interactions if there are more than one.
        if len(interactions) > 1:
            self.random.shuffle(interactions)

        # Process each interaction.
        stats = []
        for i in interactions:
            # Check if both the site and the agent have positive utility values.
            if self.site.utils[i.target] > 0 and self.utils >= 0:
                self.model.env += 1 #Increment env interactions count
                # Get the initiator and target values for the interaction.
                i_value = self.traits[i.initiator]
                t_value = self.site.traits[i.target]
                # Calculate the payoff for the interaction.
                i_utils, t_utils = i.payoffs[i_value][t_value]
                # Update the utilities of the agent and the site based on the payoff.
                self.utils += i_utils
                self.site.utils[i.target] += t_utils
                stats.append((int(i.db_id), i_utils, t_utils))
            # Stop processing if the agent's utility becomes negative.
            elif self.utils < 0:
                break
        self.update_interaction_stats(stats)

    def do_agent_interactions(self, initiator: Agent, target: Agent) -> Tuple[float, float]:
        # Determine the relevant interactions based on whether the agent is the initiator or target.
        if initiator is self:
            # If self is the initiator, find interactions where the target's traits match.
            interactions = [x for x in self.role.interactions['initiator'] if x.target in target.traits.keys()]
        else:
            # If self is the target, find interactions where the initiator's traits match.
            interactions = [x for x in self.role.interactions['target'] if x.initiator in initiator.traits.keys()]

        i_payoff, t_payoff = 0, 0  # Initialize payoffs for initiator and target.

        # Calculate and aggregate payoffs from all relevant interactions.
        stats = []
        for i in interactions:
            # Determine the initiator and target values for the interaction.
            i_value = initiator.traits[i.initiator]
            t_value = target.traits[i.target]
            # Calculate the payoff for the interaction.
            i_utils, t_utils = i.payoffs[i_value][t_value]
            stats.append((int(i.db_id), i_utils, t_utils))
            # Sum up the payoffs for the initiator and target.
            i_payoff += i_utils
            t_payoff += t_utils

        return (i_payoff, t_payoff, stats)  # Return the cumulative payoffs for initiator and target.

    def update_interaction_stats(self, interaction_stats):
        for db_id, i_utils, t_utils in interaction_stats:
            self.site.interaction_stats[db_id][0] += 1 #increment interaction count
            self.site.interaction_stats[db_id][1] += i_utils #increment initiator utils
            self.site.interaction_stats[db_id][2] += t_utils #increment target utils

    def handle_cached_target(self, agent_target, cache_result):
        self_payoff, agent_payoff, interaction_stats = cache_result
        self.model.cached += len(interaction_stats)  # Increment cached interactions count.
        self.update_interaction_stats(interaction_stats)
        # Update utilities for both agents based on cached payoffs.
        self.utils += self_payoff
        agent_target.utils += agent_payoff

    def handle_new_target(self, agent_target, cache):
        # Calculate payoffs and collect interactions stats if not cached, including interactions in both directions.
        self_payoff, agent_payoff = 0, 0
        all_stats = []
        # self as initiator
        initiator_payoff, target_payoff, stats = self.do_agent_interactions(self, agent_target)
        self_payoff += initiator_payoff
        agent_payoff += target_payoff
        all_stats.extend(stats)
        # self as target
        initiator_payoff, target_payoff, stats = self.do_agent_interactions(agent_target, self)
        self_payoff += target_payoff
        agent_payoff += initiator_payoff
        all_stats.extend(stats)
        interaction_stats = tuple(all_stats)
        self.model.new += len(interaction_stats)  # Increment count of new payoffs calculated.
        self.update_interaction_stats(interaction_stats)
        # Update utilities for both agents based on new payoffs.
        self.utils += self_payoff
        agent_target.utils += agent_payoff
        cache_value = (self_payoff, agent_payoff, interaction_stats)
        # Cache the newly calculated payoffs.
        if self.phenotype not in cache:
            cache[self.phenotype] = {}
        cache[self.phenotype][agent_target.phenotype] = cache_value

    def interact(self) -> None:
        # Conduct environmental interactions for the agent.
        self.do_env_interactions()

        # If the agent still has non-negative utility, determine a target agent for interaction
        agent_target = self.get_agent_target() if self.utils >= 0 else None
        cache = self.model.cached_payoffs  # Access the model's payoff cache.

        if agent_target is not None:
            try:
                # Retrieve cached payoffs for interactions with the target phenotype.
                cache_result = cache[self.phenotype][agent_target.phenotype]
                self.handle_cached_target(agent_target, cache_result)
            except KeyError:
                self.handle_new_target(agent_target, cache)
            # Kill target agent if its utility falls below zero.
            if agent_target.utils < 0:
                agent_target.die()

    def process_trait_mutation(self, child_traits):
        # Iterate over each feature for potential mutation.
        for feature in child_traits:
            # Check for trait mutation.
            if self.random.random() <= self.model.trait_mutate_chance:
                # Determine whether to create a new trait or mutate an existing one, excluding shadow agents.
                if self.random.random() <= self.model.trait_create_chance and not self.shadow:
                    child_traits[feature] = feature.next_trait()
                else:
                    # Select a different trait value, if available.
                    new_traits = feature.values.copy()
                    try:
                        new_traits.remove(child_traits[feature])
                    except ValueError:
                        pass
                    try:
                        child_traits[feature] = self.random.choice(new_traits)
                    except IndexError:
                        pass
        return child_traits

    def process_feature_mutation(self, child_traits):
        # Determine whether to add a new feature or alter existing features, excluding shadow agents.
        if self.random.random() <= self.model.feature_create_chance and not self.shadow:
            feature = self.model.next_feature()
            child_traits[feature] = self.random.choice(feature.values)
        else:
            # Collect agent features not possessed by the child.
            features = [
                f for f in self.model.feature_interactions.nodes
                if f.env is False and f not in child_traits
            ]
            # Add a new feature or remove an existing one.
            if self.random.random() <= self.model.feature_gain_chance and len(features) > 0:
                feature = self.random.choice(features)
                child_traits[feature] = self.random.choice(feature.values)
            elif len(child_traits) > 0:
                key = self.random.choice(list(child_traits.keys()))
                del child_traits[key]
                # Ensure shadow agents retain at least one feature.
                if self.shadow and len(child_traits) == 0:
                    features = [
                        f for f in self.model.feature_interactions.nodes
                        if f.env is False
                    ]
                    feature = self.random.choice(features)
                    child_traits[feature] = self.random.choice(feature.values)
        return child_traits

    def reproduce(self) -> None:
        # Reproduction logic, potentially mutating traits for the offspring.
        child_traits = self.traits.copy()  # Copy parent traits to child.
        child_traits = self.process_trait_mutation(child_traits)
        # Check for feature mutation.
        if self.random.random() <= self.model.feature_mutate_chance:
            child_traits = self.process_feature_mutation(child_traits)
        # Create and place a new agent if there are traits to inherit or if it's a shadow agent.
        if len(child_traits) > 0 or self.shadow:
            new_agent = Agent(
                unique_id = self.model.next_id(),
                model = self.model,
                utils = self.model.base_agent_utils,
                traits = child_traits,
                shadow = self.shadow
            )
            # Add non-shadow agents to the schedule and reproduce shadow agents.
            if not self.shadow:
                self.model.schedule.add(new_agent)
                sa = self.get_shadow_agent()
                sa.reproduce()
            self.model.grid.place_agent(new_agent, self.pos)
            new_agent.site = new_agent.get_site()
            self.site.born += 1

    def move(self) -> None:
        # Update current site's moved-out count and decrement agent's phenotype.
        self.site.moved_out += 1
        self.decrement_phenotype()

        # Determine a new position for the agent to move to.
        neighborhood = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
        new_position = self.random.choice(neighborhood)

        # If the agent is not a shadow, move its corresponding shadow agent as well.
        if not self.shadow:
            sa = self.get_shadow_agent()
            sa.site.moved_out += 1
            sa.decrement_phenotype()
            sa.model.grid.move_agent(sa, new_position)
            sa.site = sa.get_site()
            sa.site.moved_in += 1

        # Move the agent to the new position and update its site.
        self.model.grid.move_agent(self, new_position)
        self.site = self.get_site()
        self.site.moved_in += 1

    def die(self) -> None:
        # Remove the agent from the schedule and its shadow agent if it's not a shadow.
        if not self.shadow:
            self.model.schedule.remove(self)
            sa = self.get_shadow_agent()
            sa.die()

        # Decrement phenotype, remove the agent from the grid, and update the death count.
        self.decrement_phenotype()
        self.model.grid.remove_agent(self)
        self.site.died += 1

    def step(self):
        # Track the starting utility at the beginning of the step.
        self.start = self.utils

        # Perform interactions if the utility is non-negative.
        if self.utils >= 0:
            self.interact()

        # Calculate and apply the cost of features based on site population cost.
        cost = self.feature_cost * self.model.sites[self.pos].pop_cost
        self.utils -= cost

        # Handle the agent's death if utility falls below zero.
        if self.utils < 0:
            self.die()
        else:
            # Age the agent and check for reproduction conditions.
            self.age += 1
            if self.utils > len(self.traits) * self.model.repr_multi:
                self.reproduce()

            # Decide whether to move the agent based on utility change or a random chance.
            if self.utils == self.start - cost or self.random.random() < self.model.move_chance:
                self.move()

    def cleanup(self):
        # Clear any internal structures and references.
        self.site = None
        self.role = None
        self.traits = None
        self.model = None
        self.shadow = None

    def __repr__(self) -> str:
        return "Agent {}".format(self.unique_id)
