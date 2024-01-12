import itertools
from datetime import datetime
from typing import Any, Dict, Iterator, Iterable, List, Union
from mesa import Model
from mesa.space import accept_tuple_argument, Coordinate, Grid, GridContent
from mesa.time import BaseScheduler
from agents import Agent, Site
from features import Feature, Interaction
from output import *


class ListDict(object):

    def __init__(self):
        # Initialize the ListDict with an agent-to-index mapping and a list of agents.
        self.agent_to_index = {}
        self.agents = []

    def add_agent(self, agent: Agent):
        # Add an agent to the list and map it to its index unless it already exists.
        if agent in self.agent_to_index:
            return
        self.agents.append(agent)
        self.agent_to_index[agent] = len(self.agents) - 1

    def remove_agent(self, agent: Agent):
        # Remove an agent from the list and update the mapping.
        index = self.agent_to_index.pop(agent)  # Remove the agent from the mapping and get its index.
        last_agent = self.agents.pop()  # Remove the last agent from the list.
        # If the removed agent is not the last, place the last agent in the removed agent's position.
        if index != len(self.agents):
            self.agents[index] = last_agent
            self.agent_to_index[last_agent] = index

    def clear(self):
        self.agent_to_index.clear()
        self.agents.clear()

    def __iter__(self):
        return iter(self.agents)

    def __len__(self):
        return len(self.agents)


class ListDictMultiGrid(Grid):

    @staticmethod
    def default_val() -> Set[Agent]:
        """Default value for new cell elements."""
        return ListDict()

    def _place_agent(self, pos: Coordinate, agent: Agent) -> None:
        """Place the agent at the correct location."""
        x, y = pos
        self.grid[x][y].add_agent(agent)
        self.empties.discard(pos)

    def _remove_agent(self, pos: Coordinate, agent: Agent) -> None:
        """Remove the agent from the given location."""
        x, y = pos
        self.grid[x][y].remove_agent(agent)
        if self.is_cell_empty(pos):
            self.empties.add(pos)

    @accept_tuple_argument
    def iter_cell_list_contents(
        self, cell_list: Iterable[Coordinate]
    ) -> Iterator[GridContent]:
        """Returns an iterator of the contents of the
        cells identified in cell_list.

        Args:
            cell_list: Array-like of (x, y) tuples, or single tuple.

        Returns:
            A iterator of the contents of the cells identified in cell_list

        """
        return itertools.chain.from_iterable(
            self[x][y].agents for x, y in cell_list if not self.is_cell_empty((x, y))
        )

    def clear_grid(self):
        for _, x, y in self.coord_iter():
            self.grid[x][y].clear()


class SampledActivation(BaseScheduler):

    def agent_buffer(self, size: int) -> Iterator[Agent]:
        # Generate a buffer of agents up to a specified size.
        n = self.get_agent_count()  # Total number of agents.
        if n > size > 0:  # Limit the buffer size if necessary.
            n = size
        agent_keys = self.model.random.sample(self._agents.keys(), n)  # Randomly sample agent keys.
        # Yield agents from the sampled keys if they still exist.
        for agent_key in agent_keys:
            if agent_key in self._agents:
                yield self._agents[agent_key]

    def step(self) -> None:
        # Perform a step in the simulation, allowing a subset of agents to act.
        m = self.model
        # Have each agent in the buffer perform its step.
        for agent in self.agent_buffer(m.active_pop_limit):
            agent.step()

        # Determine agents to be removed based on a mortality rate.
        kill_list = [m.random.random() <= m.mortality for i in range(self.get_agent_count())]
        n = self.get_agent_count()  # Current agent count.
        killed = 0  # Counter for killed agents.
        # Remove agents based on the kill list and count them.
        for agent in itertools.compress(self.agents, kill_list):
            agent.die()
            killed += 1
        print(f"{killed} agents died from mortality")  # Log actual mortality count.

        # Increment simulation steps and time.
        self.steps += 1
        self.time += 1

    def clear_schedule(self):
        self._agents.clear()


class Shadow(Model):

    def __init__(self, model: "World") -> None:
        # Initialize the Shadow model as a duplicate of the given World model.
        super().__init__()
        self.model = model  # Reference to the original World model.
        self.random = model.random  # Random number generator from the World model.

        # Copy relevant attributes from the World model to the Shadow model.
        self.feature_interactions = self.model.feature_interactions
        self.base_agent_utils = self.model.base_agent_utils
        self.base_env_utils = self.model.base_env_utils
        self.repr_multi = self.model.repr_multi
        self.mortality = self.model.mortality
        self.move_chance = self.model.move_chance
        self.trait_mutate_chance = self.model.trait_mutate_chance
        self.feature_mutate_chance = self.model.feature_mutate_chance
        self.trait_create_chance = self.model.trait_create_chance
        self.feature_create_chance = self.model.feature_create_chance
        self.feature_gain_chance = self.model.feature_gain_chance
        self.site_pop_limit = self.model.site_pop_limit
        self.pop_cost_exp = self.model.pop_cost_exp

        # Create a grid and initialize sites and roles dictionaries for the Shadow model.
        self.grid = ListDictMultiGrid(self.model.grid_size, self.model.grid_size, True)
        self.sites = {}
        self.roles_dict = {}
        # Duplicate each site from the World model in the Shadow model.
        for pos, site in self.model.sites.items():
            new_site = Site(model=self, pos=pos, traits=site.traits)
            self.sites[pos] = new_site
        self.reset_shadow()  # Reset the Shadow model's state.

    def reset_shadow(self):
        # Reset the state of the Shadow model, removing all agents and recreating them.
        # Remove existing agents from shadow sites.
        for shadow_site in self.sites.values():
            for agent in shadow_site.agents:
                agent.die()
        # Duplicate each agent from the World model in the Shadow model.
        for pos, site in self.model.sites.items():
            for agent in site.agents:
                new_agent = Agent(unique_id=self.next_id(), model=self, shadow=True, utils=agent.utils, traits=agent.traits)
                self.grid.place_agent(new_agent, pos)
                new_agent.site = new_agent.get_site()
        print("Shadow Reset")  # Log the reset.

    def get_features_list(self, env: bool = False):
        # Return a list of features, filtered based on the environment flag.
        return self.model.get_features_list(env=env)

    def agents(self) -> Iterator['Agent']:
        # Iterate over all agents in the Shadow model.
        return self.grid.iter_cell_list_contents([pos for pos in self.sites])

    def cleanup(self):
        # Cleanup roles and sites
        for role in self.roles_dict.values():
            role.cleanup()
        for site in self.sites.values():
            site.cleanup()
        # Clear collections
        self.grid.clear_grid()
        self.sites.clear()
        self.roles_dict.clear()
        # Remove references to complex objects and collections
        self.grid = None
        self.model = None
        self.random = None


class World(Model):

    def __init__(
        self,
        controller: "Controller",
        world_id,
        network_id,
        # features network variables
        init_env_features: int,
        init_agent_features: int,
        max_feature_interactions: int,
        trait_payoff_mod: float,
        anchor_bias: float,
        payoff_bias: float,
        # mutation variables
        trait_mutate_chance: float,
        trait_create_chance: float,
        feature_mutate_chance: float,
        feature_create_chance: float,
        feature_gain_chance: float,
        feature_timeout: int,
        trait_timeout: int,
        #other variables
        init_agents: int,
        base_agent_utils: float,
        base_env_utils: float,
        total_pop_limit: int,
        pop_cost_exp: int,
        feature_cost_exp: float,
        grid_size: int,
        repr_multi: int,
        mortality: float,
        move_chance: float,
        snap_interval: int,
        target_sample: int,
        active_pop_limit: int,
    ) -> None:
        # Initialize the World model with various simulation parameters and settings.
        super().__init__()
        # Assertions to ensure parameter values are within expected bounds.
        assert trait_payoff_mod <= 1 and trait_payoff_mod >= 0
        assert anchor_bias <= 1 and anchor_bias >= -1
        # Set up initial configuration and parameters of the World model.
        self.controller = controller  # Reference to the controller managing this model.
        self.db = controller.db_manager  # Database manager associated with the controller.
        self.world_id = world_id
        self.network_id = network_id
        # Various parameters defining feature interactions, mutation variables, agent utilities, etc.
        self.base_agent_utils = base_agent_utils
        self.base_env_utils = base_env_utils
        self.max_feature_interactions = max_feature_interactions
        self.trait_payoff_mod = trait_payoff_mod
        self.anchor_bias = anchor_bias
        self.payoff_bias = payoff_bias
        self.trait_mutate_chance = trait_mutate_chance
        self.feature_mutate_chance = feature_mutate_chance
        self.trait_create_chance = trait_create_chance
        self.feature_create_chance = feature_create_chance
        self.feature_gain_chance = feature_gain_chance
        self.grid_size = grid_size
        self.repr_multi = repr_multi
        self.mortality = mortality
        self.move_chance = move_chance
        self.active_pop_limit = active_pop_limit
        # Calculate site population limit based on total population limit and grid size.
        self.site_pop_limit = total_pop_limit / (grid_size ** 2)
        self.pop_cost_exp = pop_cost_exp
        self.feature_cost_exp = feature_cost_exp

        # Initialize the grid and scheduling for the agents.
        self.grid = ListDictMultiGrid(grid_size, grid_size, True)
        self.schedule = SampledActivation(self)

        # Initialize various dictionaries and data structures for managing the model's state.
        self.cached_payoffs = {}
        self.roles_dict = {}
        self.sites = {}
        self.feature_timeout = feature_timeout
        self.trait_timeout = trait_timeout
        self.target_sample = target_sample
        self.snap_interval = snap_interval
        # Prepare for database interaction and get network dataframes.
        self.db_rows = self.get_db_rows_dict()
        self.spacetimes = self.spacetime_enumerator()
        self.spacetime_dict = self.get_spacetime_dict()
        self.network_dfs = self.db.get_network_dataframes(self.network_id)
        self.db_ids = self.get_db_ids_dict()
        # Initialize features network, sites, and agents in the World model.
        self.get_or_create_init_features_network(
            init_env_features,
            init_agent_features
        )
        self.create_sites()
        self.create_init_agents(init_agents)
        # Create a Shadow model mirroring the World model.
        self.shadow = Shadow(model=self)
        self.running = True  # Set the model to running state.

        # Print initial reports for the environment, roles distribution, and interactions.
        print("Environment -------------------")
        env_report(self)
        print("Roles Distribution ------------")
        print(role_dist(self))
        print("Interaction Report ------------")
        interaction_report(self)

    def next_feature_id(self) -> int:
        # Increment and return the next unique feature identifier.
        self.current_feature_id += 1
        return self.current_feature_id

    def next_db_id(self, table_name: str):
        # Retrieve and increment the next database identifier for a specific table.
        db_id = self.db_ids[table_name]
        self.db_ids[table_name] += self.controller.total_networks
        return db_id

    def get_features_list(self, env: bool = False) -> List[Feature]:
        # Return a list of features filtered based on whether they are environmental.
        return [f for f in self.feature_interactions.nodes if f.env is env]

    def get_feature_by_name(self, name: str):
        # Find and return a feature by its name, returning None if not found.
        f = [f for f in self.feature_interactions.nodes if f.name == name]
        return f[0] if len(f) > 0 else None

    def get_or_create_init_features_network(self, num_env: int, num_agent: int) -> None:
        # Initialize or restores the features network for the model.
        self.current_feature_id = 0
        self.current_feature_db_id = 0
        self.feature_interactions = nx.DiGraph()  # Initialize a directed graph for feature interactions.

        # Get or create the specified number of environmental features.
        for i in range(num_env):
            self.next_feature(env=True)

        # Get or create the specified number of agent features.
        for i in range(num_agent):
            self.next_feature()

        # Print the lists of environmental and agent features.
        print(f"Env Features {self.get_features_list(env=True)}")
        print(f"Agent Features {self.get_features_list()}")

    def create_sites(self) -> None:
        # Initialize sites on the grid.
        for _, x, y in self.grid.coord_iter():
            pos = (x, y)  # Determine the grid position.
            site = Site(model=self, pos=pos)  # Create a new site at the position.
            self.sites[pos] = site  # Store the site in the sites dictionary.

    def create_init_agents(self, init_agents: int) -> None:
        # Create and add initial agents to the simulation.
        for i in range(init_agents):
            agent = self.create_agent()  # Create a new agent.
            self.schedule.add(agent)  # Add the agent to the schedule for activation.

        # Assign each agent to its respective site.
        for agent in self.schedule.agents:
            agent.site = agent.get_site()  # Set the agent's site based on its position.

    def next_feature(self, env: bool = False) -> Feature:
        # Retrieve the next feature from the database or create a new one if not found.
        restored = self.db.get_next_feature(self)
        if restored:
            # Restore the feature from the database if it exists.
            feature = self.restore_feature(restored)
            self.next_db_id('features')  # Increment the database ID for features.
        else:
            # Create a new feature if not found in the database.
            feature = self.create_feature(env=env)

        # Record the addition of the new or restored feature.
        feature_changes_row = (self.spacetime_dict["world"], feature.db_id, "added")
        self.db_rows['feature_changes'].append(feature_changes_row)
        return feature

    def restore_feature(self, feature_dict: Dict[str, Any]):
        # Restore a feature from a dictionary representation.
        feature_db_id = int(feature_dict['feature_id'])
        self.current_feature_db_id = feature_db_id  # Update the current feature database ID.
        feature_id = self.next_feature_id()  # Get the next feature ID.
        # Create a Feature instance from the restored data.
        feature = Feature(feature_id=feature_id, db_id=feature_db_id, model=self, env=feature_dict['env'])
        self.feature_interactions.add_node(feature)  # Add the restored feature to the interactions graph.
        print(f"restored feature {feature.db_id} {feature}")  # Log the restoration.
        self.restore_interactions(feature)  # Restore interactions related to the feature.
        return feature

    def create_feature(self, env: bool = False) -> Feature:
        # Create a new feature.
        feature_id = self.next_feature_id()  # Get the next feature ID.
        feature = Feature(feature_id=feature_id, model=self, env=env)  # Create a new Feature instance.
        self.feature_interactions.add_node(feature)  # Add the feature to the features network.

        # If the feature is not environmental, create interactions for it.
        if feature.env is False:
            num_features = self.feature_interactions.number_of_nodes()
            max_ints = min(self.max_feature_interactions, num_features)
            if max_ints > 0:
                num_ints = self.random.randrange(1, max_ints)
                for i in range(num_ints):
                    self.create_interaction(feature)  # Create interactions involving the feature.

        print(f"New feature {feature.db_id} {feature}")  # Log the creation of the new feature.
        return feature

    def restore_interactions(self, initiator: Feature) -> None:
        # Restore interactions for a given feature from the database.
        interactions = self.db.get_feature_interactions(self, initiator.db_id)
        for i in interactions:
            # Retrieve the target feature for each interaction.
            target = get_feature_by_id(self, i['target'])
            if target:
                # Create an Interaction object for each valid interaction.
                interaction = Interaction(
                    model=self,
                    initiator=initiator,
                    target=target,
                    db_id=int(i['db_id']),
                    restored=True,
                    anchors={"i": i['i_anchor'], "t": i['t_anchor']}
                )
                # Add the interaction to the feature interactions graph.
                self.feature_interactions.add_edge(initiator, target, interaction=interaction)
                print(f"restored interaction {interaction.db_id} {interaction}")  # Log the restoration.

                # Update interactions for roles affected by the restored interaction.
                affected_roles = [
                    role for role in self.roles_dict.values()
                    if any(f in role.features for f in [initiator, target])
                ]
                for role in affected_roles:
                    role.update()  # Update each affected role.

    def create_interaction(self, initiator: Feature) -> None:
        # Create a new interaction for a given feature.
        # Identify existing targets for the initiator to avoid duplication.
        extant_targets = list(self.feature_interactions.neighbors(initiator))
        # Choose a target feature for the interaction, excluding existing targets.
        target_choices = [x for x in self.feature_interactions.nodes if x not in extant_targets]

        if len(target_choices) > 0:
            # Randomly select a target from available choices.
            target = self.random.choice(target_choices)
            # Create a new Interaction instance between initiator and selected target.
            interaction = Interaction(model=self, initiator=initiator, target=target)
            # Add the new interaction to the feature interactions graph.
            self.feature_interactions.add_edge(initiator, target, interaction=interaction)
            print(f"new interaction {interaction.db_id} {interaction}")  # Log the creation of the new interaction.

            # Update interactions for roles affected by the new interaction.
            affected_roles = [
                role for role in self.roles_dict.values()
                if any(f in role.features for f in [initiator, target])
            ]
            for role in affected_roles:
                role.interactions = role.get_interactions()  # Refresh interactions for each affected role.

    def remove_feature(self, feature: Feature) -> None:
        # Remove a specified feature from the model.
        print(f"Removing feature {feature.db_id} {feature}")  # Log feature removal.
        # Record the feature removal in the changes database.
        feature_changes_row = (self.spacetime_dict["world"], feature.db_id, "removed")
        self.db_rows['feature_changes'].append(feature_changes_row)

        # Identify incoming and outgoing interactions of the feature.
        in_edges = feature.in_edges()
        out_edges = feature.out_edges()
        # Remove the feature from the feature interactions graph.
        self.feature_interactions.remove_node(feature)

        # Determine features and roles affected by this removal.
        affected_features = [x.initiator for x in in_edges] + [x.target for x in out_edges]
        affected_roles = [role for role in self.roles_dict.values() if any(f in role.features for f in affected_features)]
        roles_to_remove = [role for role in self.roles_dict.values() if feature in role.features]

        # Update the cached payoffs and roles dictionary.
        sd = self.sites
        cache = self.cached_payoffs
        for role in roles_to_remove:
            # Clear phenotype cache for each role to be removed.
            pl = [p for s in sd for p in role.types[s].keys()]
            for phenotype in pl:
                keys = [k for k, v in cache.items() if v == phenotype]
                if phenotype in cache:
                    del cache[phenotype]
                for k in keys:
                    del cache[k][phenotype]
            del self.roles_dict[role.features]
        # Update affected roles.
        for role in affected_roles:
            role.update()

    def prune_features(self):
        # Prune features based on their activity.
        # Check and update empty steps for each feature.
        for feature in self.get_features_list():
            feature.check_empty()
            feature.prune_traits()

        # Identify features eligible for pruning based on timeout criteria.
        pruneable = [f for f in self.get_features_list() if f.empty_steps >= self.feature_timeout]
        # Remove each pruneable feature.
        for feature in pruneable:
            self.remove_feature(feature)

    def create_agent(self, utils: float = None) -> Agent:
        # Set default utility value for the agent if not provided.
        if utils is None:
            utils = self.base_agent_utils

        # Assign a random trait to the agent from available features.
        traits = {}
        agent_features = self.get_features_list()  # Get list of possible features for agents.
        feature = self.random.choice(agent_features)  # Randomly select one feature.
        traits[feature] = self.random.choice(feature.values)  # Assign a random value to the selected feature.

        # Create the agent with a unique ID, the model reference, its utility, and assigned traits.
        agent = Agent(
            unique_id=self.next_id(),
            model=self,
            utils=utils,
            traits=traits,
        )

        # Randomly determine the agent's position on the grid.
        x = self.random.randrange(self.grid_size)
        y = self.random.randrange(self.grid_size)
        # Place the agent on the grid at the chosen coordinates.
        self.grid.place_agent(agent, (x, y))

        return agent  # Return the newly created agent.

    def verify_shadow(self):
        # Verify that the shadow model's site statistics match those of the main model.
        shadow = self.shadow  # Reference to the shadow model.
        for pos in shadow.sites:
            # Compare each corresponding site in the shadow model and the main model.
            shadow_site = shadow.sites[pos]
            live = self.sites[pos]
            # Assert that the site metrics (died, moved in/out, born) are identical.
            assert shadow_site.died == live.died
            assert shadow_site.moved_in == live.moved_in
            assert shadow_site.moved_out == live.moved_out
            assert shadow_site.born == live.born

    def spacetime_enumerator(self) -> Iterator:
        # Generate a sequence of spacetime IDs for the model's steps and sites.
        max_steps = self.controller.max_steps  # Maximum number of steps in the simulation.
        sites = [(x, y) for _, x, y in self.grid.coord_iter()]  # List of grid coordinates.
        sites.append('world')  # Include the world as a whole.
        start = 1 + (max_steps * len(sites) * (self.world_id - 1))  # Calculate the starting ID.

        # Enumerate over all possible combinations of world ID, steps, and sites.
        return enumerate(
            [(self.world_id, step, str(site)) for step in range(max_steps + 1) for site in sites],
            start=start)

    def get_spacetime_dict(self) -> Dict[Union[str, tuple], int]:
        # Create a dictionary mapping spacetime coordinates to their IDs.
        num_sites = (self.grid.height * self.grid.width) + 1  # Total number of sites plus the world.
        # Retrieve the next set of spacetime entries.
        spacetimes = [(i, *v) for i, v in [next(self.spacetimes) for _ in range(num_sites)]]

        spacetime_dict = {}  # Initialize the spacetime dictionary.
        for s in spacetimes:
            # Append each spacetime entry to the database rows.
            self.db_rows['spacetime'].append(s)
            # Populate the spacetime dictionary, interpreting 'world' as a special case.
            if s[3] == "world":
                spacetime_dict["world"] = s[0]
            else:
                spacetime_dict[eval(s[3])] = s[0]
        return spacetime_dict

    def get_db_ids_dict(self):
        # Initialize a dictionary to store database IDs for various tables.
        db_ids = {}
        for table in ['features', 'interactions', 'traits']:
            # Retrieve the dataframe for each table and calculate the starting ID.
            df = self.network_dfs[table]
            db_ids[table] = self.network_id  # Base ID from the network ID.
            db_ids[table] += self.controller.total_networks * df.shape[0]  # Adjust ID based on the number of entries.

        return db_ids  # Return the dictionary containing database IDs for each table.

    def get_db_rows_dict(self) -> Dict[str, List[Tuple[Any]]]:
        # Create a dictionary to hold lists of database rows for different database tables.
        return {
            'features': [],
            'interactions': [],
            'traits': [],
            'feature_changes': [],
            'trait_changes': [],
            'payoffs': [],
            'model_vars': [],
            'phenotypes': [],
            'demographics': [],
            'environment': [],
            'spacetime': []
        }

    def database_update(self, override: bool = False) -> None:
        # Collect data at regular intervals or if override is True.
        if self.schedule.time % self.controller.data_interval == 0 or override:
            print("Recording")  # Log the start of the recording process.

            # Retrieve spacetime dictionary and database rows dictionary.
            sd = self.spacetime_dict
            rd = self.db_rows

            # Gather various data for recording.
            get_model_vars_row(self, sd, rd)  # collect model variables.
            get_phenotypes_rows(self, sd, rd)  # collect phenotypes for the main model.
            get_phenotypes_rows(self.shadow, sd, rd, True)  # collect phenotypes for the shadow model.
            get_sites_rows(self, sd, rd)  # collect site data.

            print(f"Recorded {datetime.now()}", flush=True)  # Log the completion of the recording process.

        # Write to the database at specified intervals or if override is True.
        if self.schedule.time % self.controller.db_interval == 0 or override:
            # Remove empty rows before writing to the database.
            for k in [k for k, v in rd.items() if len(v) == 0]:
                del rd[k]

            print(f"Writing to DB {datetime.now()}")  # Log the start of the database writing process.
            self.db.write_rows(rd)  # Write the collected data to the database.
            print(f"Write complete {datetime.now()}", flush=True)  # Log the completion of the writing process.

            # Reset the database rows dictionary for the next interval.
            self.db_rows = self.get_db_rows_dict()

    def step(self):
        # Perform a single step in the simulation.
        self.new = 0  # Reset the count of new payoffs.
        self.cached = 0  # Reset the count of cached payoffs.

        # Reset all sites in both the main model and the shadow model.
        for site in self.sites.values():
            site.reset()
        for site in self.shadow.sites.values():
            site.reset()

        self.schedule.step()  # Execute scheuduler step.

        # Reset the shadow model or verify its integrity based on the snap interval.
        if self.schedule.time % self.snap_interval == 0:
            self.shadow.reset_shadow()  # Reset the shadow model at specified intervals.
        else:
            self.verify_shadow()  # Verify the shadow model's integrity at other times.

        # Print a status report including time, network ID, world ID, and agent count.
        print(
            "N:{1}/{5}, W:{3}/{4} id={2}, Step:{0}/{6}, Pop:{7}".format(
                self.schedule.time,
                self.network_id,
                self.world_id,
                self.controller.world_num,
                self.controller.network_worlds,
                self.controller.total_networks,
                self.controller.max_steps,
                self.schedule.get_agent_count()
            )
        )
        print(role_dist(self))  # Print the distribution of roles in the model.

        self.prune_features()  # Remove inactive features from the model.
        self.database_update()  # Update the database with the current state of the model.
        self.spacetime_dict = self.get_spacetime_dict()  # Update the spacetime dictionary.

        # Check if all agents are dead and end the simulation if so.
        if self.schedule.get_agent_count() == 0:
            print("All the agents are dead... :(")
            self.running = False  # Stop the simulation.

    def cleanup(self):
        # Cleanup Features and Interactions stored in the nx.DiGraph.
        for feature in self.feature_interactions.nodes():
            feature.cleanup()
        for _, _, interaction in self.feature_interactions.edges(data='interaction'):
            interaction.cleanup()

        # Remove all nodes and edges from the graph.
        self.feature_interactions.clear()

        # Cleanup Roles, which are dependent on Features and Interactions.
        for role in self.roles.values():
            role.cleanup()

        # Cleanup Agents and Sites, which are interdependent.
        for agent in self.schedule.agents:
            agent.cleanup()
        for site in self.sites.values():
            site.cleanup()

        # Clear collections inside complex objects
        self.schedule.clear_schedule()
        self.grid.clear_grid()
        self.shadow.cleanup()
        self.feature_interactions.clear()

        # Clear dictionaries and lists
        self.sites.clear()
        self.cached_payoffs.clear()
        self.roles_dict.clear()
        self.spacetime_dict.clear()
        self.db_rows.clear()
        self.network_dfs.clear()
        self.db_ids.clear()

        # Remove references to complex objects and collections
        self.schedule = None
        self.grid = None
        self.shadow = None
        self.feature_interactions = None
