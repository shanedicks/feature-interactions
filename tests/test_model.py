import itertools
import random
import unittest
import pandas as pd
from unittest.mock import Mock, MagicMock, PropertyMock, call, create_autospec, patch
from model.agents import Agent, Site
from model.features import Feature, Interaction
from model.model import ListDict, ListDictMultiGrid, SampledActivation, Shadow, World

class TestListDict(unittest.TestCase):

    def setUp(self):
        # Create a ListDict instance
        self.list_dict = ListDict()

        # Create mock agents
        self.mock_agent1 = Mock(spec=Agent)
        self.mock_agent2 = Mock(spec=Agent)
        self.mock_agent3 = Mock(spec=Agent)

        # Add mock agents to the ListDict
        self.list_dict.add_agent(self.mock_agent1)
        self.list_dict.add_agent(self.mock_agent2)

    def test_add_agent(self):
        # Test adding an agent that's already in the ListDict
        existing_agent_count = len(self.list_dict)
        self.list_dict.add_agent(self.mock_agent1)
        self.assertEqual(len(self.list_dict), existing_agent_count, "Adding an existing agent should not change the count")

        # Test adding a new agent
        self.list_dict.add_agent(self.mock_agent3)
        self.assertEqual(len(self.list_dict), existing_agent_count + 1, "Adding a new agent should increase the count by 1")
        self.assertIn(self.mock_agent3, self.list_dict.agent_to_index, "New agent should be in agent_to_index")
        self.assertEqual(self.list_dict.agent_to_index[self.mock_agent3], existing_agent_count, "New agent index should be correct")
        self.assertIn(self.mock_agent3, self.list_dict.agents, "New agent should be in agents list")

    def test_remove_agent(self):
        self.list_dict.add_agent(self.mock_agent3)

        # Test removing an agent that's not the last in the list (middle agent)
        agent_to_remove = self.mock_agent2
        last_agent_before_removal = self.mock_agent3
        index_of_agent_to_remove = self.list_dict.agent_to_index[agent_to_remove]

        self.list_dict.remove_agent(agent_to_remove)
        self.assertNotIn(agent_to_remove, self.list_dict.agent_to_index, "Removed agent should not be in agent_to_index")
        self.assertNotIn(agent_to_remove, self.list_dict.agents, "Removed agent should not be in agents list")
        self.assertEqual(self.list_dict.agents[index_of_agent_to_remove], last_agent_before_removal, "Last agent should move to the removed agent's position")
        self.assertEqual(self.list_dict.agent_to_index[last_agent_before_removal], index_of_agent_to_remove, "Index of last agent should be updated")

        # Test removing the last agent in the list
        last_agent = self.list_dict.agents[-1]
        self.list_dict.remove_agent(last_agent)
        self.assertNotIn(last_agent, self.list_dict.agent_to_index, "Removed last agent should not be in agent_to_index")
        self.assertNotIn(last_agent, self.list_dict.agents, "Removed last agent should not be in agents list")

        # Confirm the list length is as expected after removing two agents
        self.assertEqual(len(self.list_dict.agents), 1, "Agents list should have one agent after removing two agents")


class TestListDictMultiGrid(unittest.TestCase):

    def setUp(self):
        # Mock necessary attributes and methods for the Grid parent class
        self.grid_width = 3
        self.grid_height = 1
        self.grid_torus = True

        # Initialize ListDictMultiGrid
        self.list_dict_multigrid = ListDictMultiGrid(self.grid_width, self.grid_height, self.grid_torus)

        # Create mock agents
        self.mock_agent = create_autospec(Agent, instance=True)

        # Set up coordinates
        self.coord = (0, 0)

    def test_add_agent(self):
        self.list_dict_multigrid._place_agent = Mock()
        self.list_dict_multigrid._place_agent(self.coord, self.mock_agent)
        self.list_dict_multigrid._place_agent.assert_called_once_with(self.coord, self.mock_agent)

    def test_remove_agent(self):
        self.list_dict_multigrid._remove_agent = Mock()
        self.list_dict_multigrid._remove_agent(self.coord, self.mock_agent)
        self.list_dict_multigrid._remove_agent.assert_called_once_with(self.coord, self.mock_agent)


class TestSampledActivation(unittest.TestCase):

    def setUp(self):
        # Mock the model
        self.mock_model = Mock(spec=World)
        self.mock_model.active_pop_limit = 5
        self.mock_model.mortality = 0.2
        self.mock_model.random = Mock()
        self.mock_model.random.sample.side_effect = lambda population, k: list(population)[:k]  # Convert to list and return the first k elements
        self.mock_model.random.random.side_effect = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # Predefined mortality chances
        
        # Create a list of mock agents
        self.mock_agents = [Mock(spec=Agent, unique_id=i) for i in range(10)]
        
        # Assign each agent's step and die method
        for agent in self.mock_agents:
            agent.step = Mock()
            agent.die = Mock()
        
        # Instantiate SampledActivation
        self.scheduler = SampledActivation(self.mock_model)

        # Add agents to the scheduler
        for agent in self.mock_agents:
            self.scheduler.add(agent)

    def test_agent_buffer(self):
        # Define the buffer size
        buffer_size = 5

        # Call the agent_buffer method
        agent_buffer = list(self.scheduler.agent_buffer(buffer_size))

        # Assert that the returned buffer is of the correct size
        self.assertEqual(len(agent_buffer), buffer_size, "Agent buffer should return the correct number of agents.")

        # Assert that the agents are randomly selected
        expected_agents = self.mock_agents[:buffer_size]
        self.assertEqual(agent_buffer, expected_agents, "Agent buffer should return a random subset of agents.")

        # Test the case when buffer size is larger than the number of agents
        large_buffer_size = 15
        agent_buffer_large = list(self.scheduler.agent_buffer(large_buffer_size))
        self.assertEqual(len(agent_buffer_large), len(self.mock_agents), "Agent buffer should not exceed the total number of agents.")
        self.assertEqual(set(agent_buffer_large), set(self.mock_agents), "Agent buffer should return all agents when the buffer size exceeds the number of agents.")

    def test_step(self):
        # Set the mortality rate for the model
        self.mock_model.mortality = 0.2  # 20% mortality rate
        self.mock_model.active_pop_limit = 5  # Limit the number of active agents per step
        initial_agent_count = len(self.scheduler.agents)

        # Create a mock for the die method
        for agent in self.scheduler.agents:
            agent.die = Mock()

        # Call the step method
        self.scheduler.step()

        # Assert that the step and time counters were incremented
        self.assertEqual(self.scheduler.steps, 1, "Step counter should be incremented.")
        self.assertEqual(self.scheduler.time, 1, "Time counter should be incremented.")

        # Assert that agents were stepped
        for agent in self.scheduler.agent_buffer(self.mock_model.active_pop_limit):
            agent.step.assert_called_once()

        # Assert that the mortality rate was applied
        expected_deaths = 2
        actual_deaths = sum(agent.die.call_count for agent in self.scheduler.agents)
        self.assertEqual(actual_deaths, expected_deaths, f"{expected_deaths} agents should die from mortality, but {actual_deaths} died.")

        # Assert that the schedule is cleared if all agents die
        if actual_deaths == initial_agent_count:
            self.assertEqual(len(self.scheduler.agents), 0, "All agents should be removed if they all die.")


class TestShadow(unittest.TestCase):

    def setUp(self):
        # Start the patch for Agent.get_role
        self.patcher = patch('model.agents.Agent.get_role')
        self.mock_get_role = self.patcher.start()
        self.mock_get_role.return_value = Mock()
        mock_role = Mock()
        mock_role.types = {(0,0): {}, (0,1): {}, (1,0): {}, (1,1): {}}  # Mock the 'types' attribute to behave like a dict
        self.mock_get_role.return_value = mock_role
        # Mock the World model
        self.mock_world_model = Mock(spec=World)
        self.mock_world_model.grid_size = 2
        self.mock_world_model.base_agent_utils = 1.0
        self.mock_world_model.base_env_utils = 1.0
        self.mock_world_model.repr_multi = 0.5
        self.mock_world_model.mortality = 0.1
        self.mock_world_model.move_chance = 0.2
        self.mock_world_model.trait_mutate_chance = 0.01
        self.mock_world_model.feature_mutate_chance = 0.01
        self.mock_world_model.trait_create_chance = 0.1
        self.mock_world_model.feature_create_chance = 0.1
        self.mock_world_model.feature_gain_chance = 0.1
        self.mock_world_model.site_pop_limit = 5
        self.mock_world_model.pop_cost_exp = 2.0
        self.mock_world_model.feature_interactions = Mock()
        self.mock_world_model.random = Mock()
        self.mock_world_model.next_id = Mock(side_effect=itertools.count(start=1).__next__)

        self.trait_keys = ['F', 'G', 'H']
        self.trait_values = ['a', 'b', 'c']

        self.mock_world_model.sites = {
            (0, 0): Mock(spec=Site, traits={'A': 'a'}, agents=[Mock(spec=Agent, utils=0.5, traits={random.choice(self.trait_keys): random.choice(self.trait_values)}) for _ in range(3)]),
            (0, 1): Mock(spec=Site, traits={'B': 'a'}, agents=[Mock(spec=Agent, utils=0.6, traits={random.choice(self.trait_keys): random.choice(self.trait_values)}) for _ in range(4)]),
            (1, 0): Mock(spec=Site, traits={'C': 'a'}, agents=[Mock(spec=Agent, utils=0.7, traits={random.choice(self.trait_keys): random.choice(self.trait_values)}) for _ in range(2)]),
            (1, 1): Mock(spec=Site, traits={'D': 'a'}, agents=[Mock(spec=Agent, utils=0.8, traits={random.choice(self.trait_keys): random.choice(self.trait_values)}) for _ in range(5)])
        }

        self.mock_world_model.get_features_list = Mock(return_value=['A', 'B', 'C'])

        # Create an instance of the Shadow class with the mocked World model
        self.shadow = Shadow(self.mock_world_model)

    def test_init(self):
        # Check if the Shadow model's attributes match those of the World model
        self.assertEqual(self.shadow.base_agent_utils, self.mock_world_model.base_agent_utils)
        self.assertEqual(self.shadow.base_env_utils, self.mock_world_model.base_env_utils)
        self.assertEqual(self.shadow.repr_multi, self.mock_world_model.repr_multi)
        self.assertEqual(self.shadow.mortality, self.mock_world_model.mortality)
        self.assertEqual(self.shadow.move_chance, self.mock_world_model.move_chance)
        self.assertEqual(self.shadow.trait_mutate_chance, self.mock_world_model.trait_mutate_chance)
        self.assertEqual(self.shadow.feature_mutate_chance, self.mock_world_model.feature_mutate_chance)
        self.assertEqual(self.shadow.trait_create_chance, self.mock_world_model.trait_create_chance)
        self.assertEqual(self.shadow.feature_create_chance, self.mock_world_model.feature_create_chance)
        self.assertEqual(self.shadow.feature_gain_chance, self.mock_world_model.feature_gain_chance)
        self.assertEqual(self.shadow.site_pop_limit, self.mock_world_model.site_pop_limit)
        self.assertEqual(self.shadow.pop_cost_exp, self.mock_world_model.pop_cost_exp)
        self.assertEqual(self.shadow.feature_interactions, self.mock_world_model.feature_interactions)
        self.assertEqual(self.shadow.random, self.mock_world_model.random)

        # Check if sites from the World model are duplicated in the Shadow model
        for pos, site in self.mock_world_model.sites.items():
            self.assertIn(pos, self.shadow.sites)
            self.assertIsNot(self.shadow.sites[pos], site)  # Ensure it's a new instance
            self.assertEqual(self.shadow.sites[pos].traits, site.traits)

        # Check if agents from the World model are duplicated in the Shadow model
        for pos, mock_site in self.mock_world_model.sites.items():
            shadow_site = self.shadow.sites[pos]
            self.assertEqual(len(mock_site.agents), len(shadow_site.agents), f"Number of agents in site {pos} should match")
            
            for mock_agent, shadow_agent in zip(mock_site.agents, shadow_site.agents):
                self.assertEqual(mock_agent.traits, shadow_agent.traits, "Agent traits should match at site {pos}")

        # Check if the grid was created correctly
        self.assertIsInstance(self.shadow.grid, ListDictMultiGrid)

        # Check if the roles dictionary is initialized correctly
        self.assertEqual(self.shadow.roles_dict, {})

    def test_reset_shadow(self):
        # Add new agents to mock_world_model sites
        for pos, site in self.mock_world_model.sites.items():
            site.agents.append(Mock(spec=Agent, utils=0.5, traits={random.choice(self.trait_keys): random.choice(self.trait_values)}))

        # Reset the shadow
        self.shadow.reset_shadow()

        # Check if agents from the World model are duplicated in the Shadow model
        for pos, mock_site in self.mock_world_model.sites.items():
            shadow_site = self.shadow.sites[pos]
            self.assertEqual(len(mock_site.agents), len(shadow_site.agents), f"Number of agents in site {pos} should match")
            
            for mock_agent, shadow_agent in zip(mock_site.agents, shadow_site.agents):
                self.assertEqual(mock_agent.traits, shadow_agent.traits, "Agent traits should match at site {pos}")

    def tearDown(self):
        # Stop the patcher
        self.patcher.stop()


class TestWorld(unittest.TestCase):

    def setUp(self):
        # Mock the dependencies
        self.mock_controller = MagicMock(max_steps=10, total_networks=10, data_interval=5, db_interval=10)
        self.mock_db_manager = MagicMock()
        self.mock_db_manager.get_network_dataframes.return_value = MagicMock()
        self.mock_db_manager.write_rows.return_value = MagicMock()
        self.mock_db_manager.get_next_feature.return_value = None
        self.mock_db_manager.get_feature_interactions.return_value = []
        self.mock_controller.db_manager = self.mock_db_manager

        # Dictionary to hold the patches
        self.patches = {
            'Agent': patch('model.model.Agent'),
            'Site': patch('model.model.Site'),
            'Feature': patch('model.model.Feature'),
            'Interaction': patch('model.model.Interaction'),
            'ListDictMultiGrid': patch('model.model.ListDictMultiGrid'),
            'SampledActivation': patch('model.model.SampledActivation'),
            'nx.DiGraph': patch('networkx.DiGraph'),
            'Shadow': patch('model.model.Shadow'),
            'random.choice': patch.object(random.Random, 'choice'),
        }
        
        # Initialize the mocks
        self.mocks = {k: v.start() for k, v in self.patches.items()}

        # Parameters for initializing the World class (example values, adjust as needed)
        self.params = {
            "controller": self.mock_controller,
            "world_id": 1,
            "network_id": 1,
            "init_env_features": 5,
            "init_agent_features": 3,
            "max_feature_interactions": 5,
            "trait_payoff_mod": 0.5,
            "anchor_bias": 0.5,
            "payoff_bias": 0.5,
            "trait_mutate_chance": 0.1,
            "trait_create_chance": 0.1,
            "feature_mutate_chance": 0.1,
            "feature_create_chance": 0.1,
            "feature_gain_chance": 0.1,
            "feature_timeout": 10,
            "trait_timeout": 10,
            "init_agents": 25,
            "base_agent_utils": 0.0,
            "base_env_utils": 10.0,
            "total_pop_limit": 100,
            "pop_cost_exp": 2,
            "feature_cost_exp": 1.5,
            "grid_size": 2,
            "repr_multi": 2,
            "mortality": 0.1,
            "move_chance": 0.5,
            "snap_interval": 10,
            "target_sample": 10,
            "active_pop_limit": 50,
        }

        # Initialize the World instance with mocked dependencies
        self.world = World(**self.params)

        grid_size = self.params['grid_size']
        self.world.grid.coord_iter.return_value = [
            ([], x, y) for x in range(grid_size) for y in range(grid_size)
        ]
        self.world.grid.height = self.world.grid.width = grid_size

        self.mock_features = [
            Mock(spec=Feature, values=['a', 'b', 'c']),
            Mock(spec=Feature, values=['a', 'b', 'c']),
            Mock(spec=Feature, values=['a', 'b', 'c']),
            Mock(spec=Feature, values=['a', 'b', 'c']),
            Mock(spec=Feature, values=['a', 'b', 'c']),
            Mock(spec=Feature, values=['a', 'b', 'c']),
        ]
        
        names = ["A", "B", "C", "F", "G", "H"]
        envs = [True, True, True, False, False, False]
        db_ids = [1, 2, 3, 4, 5, 6]
        
        for mock_feature, name, env, db_id in zip(self.mock_features, names, envs, db_ids):
            type(mock_feature).name = PropertyMock(return_value=name)
            type(mock_feature).env = PropertyMock(return_value=env)
            type(mock_feature).db_id = PropertyMock(return_value=db_id)

        # Mock methods and attributes of the created mock objects
        self.mocks['nx.DiGraph'].return_value.nodes.return_value = MagicMock()
        self.mocks['nx.DiGraph'].return_value.edges.return_value = MagicMock()

    @patch('model.model.World.print_report')
    @patch('model.model.World.get_db_rows_dict')
    @patch('model.model.World.spacetime_enumerator')
    @patch('model.model.World.get_spacetime_dict')
    @patch('model.model.World.get_db_ids_dict')
    @patch('model.model.World.get_or_create_init_features_network')
    @patch('model.model.World.create_sites')
    @patch('model.model.World.create_init_agents')
    def test_init(self, create_init_agents, create_sites, get_or_create_init_features_network, get_db_ids_dict, get_spacetime_dict, spacetime_enumerator, get_db_rows_dict, print_report):
        self.mocks['Shadow'].reset_mock()
        self.mocks['ListDictMultiGrid'].reset_mock()
        self.mocks['SampledActivation'].reset_mock()
        world = World(**self.params)

        self.mocks['ListDictMultiGrid'].assert_called_once_with(self.params['grid_size'], self.params['grid_size'], True)
        self.mocks['SampledActivation'].assert_called_once_with(world)
        get_db_rows_dict.assert_called_once()
        spacetime_enumerator.assert_called_once()
        get_spacetime_dict.assert_called_once()
        get_db_ids_dict.assert_called_once()
        get_or_create_init_features_network.assert_called_once_with(self.params['init_env_features'], self.params['init_agent_features'])
        create_sites.assert_called_once()
        create_init_agents.assert_called_once_with(self.params['init_agents'])
        self.mocks['Shadow'].assert_called_once_with(model=world)
        print_report.assert_called_once()
        self.assertTrue(world.running)

    def test_get_features(self):
        self.world.feature_interactions = self.mocks['nx.DiGraph']()
        self.world.feature_interactions.nodes.__iter__.return_value = self.mock_features

        # Testing get_features_list
        env_features = self.world.get_features_list(env=True)
        non_env_features = self.world.get_features_list()
        self.assertEqual(env_features, self.mock_features[:3])
        self.assertEqual(non_env_features, self.mock_features[3:])

        # Testing get_feature_by_name
        feature = self.world.get_feature_by_name("A")
        self.assertEqual(feature, self.mock_features[0])

        no_feature = self.world.get_feature_by_name("NonExistent")
        self.assertIsNone(no_feature)

        # Testing get_feature_by_id
        feature = self.world.get_feature_by_id(1)
        self.assertEqual(feature, self.mock_features[0])

        no_feature = self.world.get_feature_by_id(20)
        self.assertIsNone(no_feature)

    def test_get_or_create_init_features_network(self):
        num_env = self.params['init_env_features']
        num_agent = self.params['init_agent_features']

        # Mock the get_features_list and next_feature methods of World
        self.world.get_features_list = MagicMock()
        self.world.next_feature = MagicMock()

        # Call the method under test
        self.world.get_or_create_init_features_network(num_env, num_agent)

        # Check if next_feature is called the correct number of times with correct arguments
        self.assertEqual(self.world.next_feature.call_count, num_env + num_agent)
        self.world.next_feature.assert_has_calls([call(env=True)] * num_env + [call()] * num_agent)

        # Check if get_features_list is called correctly
        self.assertEqual(self.world.get_features_list.call_count, 2)
        self.world.get_features_list.assert_any_call(env=True)
        self.world.get_features_list.assert_any_call()

    def test_create_sites(self):
        grid_size = self.params['grid_size']
        self.world.create_sites()
        expected_sites = {
            (x, y): self.mocks['Site'].return_value
            for x in range(grid_size) for y in range(grid_size)
        }
        self.assertEqual(self.world.sites, expected_sites)

    def test_create_init_agents(self):
        init_agents = self.params['init_agents']
        self.world.create_agent = MagicMock(side_effect=[Mock() for _ in range(init_agents)])
        self.world.schedule.reset_mock()
        self.world.schedule.agents = []

        # Side effect for schedule.add to append the agent to schedule.agents
        self.world.schedule.add.side_effect = lambda agent: self.world.schedule.agents.append(agent)

        self.world.create_init_agents(init_agents)

        # Assert create_agent is called the correct number of times
        self.assertEqual(self.world.create_agent.call_count, init_agents)

        # Assert schedule.add is called the correct number of times
        self.assertEqual(self.world.schedule.add.call_count, init_agents)

        # Assert the length of schedule.agents matches the number of initial agents
        self.assertEqual(len(self.world.schedule.agents), init_agents)

        # Assert get_site is called once for each agent
        for agent in self.world.schedule.agents:
            agent.get_site.assert_called_once()

        # Assert each agent's site is set correctly
        for agent in self.world.schedule.agents:
            self.assertEqual(agent.site, agent.get_site.return_value)

    def test_next_feature(self):
        # Mocking db.get_next_feature to return a dict on the first call and None afterwards
        mock_restored_feature = {'db_id': 1}
        self.world.db.get_next_feature = Mock(side_effect=[mock_restored_feature, None, None])

        # Mocking restore_feature and create_feature
        self.world.restore_feature = MagicMock(return_value=Mock(db_id=1))
        self.world.create_feature = MagicMock(side_effect=[Mock(db_id=i) for i in range(2, 4)])
        self.world.next_db_id = Mock()
        self.world.db_rows['feature_changes'] = []

        # Call next_feature and assert restore_feature was called correctly
        feature1 = self.world.next_feature()
        self.world.restore_feature.assert_called_once_with(mock_restored_feature)
        self.world.next_db_id.assert_called_once_with('features')
        self.world.create_feature.assert_not_called()

        # Reset mocks for create_feature and next_db_id
        self.world.create_feature.reset_mock()
        self.world.next_db_id.reset_mock()

        # Call next_feature and assert create_feature was called correctly with env=False
        feature2 = self.world.next_feature()
        self.world.create_feature.assert_called_once_with(env=False)
        self.world.next_db_id.assert_not_called()

        # Reset mocks for create_feature
        self.world.create_feature.reset_mock()

        # Call next_feature(env=True) and assert create_feature was called correctly with env=True
        feature3 = self.world.next_feature(env=True)
        self.world.create_feature.assert_called_once_with(env=True)

        # Assert that db_rows['features'] matches the expected_rows
        expected_rows = [
            (self.world.spacetime_dict["world"], 1, "added"),  # After restoring the first feature from the database
            (self.world.spacetime_dict["world"], 2, "added"),  # After creating the second feature (env=False)
            (self.world.spacetime_dict["world"], 3, "added")   # After creating the third feature (env=True)
        ]
        self.assertEqual(expected_rows, self.world.db_rows['feature_changes'])


    @patch('model.model.Feature')
    def test_restore_feature(self, mock_feature):
        # Configure the mocks
        self.world.feature_interactions.reset_mock()
        self.world.next_feature_id = Mock()
        self.world.restore_interactions = Mock()
        self.world.next_feature_id = Mock(return_value=1)
        feature_dict = {'feature_id': 10, 'env': False}
        restored_feature = MagicMock(spec=Feature, feature_id=1, db_id=10, env=False)
        mock_feature.return_value = restored_feature

        # Call the method
        result_feature = self.world.restore_feature(feature_dict)

        # Assertions
        mock_feature.assert_called_once_with(feature_id=1, db_id=10, model=self.world, env=False)
        self.assertEqual(self.world.current_feature_db_id, 10)
        self.world.feature_interactions.add_node.assert_called_once_with(restored_feature)
        self.world.next_feature_id.assert_called_once()
        self.world.restore_interactions.assert_called_once_with(restored_feature)
        self.assertEqual(result_feature, restored_feature)

    @patch('model.model.Feature')
    def test_create_feature(self, mock_feature):
        # Mock the Feature class
        mock_feature_instance = MagicMock()
        def configure_mock_feature(*args, **kwargs):
            nonlocal mock_feature_instance
            mock_feature_instance.configure_mock(**kwargs)
            return mock_feature_instance

        mock_feature.side_effect = configure_mock_feature

        # Mock and set up necessary attributes and methods
        self.world.next_feature_id = Mock(side_effect=[1,2,3])
        self.world.feature_interactions = Mock()
        self.world.feature_interactions.add_node = Mock()
        self.world.feature_interactions.number_of_nodes = Mock(return_value=10)
        self.world.create_interaction = Mock()
        self.world.max_feature_interactions = 5
        self.world.random = Mock()
        self.world.random.randrange = Mock(return_value=3)

        # Test with env=False
        returned_feature_env_false = self.world.create_feature(env=False)
        self.world.next_feature_id.assert_called_once()
        self.world.feature_interactions.add_node.assert_called_once_with(mock_feature_instance)
        self.world.random.randrange.assert_called_once_with(1, min(self.world.max_feature_interactions, 10))
        self.assertEqual(self.world.create_interaction.call_count, 3)
        self.assertEqual(returned_feature_env_false, mock_feature_instance)

        # Reset mocks to test with env=True
        self.world.next_feature_id.reset_mock()
        self.world.feature_interactions.add_node.reset_mock()
        self.world.create_interaction.reset_mock()
        self.world.random.randrange.reset_mock()

        # Test with env=True
        returned_feature_env_true = self.world.create_feature(env=True)
        self.world.next_feature_id.assert_called_once()
        self.world.feature_interactions.add_node.assert_called_once_with(mock_feature_instance)
        self.world.random.randrange.assert_not_called()
        self.world.create_interaction.assert_not_called()
        self.assertEqual(returned_feature_env_true, mock_feature_instance)

    def test_restore_interactions(self):
        # Mock necessary data for interactions
        mock_interaction_data = [
            {'target': 2, 'db_id': 101, 'i_anchor': 0.0, 't_anchor': 0.1},
            {'target': 3, 'db_id': 102, 'i_anchor': 0.2, 't_anchor': 0.3}
        ]

        # Mock world.feature_interactions.nodes and edges
        self.world.feature_interactions.edges = []
        self.world.feature_interactions.nodes.__iter__.return_value = self.mock_features
        
        # Mock the add_edge method to append a tuple representing an edge to mock_edges
        self.world.feature_interactions.add_edge.side_effect = lambda u, v, interaction: self.world.feature_interactions.edges.append((u, v, interaction))
        
        # Configure the mock DB to return the mock interaction data
        self.mock_db_manager.get_feature_interactions.return_value = mock_interaction_data

        # Mock the Interaction constructor
        self.mocks['Interaction'].side_effect = lambda model, initiator, target, db_id, restored, anchors: Mock(
            db_id=db_id, initiator=initiator, target=target, anchors=anchors)

        # Mock the roles_dict in the world object
        role_features = [frozenset([self.mock_features[3], self.mock_features[4]]), frozenset([self.mock_features[3]])]
        mock_roles = [Mock(features=features) for features in role_features]
        self.world.roles_dict = {features: role for features, role in zip(role_features, mock_roles)}
        
        # Mock the update method for each mock role
        for role in mock_roles:
            role.update = MagicMock()

        # Select a mock feature as the initiator
        initiator = self.mock_features[3]

        # Call the method under test
        self.world.restore_interactions(initiator)

        # Assertions
        # Check if the interactions are restored correctly
        self.assertEqual(len(self.world.feature_interactions.edges), 2, "Not all interactions were restored.")

        # Check if the Interaction objects were created correctly
        self.mocks['Interaction'].assert_has_calls([
            call(model=self.world, initiator=initiator, target=self.mock_features[1], db_id=101, restored=True, anchors={'i': 0.0, 't': 0.1}),
            call(model=self.world, initiator=initiator, target=self.mock_features[2], db_id=102, restored=True, anchors={'i': 0.2, 't': 0.3})
        ], any_order=True)

        # Check if the roles' update method was called for affected roles
        for role in mock_roles:
            role.update.assert_called()

    def test_create_interaction(self):
        # Mock necessary attributes
        initiator = self.mock_features[3]
        existing_target = self.mock_features[0]
        other_feature = self.mock_features[1]

        # Setup the feature interactions
        self.world.feature_interactions.neighbors.return_value = [existing_target]
        self.world.feature_interactions.nodes.__iter__.return_value = self.mock_features
        # Capture the arguments passed to self.world.random.choice
        target_choices_passed_in = None
        def capture_choices(args):
            nonlocal target_choices_passed_in
            target_choices_passed_in = args
            return args[0]  # return the first element as the chosen target for simplicity

        self.world.random.choice.side_effect = capture_choices
        
        # Mock Interaction constructor
        mock_interaction = MagicMock()
        self.mocks['Interaction'].return_value = mock_interaction
        
        # Mock roles_dict and affected roles
        mock_role1 = MagicMock(features=frozenset([initiator, other_feature]))
        mock_role2 = MagicMock(features=frozenset([existing_target]))
        self.world.roles_dict = {mock_role1.features: mock_role1, mock_role2.features: mock_role2}
        
        # Call the method under test
        self.world.create_interaction(initiator)

        # Assert target_choices construction
        expected_target_choices = self.mock_features[1:]  
        self.assertEqual(target_choices_passed_in, expected_target_choices, "target_choices not constructed correctly")

        # Assert the new interaction was added to the graph
        self.world.feature_interactions.add_edge.assert_called_once_with(initiator, other_feature, interaction=mock_interaction)

        # Assert the affected role's update method was called
        mock_role1.update.assert_called_once()
        mock_role2.update.assert_not_called()

    def test_remove_feature(self):
        # Mock the feature to be removed
        feature_to_remove = self.mock_features[3]
        all_other_features = self.mock_features[:3] + self.mock_features [4:]

        # Mock in_edges and out_edges of the feature
        in_edges = [Mock(initiator=feature) for feature in self.mock_features[4:]]
        out_edges = [Mock(target=feature) for feature in all_other_features]

        # Set the return values of in_edges and out_edges methods
        feature_to_remove.in_edges.return_value = in_edges
        feature_to_remove.out_edges.return_value = out_edges

        # Mock roles and their 'features' attribute
        role1 = MagicMock(features=frozenset([self.mock_features[3]]))
        role2 = MagicMock(features=frozenset([self.mock_features[4]]))
        role3 = MagicMock(features=frozenset(self.mock_features[4:]))

        # Mock types for each role with a few valid phenotypes
        role1.types = {(0, 0): {'F.a': 1, 'F.b': 2}}
        role2.types = {(0, 0): {'G.b': 1}}
        role3.types = {(0, 0): {'G.d': 1, 'H.a': 2}}

        # Mock roles_dict in the world object
        self.world.roles_dict = {
            role1.features: role1,
            role2.features: role2,
            role3.features: role3
        }

        # Mock self.world.sites to give us at least one key to use for looking up role.types[sites key]
        self.world.sites = {(0, 0): Mock()}

        # Mock self.world.cached_payoffs of the form {initiator_phenotype: {target_phenotype: payoff 2-tuple}}
        self.world.cached_payoffs = {
            'F.a': {'G.b': (0.5, 1.0), 'H.a': (0.3, 0.7)},
            'G.b': {'F.a': (0.6, 0.2)},
            'G.d': {'H.a': (0.4, 0.8)},
            'H.a': {'G.d': (0.2, 0.5)},
            'F.b': {'G.b': (0.5, 1.0)}
        }

        # Pre-remove assertions (to confirm initial state before removal)
        for phenotype in role1.types[(0, 0)].keys():
            self.assertIn(phenotype, self.world.cached_payoffs, "Phenotype should exist in cached_payoffs before removal")

        # Call the method under test
        self.world.remove_feature(feature_to_remove)

        # Check if the feature was removed from feature_interactions
        self.world.feature_interactions.remove_node.assert_called_with(feature_to_remove)

        # Check if the feature_changes_row is correctly appended
        expected_feature_changes_row = (self.world.spacetime_dict["world"], feature_to_remove.db_id, "removed")
        self.assertIn(expected_feature_changes_row, self.world.db_rows['feature_changes'])

        # Check if cached_payoffs has been updated correctly
        for phenotype in role1.types[(0, 0)].keys():
            self.assertNotIn(phenotype, self.world.cached_payoffs, "Phenotype should be removed from cached_payoffs after removal")
            for target_phenotype in self.world.cached_payoffs.keys():
                self.assertNotIn(phenotype, self.world.cached_payoffs[target_phenotype], "Phenotype should not be a target in cached_payoffs after removal")

        # Check if the correct roles_to_remove are identified and removed
        self.assertNotIn(role1.features, self.world.roles_dict)
        self.assertIn(role2.features, self.world.roles_dict)
        self.assertIn(role3.features, self.world.roles_dict)

        # Check if the correct affected_roles are identified and their update method was called
        role1.update.assert_not_called()
        role2.update.assert_called()
        role3.update.assert_called()

    def test_prune_features(self):
        # Set the feature_timeout
        self.world.feature_timeout = 10

        agent_features = self.mock_features[3:]

        # Mock feature_interactions.nodes
        self.world.feature_interactions.nodes.__iter__.return_value = self.mock_features

        # Configure mock_features with varying empty_steps
        for i, feature in enumerate(agent_features):
            type(feature).empty_steps = PropertyMock(return_value=i * 5)
            feature.check_empty = MagicMock()
            feature.prune_traits = MagicMock()
        
        # Mock the remove_feature method
        self.world.remove_feature = MagicMock()

        # Call prune_features
        self.world.prune_features()

        # Assertions
        # check_empty and prune_traits should be called for each non-env feature
        for feature in agent_features:
            feature.check_empty.assert_called_once()
            feature.prune_traits.assert_called_once()

        # remove_feature should be called for each non-env feature with empty_steps >= feature_timeout
        for feature in agent_features:
            if feature.empty_steps >= self.world.feature_timeout:
                self.world.remove_feature.assert_any_call(feature)
        
        # Check the exact number of calls to remove_feature
        expected_prune_count = sum(f.empty_steps >= self.world.feature_timeout for f in agent_features)
        self.assertEqual(self.world.remove_feature.call_count, expected_prune_count)

    def test_create_agent(self):
        # Mock the random.choice method for consistent testing
        agent_feature = self.mock_features[3]
        self.world.next_id = Mock(return_value=1)
        self.world.random.choice = MagicMock(side_effect=[agent_feature, 'a'])
        self.world.random.randrange = MagicMock(side_effect=[0, 0])  # Mock grid coordinates

        # Mock Agent class
        self.patches['Agent'].return_value = Mock(spec=Agent)

        # Call the create_agent method
        agent = self.world.create_agent()

        # Assertions
        # Check that the agent was created with the right traits
        self.mocks['Agent'].assert_called_with(
            unique_id=1,
            model=self.world,
            utils=self.world.base_agent_utils,
            traits={agent_feature: 'a'}
        )

        # Check that the agent is placed at the right coordinates
        self.world.grid.place_agent.assert_called_with(agent, (0, 0))

    def test_spacetime_enumerator(self):
        grid_size = self.params['grid_size']
        # Generate the sequence
        spacetime_sequence = list(self.world.spacetime_enumerator())

        # Expected sequence
        expected_sequence = []
        sites = [(x, y) for x in range(grid_size) for y in range(grid_size)] + ['world']
        for step in range(self.world.controller.max_steps + 1):
            for site in sites:
                expected_sequence.append((self.world.world_id, step, str(site)))

        # Check if the spacetime_sequence matches the expected_sequence
        start = 1
        for i, item in enumerate(spacetime_sequence):
            self.assertEqual(item[0], start + i)
            self.assertEqual(item[1], expected_sequence[i])

        # Also, check the length of the sequence
        self.assertEqual(len(spacetime_sequence), len(expected_sequence))

    def test_get_spacetime_dict(self):
        # Set up a clean spacetimes and db_rows['spacetime']
        self.world.spacetimes = self.world.spacetime_enumerator()
        self.world.db_rows['spacetime'] = []

        # Set up expected_spacetime_dict
        expected_spacetime_dict = {
            (0, 0): 1,
            (0, 1): 2,
            (1, 0): 3,
            (1, 1): 4,
            'world': 5,
        }

        # Call the method
        print()
        spacetime_dict = self.world.get_spacetime_dict()
        print(self.world.db_rows['spacetime'])

        # Assertions
        self.assertEqual(spacetime_dict, expected_spacetime_dict)

        for spacetime in self.world.db_rows['spacetime']:
            self.assertIn(spacetime[0], expected_spacetime_dict.values()) # spactime_id in dict
        
        self.assertEqual(len(self.world.db_rows['spacetime']), len(expected_spacetime_dict))

    def test_get_db_ids_dict(self):
        # Create sample dataframes with a specific number of rows
        self.world.network_dfs = {
            'features': pd.DataFrame(index=range(5)),  # 5 rows
            'interactions': pd.DataFrame(index=range(3)),  # 3 rows
            'traits': pd.DataFrame(index=range(7)),  # 7 rows
        }

        expected_db_ids = {
            'features': self.world.network_id + self.world.controller.total_networks * 5,  # 5 rows
            'interactions': self.world.network_id + self.world.controller.total_networks * 3,  # 3 rows
            'traits': self.world.network_id + self.world.controller.total_networks * 7,  # 7 rows
        }

        # Call the method
        db_ids = self.world.get_db_ids_dict()

        # Assertions
        self.assertEqual(db_ids, expected_db_ids)

    @patch('model.model.get_model_vars_row')
    @patch('model.model.get_phenotypes_rows')
    @patch('model.model.get_sites_rows')
    def test_database_update(self, mock_get_sites_rows, mock_get_phenotypes_rows, mock_get_model_vars_row):
        # Set side effects for the mocked methods
        def side_effect_model_vars(model, sd, rd):
            rd['model_vars'].append(('data',))

        def side_effect_phenotypes(model, sd, rd, shadow=False):
            rd['phenotypes'].append(('data',))

        def side_effect_sites(model, sd, rd):
            rd['environment'].append(('data',))
            rd['interaction_stats'].append(('data',))

        mock_get_model_vars_row.side_effect = side_effect_model_vars
        mock_get_phenotypes_rows.side_effect = side_effect_phenotypes
        mock_get_sites_rows.side_effect = side_effect_sites

        # Get clean db_rows
        self.world.db_rows = self.world.get_db_rows_dict()

        # Test without data collection or db write
        self.world.schedule.time = 4  # Not a data collection or db write time
        self.world.database_update()
        mock_get_model_vars_row.assert_not_called()
        mock_get_phenotypes_rows.assert_not_called()
        mock_get_sites_rows.assert_not_called()
        self.world.db.write_rows.assert_not_called()

        # Test data collection without db write
        self.world.schedule.time = 5  # Time for data collection but not db write
        self.world.database_update()
        mock_get_model_vars_row.assert_called_once_with(self.world, self.world.spacetime_dict, self.world.db_rows)
        mock_get_phenotypes_rows.assert_has_calls([
            # Check call for both world and shadow
            call(self.world, self.world.spacetime_dict, self.world.db_rows),
            call(self.world.shadow, self.world.spacetime_dict, self.world.db_rows, True)
        ])
        mock_get_sites_rows.assert_called_once_with(self.world, self.world.spacetime_dict, self.world.db_rows)

        # Get clean db_rows
        self.world.db_rows = self.world.get_db_rows_dict()

        expected_db_rows = {
            'model_vars': [('data',)],
            'phenotypes': [('data',), ('data',)],
            'environment': [('data',)],
            'interaction_stats': [('data',)]
        }

        # Test db write
        self.world.schedule.time = 10 # Time for db write
        self.world.database_update()
        self.world.db.write_rows.assert_called_once_with(expected_db_rows)

        # Ensure db_rows is reset after writing
        self.assertEqual(self.world.db_rows, self.world.get_db_rows_dict())

        # Test override parameter
        # Get clean db_rows
        self.world.db_rows = self.world.get_db_rows_dict()
        self.world.schedule.time = 3  # Not a data collection or db write time
        self.world.database_update(override=True)
        mock_get_model_vars_row.assert_called()
        mock_get_phenotypes_rows.assert_called()
        mock_get_sites_rows.assert_called()
        self.world.db.write_rows.assert_called()

def test_step(self):
    # Mock instance methods and attributes
    self.world.reset_shadow = Mock()
    self.world.verify_shadow = Mock()
    self.world.print_report = Mock()
    self.world.get_spacetime_dict = Mock(return_value={})
    self.world.database_update = Mock()
    self.world.prune_features = Mock()
    self.world.schedule = Mock()
    self.world.schedule.step = Mock()
    self.world.schedule.get_agent_count = Mock(return_value=5)  # Adjust as needed
    self.world.shadow = Mock()
    self.world.snap_interval = 10
    self.world.env = 5
    self.world.new = 5
    self.world.cached = 5

    # Mock the Site.reset method for all sites in both the main model and the shadow model
    for site in self.world.sites.values():
        site.reset = Mock()
    for site in self.world.shadow.sites.values():
        site.reset = Mock()

    # Scenario 1: Normal step without shadow reset or verification
    self.world.schedule.time = 1
    self.world.step()

    self.world.schedule.step.assert_called_once()
    self.world.reset_shadow.assert_not_called()
    self.world.verify_shadow.assert_called_once()
    self.world.prune_features.assert_called_once()
    self.world.database_update.assert_called_once()
    self.world.get_spacetime_dict.assert_called_once()
    self.world.print_report.assert_not_called()
    self.assertEqual(self.world.env, 0)
    self.assertEqual(self.world.cached, 0)
    self.assertEqual(self.world.new, 0)

    self.world.verify_shadow.reset_mock()

    # Scenario 2: Step at a time when shadow should be reset
    self.world.schedule.time = 10  # Time matches snap_interval
    self.world.step()

    self.world.reset_shadow.assert_called_once()
    self.world.verify_shadow.assert_not_called()

    self.world.reset_shadow.reset_mock()

    # Scenario 3: Step at a time when shadow should be verified
    self.world.schedule.time = 5  # Time does not match snap_interval
    self.world.step()

    self.world.reset_shadow.assert_not_called()
    self.world.verify_shadow.assert_called_once()

    self.world.print_report.reset_mock()

    # Scenario 4: Step at a time when a detailed report should be printed
    self.world.schedule.time = 20  # Matches condition for print_report
    self.world.step()

    self.world.print_report.assert_called_once()

    self.world.schedule.get_agent_count.reset_mock()

    # Scenario 5: Step when all agents are dead
    self.world.schedule.get_agent_count.return_value = 0
    self.world.schedule.time = 1  # Any time
    self.world.running = True
    self.world.step()

    self.assertFalse(self.world.running)  # Simulation should stop


    def tearDown(self):
        patch.stopall()


if __name__ == "__main__":
    unittest.main()
