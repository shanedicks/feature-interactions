import sys
import json
import unittest
from functools import partial
from unittest.mock import ANY, patch, MagicMock, mock_open
from model.run import Controller, get_param_set_list, main
    
def test_get_param_set_list():
    """Test get_param_set_list with combinations of single elements and lists."""
    params_dict_network = {
        "init_env_features": 10,
        "init_agent_features": [2, 3],
        "max_feature_interactions": 5,
        "trait_payoff_mod": 0.5,
        "anchor_bias": [-1, -0.75],
        "payoff_bias": 0.0
    }

    params_dict_world = {
        "trait_mutate_chance": [0.01, 0.02],
        "trait_create_chance": 0.001,
        "feature_mutate_chance": 0.001,
        "feature_create_chance": 0.001,
        "feature_gain_chance": 0.5,
        "feature_timeout": [100, 200],
        "trait_timeout": 100,
        "init_agents": 1200,
        "base_agent_utils": 0.0,
        "base_env_utils": 100.0,
        "total_pop_limit": 9000,
        "pop_cost_exp": 2,
        "feature_cost_exp": 0.75,
        "grid_size": 3,
        "repr_multi": 1,
        "mortality": 0.01,
        "move_chance": 0.01,
        "snap_interval": 500,
        "target_sample": 1
    }

    expected_result_network = [
        {'init_env_features': 10, 'init_agent_features': 2, 'max_feature_interactions': 5, 'trait_payoff_mod': 0.5, 'anchor_bias': -1, 'payoff_bias': 0.0},
        {'init_env_features': 10, 'init_agent_features': 2, 'max_feature_interactions': 5, 'trait_payoff_mod': 0.5, 'anchor_bias': -0.75, 'payoff_bias': 0.0},
        {'init_env_features': 10, 'init_agent_features': 3, 'max_feature_interactions': 5, 'trait_payoff_mod': 0.5, 'anchor_bias': -1, 'payoff_bias': 0.0},
        {'init_env_features': 10, 'init_agent_features': 3, 'max_feature_interactions': 5, 'trait_payoff_mod': 0.5, 'anchor_bias': -0.75, 'payoff_bias': 0.0},
    ]

    expected_result_world = [
        {
            "trait_mutate_chance": 0.01,
            "trait_create_chance": 0.001,
            "feature_mutate_chance": 0.001,
            "feature_create_chance": 0.001,
            "feature_gain_chance": 0.5,
            "feature_timeout": 100,
            "trait_timeout": 100,
            "init_agents": 1200,
            "base_agent_utils": 0.0,
            "base_env_utils": 100.0,
            "total_pop_limit": 9000,
            "pop_cost_exp": 2,
            "feature_cost_exp": 0.75,
            "grid_size": 3,
            "repr_multi": 1,
            "mortality": 0.01,
            "move_chance": 0.01,
            "snap_interval": 500,
            "target_sample": 1
        },
        {
            "trait_mutate_chance": 0.01,
            "trait_create_chance": 0.001,
            "feature_mutate_chance": 0.001,
            "feature_create_chance": 0.001,
            "feature_gain_chance": 0.5,
            "feature_timeout": 200,
            "trait_timeout": 100,
            "init_agents": 1200,
            "base_agent_utils": 0.0,
            "base_env_utils": 100.0,
            "total_pop_limit": 9000,
            "pop_cost_exp": 2,
            "feature_cost_exp": 0.75,
            "grid_size": 3,
            "repr_multi": 1,
            "mortality": 0.01,
            "move_chance": 0.01,
            "snap_interval": 500,
            "target_sample": 1
        },
        {
            "trait_mutate_chance": 0.02,
            "trait_create_chance": 0.001,
            "feature_mutate_chance": 0.001,
            "feature_create_chance": 0.001,
            "feature_gain_chance": 0.5,
            "feature_timeout": 100,
            "trait_timeout": 100,
            "init_agents": 1200,
            "base_agent_utils": 0.0,
            "base_env_utils": 100.0,
            "total_pop_limit": 9000,
            "pop_cost_exp": 2,
            "feature_cost_exp": 0.75,
            "grid_size": 3,
            "repr_multi": 1,
            "mortality": 0.01,
            "move_chance": 0.01,
            "snap_interval": 500,
            "target_sample": 1
        },
        {
            "trait_mutate_chance": 0.02,
            "trait_create_chance": 0.001,
            "feature_mutate_chance": 0.001,
            "feature_create_chance": 0.001,
            "feature_gain_chance": 0.5,
            "feature_timeout": 200,
            "trait_timeout": 100,
            "init_agents": 1200,
            "base_agent_utils": 0.0,
            "base_env_utils": 100.0,
            "total_pop_limit": 9000,
            "pop_cost_exp": 2,
            "feature_cost_exp": 0.75,
            "grid_size": 3,
            "repr_multi": 1,
            "mortality": 0.01,
            "move_chance": 0.01,
            "snap_interval": 500,
            "target_sample": 1
        }
    ]

    result_network = get_param_set_list(params_dict_network)
    result_world = get_param_set_list(params_dict_world)

    assert result_network == expected_result_network
    assert result_world == expected_result_world

class TestController(unittest.TestCase):

    # Define the method to simulate world stepping
    def simulate_world_step(self):
        self.mock_world.schedule.time += 1

    def setUp(self):
        # Mocking parameters
        self.network_params = {
            "init_env_features": 10,
            "init_agent_features": 2,
            "max_feature_interactions": 5,
            "trait_payoff_mod": 0.5,
            "anchor_bias": 0.0,
            "payoff_bias": 0.0
        }

        self.world_params = {
            "trait_mutate_chance": 0.01,
            "trait_create_chance": 0.001,
            "feature_mutate_chance": 0.001,
            "feature_create_chance": 0.001,
            "feature_gain_chance": 0.5,
            "feature_timeout": 100,
            "trait_timeout": 100,
            "init_agents": 1200,
            "base_agent_utils": 0.0,
            "base_env_utils": 100.0,
            "total_pop_limit": 9000,
            "pop_cost_exp": 2,
            "feature_cost_exp": 0.75,
            "grid_size": 3,
            "repr_multi": 1,
            "mortality": 0.01,
            "move_chance": 0.01,
            "snap_interval": 500,
            "target_sample": 1
        }

        # Mocking the database manager
        self.mock_manager = MagicMock()
        self.mock_manager.write_row = MagicMock()
        self.mock_manager.initialize_db = MagicMock()
        self.mock_manager.db_string = 'mock_db_string'

        # Mocking the World class
        self.mock_world = MagicMock()
        self.mock_world.next_feature = MagicMock()
        self.mock_world.get_features_list = MagicMock()
        self.mock_world.get_feature_by_name = MagicMock()
        self.mock_world.db_rows = {'features': [], 'interactions': [], 'traits': []}
        self.mock_world.db.write_rows = MagicMock()
        self.mock_world.running = True
        self.mock_world.schedule.time = 0
        self.mock_world.step = MagicMock(side_effect=self.simulate_world_step)
        self.mock_world.database_update = MagicMock()
        self.mock_world.cleanup = MagicMock()

        # Mocking the Interaction class
        self.mock_interaction = MagicMock()

        # Define a simple side effect for the world constructor to reset schedule.time
        def reset_schedule_time_and_return_mock_world(*args, **kwargs):
            self.mock_world.schedule.time = 0
            return self.mock_world

        # Mocking the World constructor to return the mock_world
        self.world_constructor_patch = patch('model.run.World', side_effect=reset_schedule_time_and_return_mock_world)
        self.mock_world_constructor = self.world_constructor_patch.start()

        # Mocking the Interaction constructor to return mock_interaction
        self.interaction_constructor_patch = patch('model.run.Interaction', return_value=self.mock_interaction)
        self.mock_interaction_constructor = self.interaction_constructor_patch.start()

        # Mocking get_db_manager to return the mock_manager
        self.get_db_manager_patch = patch('model.run.Controller.get_db_manager', return_value=self.mock_manager)
        self.mock_get_db_manager = self.get_db_manager_patch.start()

        # Setting up controller with default parameters for comparison
        self.controller = Controller(
            experiment_name="test_experiment",
            path_to_db="/path/to/db"
        )

        # Define parameters for the run method
        self.num_networks = 2
        self.num_iterations = 2
        self.max_steps = 10


        # Adjusting the mock of get_param_set_list to return values based on the input
        def mock_get_param_set_list(input_dict):
            if input_dict in [self.network_params, self.controller.default_network_params]:
                return [self.network_params]
            elif input_dict in [self.world_params, self.controller.default_world_params]:
                return [self.world_params]
            return [{'dummy_param': 'dummy_value'}]  # Fallback for unexpected inputs

        self.get_param_set_list_patch = patch('model.run.get_param_set_list', side_effect=mock_get_param_set_list)
        self.mock_get_param_set_list = self.get_param_set_list_patch.start()

        # Mocking file handling and tqdm
        self.open_patch = patch('builtins.open', mock_open(read_data='data'))
        self.mock_open = self.open_patch.start()
        self.tqdm_patch = patch('model.run.tqdm')
        self.mock_tqdm = self.tqdm_patch.start()

        # Reset mocks to clear any calls made during the Controller initialization
        self.mock_manager.reset_mock()
        self.mock_world.reset_mock()
        self.mock_interaction.reset_mock()
        self.mock_world_constructor.reset_mock()
        self.mock_interaction_constructor.reset_mock()
        self.mock_get_db_manager.reset_mock()
        self.mock_get_param_set_list.reset_mock()
        self.mock_open.reset_mock()
        self.mock_tqdm.reset_mock()

    def tearDown(self):
        # Stopping all patches
        self.world_constructor_patch.stop()
        self.interaction_constructor_patch.stop()
        self.get_db_manager_patch.stop()
        self.get_param_set_list_patch.stop()
        self.open_patch.stop()
        self.tqdm_patch.stop()

    def test_init(self):
        # Define the parameters for initialization
        experiment_name = "test_experiment"
        path_to_db = "/path/to/db"
        data_interval = 10
        db_interval = 5

        # Initialize the Controller with minimal parameters (simulate typical usage)
        controller = Controller(
            experiment_name=experiment_name,
            path_to_db=path_to_db,
            data_interval=data_interval,
            db_interval=db_interval
        )

        # Assertions to validate the logic of __init__
        self.assertEqual(controller.experiment_name, experiment_name)
        self.mock_get_db_manager.assert_called_once_with(path_to_db)  # Assert get_db_manager was called correctly
        self.assertEqual(controller.db_manager, self.mock_manager)  # Assert db_manager is set to the mock_manager
        self.assertEqual(controller.data_interval, data_interval)
        self.assertEqual(controller.db_interval, db_interval)
        self.assertIsNone(controller.features_network)  # Assert features_network is None by default

        # Asserting default values when optional parameters are not provided
        controller_default_params = Controller(
            experiment_name=experiment_name,
            path_to_db=path_to_db
        )
        self.assertEqual(controller_default_params.data_interval, 1)
        self.assertEqual(controller_default_params.db_interval, 1)
        self.assertIsNone(controller_default_params.features_network)

    @patch('model.run.Manager')
    def test_get_db_manager(self, MockManager):
        # Create a Controller instance with minimal parameters
        experiment_name = "test_experiment"
        path_to_db = "/path/to/db"
        controller = Controller(experiment_name=experiment_name, path_to_db=path_to_db)

        # Stop the automatic patching of get_db_manager
        self.get_db_manager_patch.stop()

        # Call get_db_manager
        db_manager = controller.get_db_manager(path_to_db)

        # Assert that Manager is initialized correctly
        MockManager.assert_called_once_with(path_to_db, f"{experiment_name}.db")

        # Assert that the result of get_db_manager is a mock (since Manager is patched)
        self.assertIsInstance(db_manager, MagicMock)  # Use MagicMock instead of MockManager

        # Re-enable the automatic patching of get_db_manager
        self.get_db_manager_patch.start()

    @patch('builtins.print')
    def test_construct_network(self, mock_print):
        # Set up the controller with a mock features_network
        features_network = {
            'env_features': ['env_feature1', 'env_feature2'],
            'agent_features': ['agent_feature1', 'agent_feature2'],
            'interactions': [
                {'initiator': 'feature1', 'target': 'feature2', 'anchors': {'i': 0.5, 't': 0.0}}
            ]
        }
        controller = Controller(
            experiment_name="test_experiment",
            path_to_db="/path/to/db",
            features_network=features_network
        )

        network_id = 123  # Dummy network_id for testing
        controller.construct_network(network_id)

        # Validate World instance creation
        self.mock_world_constructor.assert_called_once()
        # Validate that next_feature is called for each feature in features_network
        self.assertEqual(self.mock_world.next_feature.call_count, len(features_network['env_features']) + len(features_network['agent_features']))
        # Validate that get_feature_by_name is called for each interaction
        self.assertEqual(self.mock_world.get_feature_by_name.call_count, 2 * len(features_network['interactions']))
        # Validate that Interaction instances are created
        self.assertEqual(self.mock_interaction_constructor.call_count, len(features_network['interactions']))
        # Validate that the write_rows method is called to update the database
        self.mock_world.db.write_rows.assert_called_once()

        # Check if the db_rows passed to write_rows match the structure in construct_network
        expected_db_rows = {
            'features': self.mock_world.db_rows['features'],
            'interactions': self.mock_world.db_rows['interactions'],
            'traits': self.mock_world.db_rows['traits']
        }
        self.mock_world.db.write_rows.assert_called_with(expected_db_rows)

    def test_run_with_provided_params(self):
        # Set up the controller
        controller = Controller(
            experiment_name="test_experiment",
            path_to_db="/path/to/db"
        )

        # Call run method with provided network_params and world_params
        controller.run(self.num_networks, self.num_iterations, self.max_steps, self.network_params, self.world_params)

        # Assertions and validations
        self.mock_manager.initialize_db.assert_called_once()  # Database initialization
        self.mock_open.assert_called_once()  # File opened for redirecting stdout
        self.mock_get_param_set_list.assert_any_call(self.network_params)  # get_param_set_list called with network_params
        self.mock_get_param_set_list.assert_any_call(self.world_params)  # get_param_set_list called with world_params
        self.assertEqual(self.mock_manager.write_row.call_count, self.num_networks + (self.num_networks * self.num_iterations))  # Network and world rows written
        self.mock_world_constructor.assert_called_with(controller, ANY, ANY, **self.network_params, **self.world_params)  # World instance created
        self.mock_world.step.assert_called()  # World simulation executed
        self.mock_world.cleanup.assert_called()  # World cleanup after simulation

    def test_run_with_default_params(self):
        # Set up the controller
        controller = Controller(
            experiment_name="test_experiment",
            path_to_db="/path/to/db"
        )

        # Call run method without network_params and world_params (use defaults)
        controller.run(self.num_networks, self.num_iterations, self.max_steps)

        # Assertions and validations
        self.mock_manager.initialize_db.assert_called_once()  # Database initialization
        self.mock_open.assert_called_once()  # File opened for redirecting stdout
        self.mock_get_param_set_list.assert_any_call(controller.default_network_params)  # Default network_params used
        self.mock_get_param_set_list.assert_any_call(controller.default_world_params)  # Default world_params used
        self.assertEqual(self.mock_manager.write_row.call_count, self.num_networks + (self.num_networks * self.num_iterations))  # Network and world rows written
        self.mock_world_constructor.assert_called_with(controller, ANY, ANY, **self.network_params, **self.world_params)  # World instance created
        self.mock_world.step.assert_called()  # World simulation executed
        self.mock_world.cleanup.assert_called()  # World cleanup after simulation

    @patch('model.run.open', new_callable=mock_open)
    @patch('contextlib.redirect_stdout', new_callable=MagicMock)
    def test_run_network(self, mock_redirect_stdout, mock_file):
        # Setup for the test
        network = (123, self.network_params)  # Dummy network tuple (network_id, network_params)
        num_iterations = 2
        world_params_dict = self.world_params

        # Set up the controller with minimal parameters
        controller = Controller(
            experiment_name="test_experiment",
            path_to_db="/path/to/db"
        )
        controller.max_steps = self.max_steps

        # Call run_network method
        controller.run_network(network, num_iterations, world_params_dict)

        # Assertions and validations
        self.mock_get_param_set_list.assert_called_once_with(world_params_dict)  # get_param_set_list called with world_params_dict
        self.assertEqual(self.mock_manager.write_row.call_count, num_iterations)  # write_row called for each iteration
        self.mock_world_constructor.assert_called_with(
            controller, 
            ANY, 
            network[0], 
            **network[1], 
            **world_params_dict
        )  # World instance created
        self.assertEqual(self.mock_world.step.call_count, num_iterations * self.max_steps)  # World simulation executed num_iterations * max_steps times
        self.assertEqual(self.mock_world.cleanup.call_count, num_iterations)  # World cleanup after simulation num_iterations times

        # Check if the outfile is correctly formed and opened
        expected_outfile = controller.db_manager.db_string.replace(".db", f"_{network[0]}.txt")
        mock_file.assert_called_once_with(expected_outfile, 'w')

    @patch('model.run.tqdm')
    @patch('model.run.Pool')
    def test_run_mp(self, mock_pool, mock_tqdm):
        # Setup for the test
        num_networks = 2
        num_iterations = 2
        max_steps = 10
        num_processes = 2
        network_params_dict = self.network_params
        world_params_dict = self.world_params

        # Set up the controller
        controller = Controller(
            experiment_name="test_experiment",
            path_to_db="/path/to/db"
        )
        controller.max_steps = max_steps

        # Configure the mocks
        mock_pool.return_value.__enter__.return_value.imap_unordered.return_value = iter([None] * (num_networks * len(get_param_set_list(network_params_dict))))
        mock_tqdm.return_value.__enter__.return_value.update = MagicMock()

        # Call run_mp method
        controller.run_mp(num_networks, num_iterations, max_steps, num_processes, network_params_dict, world_params_dict)

        # Assertions and validations
        self.mock_manager.initialize_db.assert_called_once()  # Database initialization
        self.mock_get_param_set_list.assert_any_call(network_params_dict)  # get_param_set_list called with network_params_dict
        self.assertEqual(self.mock_manager.write_row.call_count, num_networks * len(get_param_set_list(network_params_dict)))  # write_row called for each network

        # Get the actual call to imap_unordered
        actual_call = mock_pool.return_value.__enter__.return_value.imap_unordered.call_args

        # Assert that imap_unordered was called once
        mock_pool.return_value.__enter__.return_value.imap_unordered.assert_called_once()

        # Assert the first argument (the function) is as expected
        actual_func = actual_call[0][0]
        expected_func = partial(
            controller.run_network,
            num_iterations=num_iterations,
            world_params_dict=world_params_dict,
        )
        # Compare the function part of the partial
        self.assertEqual(actual_func.func, expected_func.func)

        # Compare the args part of the partial
        self.assertEqual(actual_func.args, expected_func.args)

        # Compare the keywords part of the partial
        self.assertEqual(actual_func.keywords, expected_func.keywords)

        # Assert the second argument (the networks list) is as expected
        actual_networks = actual_call[0][1]
        self.assertIsInstance(actual_networks, list)  # Check it's a list
        self.assertEqual(len(actual_networks), num_networks * len(get_param_set_list(network_params_dict)))  # Check the length

        # Assert the structure of each item in the networks list
        for network in actual_networks:
            self.assertIsInstance(network, tuple)  # Each item should be a tuple
            self.assertEqual(len(network), 2)  # Each tuple should have 2 items
            self.assertIsInstance(network[0], MagicMock)  # First item is the network_id (mocked)
            self.assertEqual(network[1], network_params_dict)  # Second item is the network_params
            # Assert tqdm is used correctly
            mock_tqdm.assert_called_once()
            # Progress bar updates as many times as there are networks
            self.assertEqual(mock_tqdm.return_value.__enter__.return_value.update.call_count, num_networks)


class TestMainFunction(unittest.TestCase):

    def setUp(self):
        # Mock sys.argv
        self.mock_argv = patch('sys.argv', ['script_name', 'config_file.json']).start()

        # Mock open for file operations
        self.mock_file_data = '{"title": "test", "output_path": "/path", "db_interval": 5, "data_interval": 10, "features_network": {}, "num_networks": 2, "num_iterations": 3, "max_steps": 100, "network_params": {}, "world_params": {}}'
        self.mock_open = patch('model.run.open', mock_open(read_data=self.mock_file_data)).start()

        # Mock json.loads
        self.config_data = json.loads(self.mock_file_data)
        self.mock_json_loads = patch('model.run.json.loads', return_value=self.config_data).start()

        # Mock datetime for timestamp generation
        self.mock_datetime = patch('model.run.datetime').start()
        self.mock_datetime.today.return_value.strftime.return_value = '20220101_120000'

        # Mock os for environment variable and filesystem operations
        self.mock_os = patch('model.run.os').start()

        # Mock shutil for file operations
        self.mock_shutil = patch('model.run.shutil').start()

        # Mock the Controller
        self.mock_controller = patch('model.run.Controller').start()
        self.mock_controller_instance = self.mock_controller.return_value

    def tearDown(self):
        patch.stopall()

    def test_main_single_process(self):
        # Set up for single process
        self.mock_os.environ.get.return_value = 1

        # Call the main function
        main()

        # Assertions for single process
        self.mock_open.assert_called_once_with('config_file.json', 'r')
        self.assertEqual(self.mock_open().read.call_count, 1)  # Check the file is read once
        self.mock_os.mkdir.assert_called_once_with('/path/test_20220101_120000')
        self.mock_shutil.copy2.assert_called_once_with('config_file.json', '/path/test_20220101_120000')

        # Check Controller instantiation and method calls
        self.mock_controller.assert_called_once_with(
            experiment_name='test_20220101_120000',
            path_to_db='/path/test_20220101_120000',
            db_interval=5,
            data_interval=10,
            features_network={}
        )
        self.mock_controller_instance.run.assert_called_once_with(
            2, 3, 100, network_params_dict={}, world_params_dict={}
        )
        self.mock_controller_instance.run_mp.assert_not_called()

    def test_main_multi_process(self):
        # Set up for multiple processes
        self.mock_os.environ.get.return_value = 2

        # Call the main function
        main()

        # Assertions for multiple processes
        self.mock_open.assert_called_once_with('config_file.json', 'r')
        self.assertEqual(self.mock_open().read.call_count, 1)  # Check the file is read once
        self.mock_os.mkdir.assert_called_once_with('/path/test_20220101_120000')
        self.mock_shutil.copy2.assert_called_once_with('config_file.json', '/path/test_20220101_120000')

        # Check Controller instantiation and method calls
        self.mock_controller.assert_called_once_with(
            experiment_name='test_20220101_120000',
            path_to_db='/path/test_20220101_120000',
            db_interval=5,
            data_interval=10,
            features_network={}
        )
        self.mock_controller_instance.run_mp.assert_called_once_with(
            2, 3, 100, num_processes=2, network_params_dict={}, world_params_dict={}
        )
        self.mock_controller_instance.run.assert_not_called()

if __name__ == '__main__':
    unittest.main()
