import unittest
from math import log2
from statistics import quantiles
from unittest.mock import Mock, MagicMock, PropertyMock, patch, call
import numpy as np
from model.output import (
    get_population, get_total_utility, get_mean_utility, get_median_utility,
    get_num_agent_features, get_num_roles, get_num_phenotypes, 
    get_model_vars_row, get_phenotypes_rows, get_sites_rows, 
    env_features_dist, traits_dist, role_dist, phenotype_dist,
    env_report, payoff_quantiles, print_matrix, interaction_report,
    occupied_roles_list, occupied_phenotypes_list
)


class TestOutput(unittest.TestCase):
    def setUp(self):
        # Mock features
        self.feature_values = ['a','b','c','d','e']
        self.mock_features = ['A', 'B', 'C', 'F', 'G', 'H']

        # Mock interactions with explicitly defined payoffs
        self.mock_interactions = [Mock() for _ in range(3)]
        self.mock_interactions[0].payoffs = {
            'a': {'a': (0.1, -0.1), 'b': (0.2, -0.2)},
            'b': {'a': (0.3, -0.3), 'b': (0.4, -0.4)}
        }
        self.mock_interactions[1].payoffs = {
            'a': {'b': (0.6, -0.6)},
            'b': {'a': (0.7, -0.7)}
        }
        self.mock_interactions[2].payoffs = {
            'a': {'a': (0.9, -0.9), 'b': (1.0, -1.0)},
            'b': {'a': (1.1, -1.1), 'b': (1.2, -1.2)}
        }

        # Mock World object and its properties/methods
        self.world = Mock()
        self.world.schedule = Mock()
        self.world.schedule.get_agent_count.return_value = 10
        self.world.get_features_list.return_value = self.mock_features
        self.world.new = np.random.randint(1, 10)
        self.world.cached = np.random.randint(1, 10)
        self.world.env = np.random.randint(1, 10)

        # Mocking the agents in the schedule
        self.world.schedule.agents = [Mock() for _ in range(100)]
        for agent in self.world.schedule.agents:
            type(agent).utils = PropertyMock(return_value=np.random.rand())
        
        # Define site positions
        self.site_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]


        # Define mock feature keys for site.utils with integer trait_ids
        mock_A = Mock()
        type(mock_A).trait_ids = PropertyMock(return_value={'a': 100, 'b': 200})
        self.mock_A = mock_A
        mock_B = Mock()
        type(mock_B).trait_ids = PropertyMock(return_value={'a': 300, 'b': 400})
        self.mock_B = mock_B

        # Define sites
        self.world.sites = {
            (0, 0): Mock(
                born=5, died=2, moved_in=3, moved_out=1,
                get_pop=Mock(return_value=20),
                pos = (0, 0), pop_cost = 5/9,
                traits = {mock_A: 'a', mock_B: 'b'},
                utils={mock_A: 0.1, mock_B: 0.2},
                interaction_stats={1: (10, 0.5, 0.6), 2: (20, 0.7, 0.8)}
            ),
            (0, 1): Mock(
                born=3, died=1, moved_in=4, moved_out=2,
                get_pop=Mock(return_value=30),
                pos = (0, 1), pop_cost = 2.1,
                traits = {mock_A: 'b', mock_B: 'a'},
                utils={mock_A: 0.3, mock_B: 0.4},
                interaction_stats={1: (15, 0.9, 1.0), 2: (25, 1.1, 1.2)}
            ),
            (1, 0): Mock(
                born=6, died=3, moved_in=2, moved_out=1,
                get_pop=Mock(return_value=25),
                pos = (1, 0), pop_cost = 0,
                traits = {mock_A: 'b'},
                utils={mock_A: 0.5},
                interaction_stats={1: (5, 1.3, 1.4), 2: (10, 1.5, 1.6)}
            ),
            (1, 1): Mock(
                born=4, died=2, moved_in=3, moved_out=2,
                get_pop=Mock(return_value=15),
                pos = (1, 1), pop_cost = 3.057,
                traits = {},
                utils={},
                interaction_stats={1: (12, 1.7, 1.8), 2: (18, 1.9, 2.0)}
            )
        }

        # Mock Roles
        self.world.roles_dict = {
            'RoleA': Mock(types={
                (0, 0): {'A.a': 3, 'A.b': 0},
                (0, 1): {'A.a': 0, 'A.b': 2},
                (1, 0): {},
                (1, 1): {}
            }),
            'RoleB': Mock(types={
                (0, 0): {'B.a': 0},
                (0, 1): {'B.a': 0},
                (1, 0): {},
                (1, 1): {}
            }),
            'RoleC': Mock(types={
                (0, 0): {'C.a': 5},
                (0, 1): {'C.a': 0, 'C.b': 7},
                (1, 0): {},
                (1, 1): {}
            })
        }

        # SpacetimeDict and RowDict mocks
        spacetime_keys = [(0, 0), (0, 1), (1, 0), (1, 1), "world"]
        self.sd = {k: i + 1 for i, k in enumerate(spacetime_keys)}
        self.rd = {
            'model_vars': [],
            'phenotypes': [],
            'demographics': [],
            'environment': [],
            'interaction_stats': []
        }

    def test_get_utility_functions(self):
        # Prepare utils array manually for comparison
        manual_utils = np.array([agent.utils for agent in self.world.schedule.agents])
        
        # Test get_total_utility
        total_utility_no_arg = get_total_utility(self.world)
        total_utility_with_arg = get_total_utility(self.world, manual_utils)
        self.assertEqual(total_utility_no_arg, total_utility_with_arg)
        self.assertAlmostEqual(total_utility_no_arg, np.sum(manual_utils))

        # Test get_mean_utility
        mean_utility_no_arg = get_mean_utility(self.world)
        mean_utility_with_arg = get_mean_utility(self.world, manual_utils)
        self.assertEqual(mean_utility_no_arg, mean_utility_with_arg)
        self.assertAlmostEqual(mean_utility_no_arg, np.mean(manual_utils))

        # Test get_median_utility
        median_utility_no_arg = get_median_utility(self.world)
        median_utility_with_arg = get_median_utility(self.world, manual_utils)
        self.assertEqual(median_utility_no_arg, median_utility_with_arg)
        self.assertAlmostEqual(median_utility_no_arg, np.median(manual_utils))

        # Test with no agents in the schedule
        self.world.schedule.get_agent_count.return_value = 0
        self.world.schedule.agents = []
        
        # Test get_total_utility
        total_utility_no_agents = get_total_utility(self.world)
        self.assertEqual(total_utility_no_agents, 0)

        # Test get_mean_utility
        mean_utility_no_agents = get_mean_utility(self.world)
        self.assertEqual(mean_utility_no_agents, 0)

        # Test get_median_utility
        median_utility_no_agents = get_median_utility(self.world)
        self.assertEqual(median_utility_no_agents, 0)

    def test_occupied_lists(self):
        # Test occupied_roles_list
        # Expecting roles with any type count > 0 at any site to be considered 'occupied'
        roles_dict = self.world.roles_dict
        actual_occupied_roles = occupied_roles_list(self.world)
        self.assertIn(roles_dict['RoleA'], actual_occupied_roles, "'RoleA' should be in the occupied roles list")
        self.assertIn(roles_dict['RoleC'], actual_occupied_roles, "'RoleC' should be in the occupied roles list")
        self.assertEqual(len(actual_occupied_roles), 2, "There should be exactly 2 occupied roles")

        # Test occupied_phenotypes_list
        # Expecting phenotypes with count > 0 at any site to be considered 'occupied'
        actual_occupied_phenotypes = occupied_phenotypes_list(self.world)
        expected_occupied_phenotypes = {'A.a', 'A.b', 'C.a', 'C.b'}  # Phenotypes with count > 0
        for phenotype in expected_occupied_phenotypes:
            self.assertIn(phenotype, actual_occupied_phenotypes, f"'{phenotype}' should be in the occupied phenotypes list")
        self.assertEqual(len(actual_occupied_phenotypes), len(expected_occupied_phenotypes), "The number of occupied phenotypes does not match expected")

    @patch('model.output.occupied_roles_list')
    @patch('model.output.occupied_phenotypes_list')
    def test_count_functions(self, mock_occupied_phenotypes_list, mock_occupied_roles_list):
        # Mock the return value of occupied_roles_list and occupied_phenotypes_list
        mock_occupied_roles_list.return_value = ['role1', 'role2', 'role3']
        mock_occupied_phenotypes_list.return_value = ['phenotype1', 'phenotype2']

        # Test get_num_agent_features
        self.world.get_features_list.return_value = self.mock_features[3:]
        num_features = get_num_agent_features(self.world)
        self.assertEqual(num_features, 3)

        # Test get_num_roles
        num_roles = get_num_roles(self.world)
        self.assertEqual(num_roles, 3)  # Length of the mocked occupied_roles_list

        # Test get_num_phenotypes
        num_phenotypes = get_num_phenotypes(self.world)
        self.assertEqual(num_phenotypes, 2)  # Length of the mocked occupied_phenotypes_list

    @patch('model.output.get_num_agent_features')
    @patch('model.output.get_num_roles')
    @patch('model.output.get_num_phenotypes')
    @patch('model.output.get_median_utility')
    @patch('model.output.get_mean_utility')
    @patch('model.output.get_total_utility')
    @patch('model.output.get_population')
    def test_get_model_vars_row(self, mock_get_population, mock_get_total_utility, mock_get_mean_utility, 
                                mock_get_median_utility, mock_get_num_phenotypes, mock_get_num_roles, 
                                mock_get_num_agent_features):
        # Mock return values
        mock_get_population.return_value = 100
        mock_get_total_utility.return_value = 200
        mock_get_mean_utility.return_value = 2
        mock_get_median_utility.return_value = 3
        mock_get_num_phenotypes.return_value = 4
        mock_get_num_roles.return_value = 5
        mock_get_num_agent_features.return_value = 6
        
        # Call the function
        get_model_vars_row(self.world, self.sd, self.rd)

        
        # Assertions for function calls
        mock_get_population.assert_called_once_with(self.world)
        mock_get_num_phenotypes.assert_called_once_with(self.world)
        mock_get_num_roles.assert_called_once_with(self.world)
        mock_get_num_agent_features.assert_called_once_with(self.world)

        # Check calls to get_utility functions
        utils = np.array([a.utils for a in self.world.schedule.agents])

        # get_total_utility
        actual_args, actual_kwargs = mock_get_total_utility.call_args
        self.assertEqual(actual_args[0], self.world, "The world argument does not match")
        self.assertTrue(np.array_equal(actual_args[1], utils), "The utils arrays do not match")

        # get_mean_utility
        actual_args, actual_kwargs = mock_get_mean_utility.call_args
        self.assertEqual(actual_args[0], self.world, "The world argument does not match")
        self.assertTrue(np.array_equal(actual_args[1], utils), "The utils arrays do not match")

        # get_median_utility
        actual_args, actual_kwargs = mock_get_median_utility.call_args
        self.assertEqual(actual_args[0], self.world, "The world argument does not match")
        self.assertTrue(np.array_equal(actual_args[1], utils), "The utils arrays do not match")
        
        # Check that the expected row was appended to rd['model_vars']
        expected_row = (
            self.sd["world"],
            100,  # Population
            200,  # Total utility
            2,    # Mean utility
            3,    # Median utility
            4,    # Number of phenotypes
            5,    # Number of roles
            6,    # Number of agent features
            self.world.new + self.world.cached,
            self.world.env
        )
        self.assertIn(expected_row, self.rd['model_vars'], "Expected row was not appended to rd['model_vars']")

    @patch('model.output.phenotype_dist')
    def test_get_phenotypes_rows(self, mock_phenotype_dist):
        # Define expected phenotypes distribution for each site
        mock_phenotypes_dists = {
            (0, 0): {'P1': 10, 'P2': 5},
            (0, 1): {'P3': 8, 'P4': 7},
            (1, 0): {'P5': 6},
            (1, 1): {'P6': 4, 'P7': 3}
        }
        mock_phenotype_dist.side_effect = lambda model, site: mock_phenotypes_dists[site]

        # Call the function
        get_phenotypes_rows(self.world, self.sd, self.rd, shadow=False)

        # Assert that the correct rows are appended to rd['phenotypes']
        expected_rows = [
            (1, False, 'P1', 10),
            (1, False, 'P2', 5),
            (2, False, 'P3', 8),
            (2, False, 'P4', 7),
            (3, False, 'P5', 6),
            (4, False, 'P6', 4),
            (4, False, 'P7', 3)
        ]

        actual_rows = self.rd['phenotypes']
        self.assertListEqual(actual_rows, expected_rows, "The rows in rd['phenotypes'] do not match expected")

    def test_get_sites_rows(self):
        # Call the function
        get_sites_rows(self.world, self.sd, self.rd)

        # Define the expected rows explicitly for demographics
        expected_demographic_rows = [
            (1, 20, 5, 2, 3, 1),
            (2, 30, 3, 1, 4, 2),
            (3, 25, 6, 3, 2, 1),
            (4, 15, 4, 2, 3, 2)
        ]

        # Define the expected rows explicitly for environment
        expected_environment_rows = [
            (1, 100, 0.1),
            (1, 400, 0.2),
            (2, 200, 0.3),
            (2, 300, 0.4),
            (3, 200, 0.5),
            # Site (1, 1) has an empty utils dict
        ]

        # Define the expected rows explicitly for interaction_stats
        expected_interaction_stats_rows = [
            (1, 1, 10, 0.5, 0.6),
            (1, 2, 20, 0.7, 0.8),
            (2, 1, 15, 0.9, 1.0),
            (2, 2, 25, 1.1, 1.2),
            (3, 1, 5, 1.3, 1.4),
            (3, 2, 10, 1.5, 1.6),
            (4, 1, 12, 1.7, 1.8),
            (4, 2, 18, 1.9, 2.0)
        ]

        # Get the actual rows from rd
        actual_demographic_rows = self.rd['demographics']
        actual_environment_rows = self.rd['environment']
        actual_interaction_stats_rows = self.rd['interaction_stats']

        # Assert that the correct rows are appended to rd['demographics'], rd['environment'], and rd['interaction_stats']
        self.assertListEqual(actual_demographic_rows, expected_demographic_rows, "The rows in rd['demographics'] do not match expected")
        self.assertListEqual(actual_environment_rows, expected_environment_rows, "The rows in rd['environment'] do not match expected")
        self.assertListEqual(actual_interaction_stats_rows, expected_interaction_stats_rows, "The rows in rd['interaction_stats'] do not match expected")


    @patch('builtins.print')
    def test_env_features_dist(self, mock_print):
        # Call the function
        env_features_dist(self.world)

        # Prepare the expected calls to print
        expected_calls = [call(site, site.traits) for site in self.world.sites.values()]

        # Assert that print was called with the correct arguments for each site
        mock_print.assert_has_calls(expected_calls, any_order=True)


    def test_traits_dist(self):
        # Mock features with explicit values and traits_dict
        self.mock_feature1 = Mock(values=['a', 'b'])
        self.mock_feature1.traits_dict = {
            (0, 0): {'a': 2, 'b': 1},
            (0, 1): {'a': 1, 'b': 0},
            (1, 0): {'a': 0, 'b': 1},
            (1, 1): {}
        }

        self.mock_feature2 = Mock(values=['c', 'd'])
        self.mock_feature2.traits_dict = {
            (0, 0): {'c': 1, 'd': 2},
            (0, 1): {},
            (1, 0): {'c': 0, 'd': 1},
            (1, 1): {}
        }

        self.mock_feature3 = Mock(values=['e'])
        self.mock_feature3.traits_dict = {
            (0, 0): {},
            (0, 1): {},
            (1, 0): {},
            (1, 1): {'e': 3}
        }

        self.world.get_features_list = Mock(return_value=[self.mock_feature1, self.mock_feature2, self.mock_feature3])

        # Call the function without specifying a site
        traits_distribution = traits_dist(self.world)

        # Define the expected distribution
        expected_distribution = {
            self.mock_feature1: {'a': 3, 'b': 2},  # Summed across all sites
            self.mock_feature2: {'c': 1, 'd': 3},  # Summed across all sites
            self.mock_feature3: {'e': 3},          # Only present at site (1, 1)
        }

        self.assertEqual(traits_distribution, expected_distribution, "Traits distribution does not match expected")

        # Test with a specific site
        specific_site_distribution = traits_dist(self.world, (0, 0))
        expected_specific_site_distribution = {
            self.mock_feature1: {'a': 2, 'b': 1},
            self.mock_feature2: {'c': 1, 'd': 2},
            self.mock_feature3: {} # Feature3 has no traits at site (0, 0)
        }
        self.assertEqual(specific_site_distribution, expected_specific_site_distribution, "Traits distribution for specific site does not match expected")

    def test_role_dist(self):
        # Call the function without specifying a site
        all_sites_role_distribution = role_dist(self.world)

        # Define the expected distribution for all sites sorted by total
        expected_all_sites_distribution = [
            # RoleC: Types 'C.a': 5, 'C.b': 7
            [self.world.roles_dict['RoleC'], (round(-((5/12)*log2(5/12) + (7/12)*log2(7/12)), 2), 2, 12)],
            # RoleA: Types 'A.a': 3, 'A.b': 2
            [self.world.roles_dict['RoleA'], (round(-((3/5)*log2(3/5) + (2/5)*log2(2/5)), 2), 2, 5)],
            # RoleB doesn't have any type with count > 0
        ]

        # Assert that the distribution matches the expected distribution for all sites
        self.assertEqual(all_sites_role_distribution, expected_all_sites_distribution, "Role distribution across all sites does not match expected")

        # Test with a specific site
        specific_site = (0, 0)
        specific_site_role_distribution = role_dist(self.world, specific_site)
        
        # Define the expected distribution for the specific site sorted by total
        expected_specific_site_distribution = [
            # RoleC: Types 'C.a': 5
            [self.world.roles_dict['RoleC'], (0.00, 1, 5)],
            # RoleA: Types 'A.a': 3
            [self.world.roles_dict['RoleA'], (0.00, 1, 3)],
            # RoleB doesn't have any type with count > 0
        ]

        # Assert that the distribution matches the expected distribution for the specific site
        self.assertEqual(specific_site_role_distribution, expected_specific_site_distribution, "Role distribution for specific site does not match expected")

    def test_phenotype_dist(self):
        # Call the function without specifying a site
        all_sites_phenotype_distribution = phenotype_dist(self.world)

        # Define the expected distribution for all sites
        expected_all_sites_distribution = {
            'A.a': 3,  # Summed across all sites for phenotype 'A.a'
            'A.b': 2,  # Summed across all sites for phenotype 'A.b'
            'C.a': 5,  # Summed across all sites for phenotype 'C.a'
            'C.b': 7,  # Summed across all sites for phenotype 'C.b'
        }

        # Assert that the distribution matches the expected distribution for all sites
        self.assertEqual(all_sites_phenotype_distribution, expected_all_sites_distribution, "Phenotype distribution across all sites does not match expected")

        # Test with a specific site
        specific_site = (0, 0)
        specific_site_phenotype_distribution = phenotype_dist(self.world, specific_site)
        
        # Define the expected distribution for the specific site
        expected_specific_site_distribution = {
            'A.a': 3,  # For site (0, 0) for phenotype 'A.a'
            'C.a': 5,  # For site (0, 0) for phenotype 'C.a'
        }

        # Assert that the distribution matches the expected distribution for the specific site
        self.assertEqual(specific_site_phenotype_distribution, expected_specific_site_distribution, "Phenotype distribution for specific site does not match expected")

    @patch('builtins.print')
    def test_env_report(self, mock_print):
        # Call the function
        env_report(self.world)

        mock_A = self.mock_A
        mock_B = self.mock_B

        # Prepare the expected calls to print
        expected_calls = [
            call("Pos, Pop, Pop change, Cost, Utils"),
            call((0, 0), 20, 5 - 2, round(5/9, 2), {mock_A: 0.1, mock_B: 0.2}),
            call((0, 1), 30, 3 - 1, round(2.1, 2), {mock_A: 0.3, mock_B: 0.4}),
            call((1, 0), 25, 6 - 3, round(0, 2), {mock_A: 0.5}),
            call((1, 1), 15, 4 - 2, round(3.057, 2), {}),
        ]

        # Assert that print was called with the correct arguments for each site
        mock_print.assert_has_calls(expected_calls, any_order=False)

    def test_payoff_quantiles(self):
        # Mock interaction 1 (with more than 2 payoffs)

            # Initiator quartiles
        initiator_payoffs = [0.1, 0.2, 0.3, 0.4]
        initiator_quartiles = quantiles(initiator_payoffs)
        rounded_initiator_quartiles = [round(q, 2) for q in initiator_quartiles]

        # Target quartiles
        target_payoffs = [-0.1, -0.2, -0.3, -0.4]
        target_quartiles = quantiles(target_payoffs)
        rounded_target_quartiles = [round(q, 2) for q in target_quartiles]

        interaction1 = self.mock_interactions[0]
        result1 = payoff_quantiles(interaction1)
        expected_result1 = (4, rounded_initiator_quartiles, rounded_target_quartiles)
        self.assertEqual(result1, expected_result1, f"Payoff quantiles for interaction 1 do not match expected")

        # Mock interaction 2 (with 2 or fewer payoffs)
        interaction2 = self.mock_interactions[1]
        result2 = payoff_quantiles(interaction2)
        expected_result2 = (2, [0.6, 0.7], [-0.7, -0.6])
        self.assertEqual(result2, expected_result2, f"Payoff quantiles for interaction 2 do not match expected")

    @patch('builtins.print')
    def test_print_matrix(self, mock_print):
        # Choose an interaction to test
        interaction = self.mock_interactions[0]

        # Call the function
        print_matrix(interaction)

        # Prepare the expected calls to print
        expected_calls = [
            call(interaction),  # The first call prints the interaction itself
        ]
        # Adding calls for each item in the interaction's payoffs
        for item in interaction.payoffs.values():
            expected_calls.append(call([p for p in item.values()]))

        # Assert that print was called with the correct arguments
        mock_print.assert_has_calls(expected_calls, any_order=False)

    @patch('model.output.print_matrix')
    @patch('model.output.payoff_quantiles')
    @patch('builtins.print')
    def test_interaction_report(self, mock_print, mock_payoff_quantiles, mock_print_matrix):
        # Mock feature_interactions.edges to return a structure that includes self.mock_interactions
        edge_data = [(None, None, interaction) for interaction in self.mock_interactions]
        self.world.feature_interactions.edges = Mock(return_value=edge_data)

        # Call the function with full=True
        interaction_report(self.world, full=True)

        # Assert that print_matrix was called for each interaction when full=True
        expected_num_calls_print_matrix = len(self.mock_interactions)
        self.assertEqual(mock_print_matrix.call_count, expected_num_calls_print_matrix, f"print_matrix call count ({mock_print_matrix.call_count}) does not match expected ({expected_num_calls_print_matrix}) for full=True")

        mock_print_matrix.reset_mock()  # Reset mock for the next part of the test

        # Call the function with full=False
        interaction_report(self.world, full=False)

        # Assert that payoff_quantiles was called for each interaction when full=False
        expected_num_calls_payoff_quantiles = len(self.mock_interactions)
        self.assertEqual(mock_payoff_quantiles.call_count, expected_num_calls_payoff_quantiles, f"payoff_quantiles call count ({mock_payoff_quantiles.call_count}) does not match expected ({expected_num_calls_payoff_quantiles}) for full=False")


if __name__ == '__main__':
    unittest.main()
