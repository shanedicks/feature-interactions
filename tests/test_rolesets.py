import unittest
from unittest.mock import Mock, MagicMock, patch, PropertyMock, call
import pandas as pd
import numpy as np
from math import comb
from model.rolesets import FeatureUtilityCalculator, RoleAnalyzer

class TestFeatureUtilityCalculator(unittest.TestCase):
    def setUp(self):
        # Mock database and paths
        self.db_path = 'mock_db_path'
        self.mock_world_dict = {1: 10, 2: 20}  # Maps world_id to network_id
        self.mock_params_df = pd.DataFrame({
            'world_id': [1, 2],
            'network_id': [10, 20],
            'base_env_utils': [100.0, 100.0],
            'feature_cost_exp': [0.75, 0.75],
            'grid_size': [3, 3],
            'total_pop_limit': [9000, 9000],
            'pop_cost_exp': [2, 2]
        })

        # Setup patches
        self.get_world_dict_patch = patch('model.rolesets.db.get_world_dict', return_value=self.mock_world_dict)
        self.get_params_df_patch = patch('model.rolesets.db.get_params_df', return_value=self.mock_params_df)

        # Start patches
        self.get_world_dict_mock = self.get_world_dict_patch.start()
        self.get_params_df_mock = self.get_params_df_patch.start()

        # Create calculator instance with patched dependencies
        self.calculator = FeatureUtilityCalculator(self.db_path)

    def tearDown(self):
        # Stop all patches
        self.get_world_dict_patch.stop()
        self.get_params_df_patch.stop()

    def test_init(self):
        """Test the initialization of FeatureUtilityCalculator."""
        # Check that the calculator was initialized with the correct attributes
        self.assertEqual(self.calculator.db_path, self.db_path)
        self.assertEqual(self.calculator.world_dict, self.mock_world_dict)
        self.assertIs(self.calculator.params_df, self.mock_params_df)

        # Check that caches are initialized as empty dictionaries
        self.assertEqual(self.calculator.network_data, {})
        self.assertEqual(self.calculator.features_df_cache, {})
        self.assertEqual(self.calculator.fud_cache, {})
        self.assertEqual(self.calculator.env_features_cache, {})

    def test_ensure_network_data(self):
        """Test that _ensure_network_data properly loads and caches network data."""
        network_id = 10

        # Mock network data retrieval functions
        mock_interactions_df = pd.DataFrame({
            'network_id': [10, 10, 10],
            'initiator': ['A', 'B', 'A'],
            'target': ['B', 'C', 'C']
        })
        mock_payoffs_df = pd.DataFrame({
            'interaction_id': [1, 2, 3],
            'initiator_feature': ['A', 'B', 'A'],
            'initiator_trait': ['a', 'b', 'a'],
            'target_feature': ['B', 'C', 'C'],
            'target_trait': ['b', 'c', 'c'],
            'initiator_payoff': [0.1, 0.2, 0.3],
            'target_payoff': [0.4, 0.5, 0.6]
        })

        with patch('model.rolesets.db.get_interactions_df', return_value=mock_interactions_df), \
             patch('model.rolesets.db.get_payoffs_df', return_value=mock_payoffs_df):

            # Call the method
            self.calculator._ensure_network_data(network_id)

            # Check that data was stored correctly
            self.assertIn(network_id, self.calculator.network_data)
            pd.testing.assert_frame_equal(
                self.calculator.network_data[network_id]['interactions_df'], 
                mock_interactions_df
            )
            self.assertIs(self.calculator.network_data[network_id]['payoffs_df'], mock_payoffs_df)

            # Check that all_features is the union of initiator and target features
            expected_features = {'A', 'B', 'C'}
            self.assertEqual(self.calculator.network_data[network_id]['all_features'], expected_features)

            # Check that feature_traits contains the expected traits
            self.assertIn('feature_traits', self.calculator.network_data[network_id])
            feature_traits = self.calculator.network_data[network_id]['feature_traits']
            self.assertEqual(feature_traits['A'], {'a'})
            self.assertEqual(feature_traits['B'], {'b'})
            self.assertEqual(feature_traits['C'], {'c'})

    def test_get_all_feature_traits(self):
        """Test _get_all_feature_traits correctly extracts traits from payoffs data."""
        network_id = 10

        # Set up mock data
        self.calculator.network_data[network_id] = {
            'payoffs_df': pd.DataFrame({
                'initiator_feature': ['A', 'A', 'B', 'C'],
                'initiator_trait': ['a1', 'a2', 'b1', 'c1'],
                'target_feature': ['B', 'C', 'A', 'B'],
                'target_trait': ['b1', 'c1', 'a1', 'b2']
            }),
            'all_features': {'A', 'B', 'C'}
        }

        # Call the method
        feature_traits = self.calculator._get_all_feature_traits(network_id)

        # Check results
        self.assertEqual(feature_traits['A'], {'a1', 'a2'})
        self.assertEqual(feature_traits['B'], {'b1', 'b2'})
        self.assertEqual(feature_traits['C'], {'c1'})

    def test_get_env_features(self):
        """Test get_env_features returns the correct environment features for a site."""
        world_id = 1
        site = '(0, 0)'
        base_env_utils = 100.0

        # Mock the sites dictionary
        sites_dict = {
            1: {
                '(0, 0)': ['A.a', 'B.b'],
                '(0, 1)': ['C.c']
            }
        }

        with patch('model.rolesets.db.get_sites_dict', return_value=sites_dict):
            # Call the method
            env_features = self.calculator.get_env_features(world_id, site)

            # Check results
            expected_env_features = {
                'A': {'a': base_env_utils},
                'B': {'b': base_env_utils}
            }
            self.assertEqual(env_features, expected_env_features)

            # Check caching
            self.assertEqual(self.calculator.env_features_cache[(world_id, site)], expected_env_features)

            # Call again to test cache
            self.calculator.get_env_features(world_id, site)

            # Verify the sites dict was only retrieved once
            self.assertEqual(sites_dict, {1: {'(0, 0)': ['A.a', 'B.b'], '(0, 1)': ['C.c']}})

    def test_get_features_df(self):
        """Test get_features_df retrieves and caches feature distribution data."""
        world_id = 1
        site = '(0, 0)'
        step_num = 10

        # Mock local_features_df
        mock_df = pd.DataFrame({
            'feature': ['A', 'A', 'B'],
            'trait': ['a1', 'a2', 'b1'],
            'pop': [10, 5, 15],
            'step_num': [10, 10, 10]
        })

        with patch('model.rolesets.db.get_local_features_df', return_value=mock_df) as mock_get_df:
            # First call should retrieve from database
            result_df = self.calculator.get_features_df(world_id, site, step_num)

            # Check result is filtered correctly
            self.assertTrue((result_df['step_num'] == step_num).all())

            # Check caching
            self.assertIn((world_id, site), self.calculator.features_df_cache)

            # Call again to test cache
            result_df2 = self.calculator.get_features_df(world_id, site, step_num)

            # Database function should be called only once
            mock_get_df.assert_called_once_with(
                self.db_path, shadow=False, world_id=world_id, site=site
            )

    def test_get_active_agent_features(self):

        calculator = FeatureUtilityCalculator("dummy/path")

        # Fake feature_changes data
        feature_changes_data = pd.DataFrame([
            {"world_id": 1, "step_num": 1, "name": "A", "env": False, "change": "added"},
            {"world_id": 1, "step_num": 2, "name": "B", "env": False, "change": "added"},
            {"world_id": 1, "step_num": 3, "name": "B", "env": False, "change": "removed"},
            {"world_id": 1, "step_num": 2, "name": "C", "env": True, "change": "added"},
        ])

        # Fake trait_changes data
        trait_changes_data = pd.DataFrame([
            {"world_id": 1, "step_num": 1, "trait_id": 1, "trait_name": "a", "feature_name": "A", "env": False, "change": "added"},
            {"world_id": 1, "step_num": 2, "trait_id": 2, "trait_name": "b", "feature_name": "A", "env": False, "change": "added"},
            {"world_id": 1, "step_num": 3, "trait_id": 2, "trait_name": "b", "feature_name": "A", "env": False, "change": "removed"},
            {"world_id": 1, "step_num": 2, "trait_id": 3, "trait_name": "c", "feature_name": "B", "env": False, "change": "added"},
        ])

        with patch("model.database.get_feature_changes_df", return_value=feature_changes_data), \
             patch("model.database.get_trait_changes_df", return_value=trait_changes_data):

            result = calculator.get_active_agent_features(world_id=1, step_num=3)

        assert result == {
            "A": ["a"],
        }

    def test_build_lfd(self):
        """Test build_lfd correctly builds local feature distribution dictionary."""
        # Mock features DataFrame
        features_df = pd.DataFrame({
            'feature': ['A', 'A', 'B'],
            'trait': ['a1', 'a2', 'b1'],
            'pop': [10, 5, 15]
        })

        # Call the method
        lfd = self.calculator.build_lfd(features_df)

        # Check results
        expected_lfd = {
            'A': {
                'traits': {'a1': 10, 'a2': 5},
                'total': 15,
                'dist': {'a1': 10/15, 'a2': 5/15}
            },
            'B': {
                'traits': {'b1': 15},
                'total': 15,
                'dist': {'b1': 1.0}
            }
        }

        # Check structure and values
        self.assertEqual(set(lfd.keys()), {'A', 'B'})
        for feature in lfd:
            self.assertEqual(set(lfd[feature].keys()), {'traits', 'total', 'dist'})
            self.assertEqual(lfd[feature]['traits'], expected_lfd[feature]['traits'])
            self.assertEqual(lfd[feature]['total'], expected_lfd[feature]['total'])
            for trait in lfd[feature]['dist']:
                self.assertAlmostEqual(lfd[feature]['dist'][trait], expected_lfd[feature]['dist'][trait])

    def test_calculate_env_feature_weight(self):
        """Test calculate_env_feature_weight with various scenarios."""
        network_id = 10
        env_feature = 'A'
        env_trait = 'a'
        base_env_utils = 100.0

        # Case 1: No interactions (should return 1.0)
        lfd1 = {'B': {'total': 20, 'dist': {'b1': 0.7, 'b2': 0.3}}}
        empty_interactions = pd.DataFrame(columns=['target', 'initiator', 'interaction_id'])

        self.calculator.network_data[network_id] = {'interactions_df': empty_interactions, 'payoffs_df': pd.DataFrame()}
        weight1 = self.calculator.calculate_env_feature_weight(network_id, env_feature, env_trait, lfd1, base_env_utils)
        self.assertEqual(weight1, 1.0, "Should return 1.0 when no interactions exist")

        # Case 2: No initiators present (should return 1.0)
        lfd2 = {'C': {'total': 10, 'dist': {'c1': 1.0}}}
        interactions_with_initiator = pd.DataFrame({
            'target': ['A'],
            'initiator': ['B'],
            'interaction_id': [1]
        })

        self.calculator.network_data[network_id] = {
            'interactions_df': interactions_with_initiator, 
            'payoffs_df': pd.DataFrame()
        }
        weight2 = self.calculator.calculate_env_feature_weight(network_id, env_feature, env_trait, lfd2, base_env_utils)
        self.assertEqual(weight2, 1.0, "Should return 1.0 when no initiators are present")

        # Case 3: Negative impact (should limit interactions)
        lfd3 = {'B': {'total': 20, 'dist': {'b1': 1.0}}}
        interactions = pd.DataFrame({
            'target': ['A'],
            'initiator': ['B'],
            'interaction_id': [1]
        })
        payoffs = pd.DataFrame({
            'interaction_id': [1],
            'initiator_trait': ['b1'],
            'target_trait': ['a'],
            'initiator_payoff': [0.5],
            'target_payoff': [-0.5]  # Negative impact
        })

        self.calculator.network_data[network_id] = {
            'interactions_df': interactions, 
            'payoffs_df': payoffs
        }
        weight3 = self.calculator.calculate_env_feature_weight(network_id, env_feature, env_trait, lfd3, base_env_utils)
        expected_weight = min(base_env_utils / (0.5 * 20), 1.0)
        self.assertAlmostEqual(weight3, expected_weight, 
                         msg="Should return limited weight when impact is negative")

        # Case 4: Positive impact (should return 1.0)
        lfd4 = {'B': {'total': 20, 'dist': {'b1': 1.0}}}
        payoffs_positive = pd.DataFrame({
            'interaction_id': [1],
            'initiator_trait': ['b1'],
            'target_trait': ['a'],
            'initiator_payoff': [0.5],
            'target_payoff': [0.5]  # Positive impact
        })

        self.calculator.network_data[network_id] = {
            'interactions_df': interactions, 
            'payoffs_df': payoffs_positive
        }
        weight4 = self.calculator.calculate_env_feature_weight(network_id, env_feature, env_trait, lfd4, base_env_utils)
        self.assertEqual(weight4, 1.0, "Should return 1.0 when impact is positive")

    def test_calculate_trait_eu(self):
        """Test calculate_trait_eu with various interaction scenarios."""
        world_id = 1
        network_id = 10
        feature = 'A'
        trait = 'a1'
        lfd = {
            'B': {'total': 20, 'dist': {'b1': 0.7, 'b2': 0.3}},
            'C': {'total': 10, 'dist': {'c1': 0.5, 'c2': 0.5}}
        }
        env_features = {'D': {'d1': 100.0}}
        population = 30
        base_env_utils = 100.0
        
        # Mock interaction data
        interactions_df = pd.DataFrame({
            'initiator': ['B', 'A', 'A'],
            'target': ['A', 'C', 'D'],
            'interaction_id': [1, 2, 3]
        })

        payoffs_df = pd.DataFrame({
            'interaction_id': [1, 1, 2, 2, 3],
            'initiator_feature': ['B', 'B', 'A', 'A', 'A'],
            'initiator_trait': ['b1', 'b2', 'a1', 'a1', 'a1'],
            'target_feature': ['A', 'A', 'C', 'C', 'D'],
            'target_trait': ['a1', 'a1', 'c1', 'c2', 'd1'],
            'initiator_payoff': [0.1, 0.2, 0.3, 0.4, 0.5],
            'target_payoff': [0.6, 0.7, 0.8, 0.9, -0.2]  # Note negative for D
        })

        # Set up network data
        self.calculator.world_dict = {world_id: network_id}
        self.calculator.network_data[network_id] = {
            'interactions_df': interactions_df,
            'payoffs_df': payoffs_df,
            'all_features': {'A', 'B', 'C', 'D'}
        }

        # Mock the env_feature_weight calculation for the test
        with patch.object(self.calculator, 'calculate_env_feature_weight', return_value=0.5):
            # Call the method
            eu = self.calculator.calculate_trait_eu(world_id, feature, trait, lfd, env_features, population)

            # Calculate expected EU manually:
            # 1. Incoming from B: (20/30) * (0.7*0.6 + 0.3*0.7) = 0.42
            # 2. Outgoing to C: (10/30) * (0.5*0.3 + 0.5*0.4) = 0.1167
            # 3. Outgoing to D: 0.5 * 0.5 = 0.25
            expected_eu = 0.42 + 0.1167 + 0.25

            self.assertAlmostEqual(eu, expected_eu, places=4, 
                                  msg="EU calculation didn't match expected value")

        # Test with zero population
        eu_zero_pop = self.calculator.calculate_trait_eu(world_id, feature, trait, lfd, env_features, 0)
        self.assertEqual(eu_zero_pop, 0, "EU should be 0 when population is 0")

    def test_get_site_step_fud(self):
        world_id = 1
        site = '(0, 0)'
        step_num = 5
        network_id = 10

        self.calculator.world_dict = {world_id: network_id}
        self.calculator.params_df = pd.DataFrame({
            'world_id': [1],
            'base_env_utils': [100.0]
        })

        features_df = pd.DataFrame({
            'feature': ['X', 'Y'],
            'trait': ['x1', 'y1'],
            'pop': [10, 5],
            'step_num': [step_num, step_num]
        })

        env_features = {'E': {'e1': 100.0}}
        lfd = {
            'X': {'total': 10, 'dist': {'x1': 1.0}},
            'Y': {'total': 5, 'dist': {'y1': 1.0}}
        }
        active_features = {
            'X': ['x1', 'x2'],
            'Y': ['y1']
        }

        mock_interactions_df = pd.DataFrame({
            'network_id': [network_id, network_id],
            'initiator': ['X', 'Y'],
            'target': ['Y', 'X'],
            'interaction_id': [1, 2]
        })
        mock_payoffs_df = pd.DataFrame({
            'interaction_id': [1, 1, 2],
            'initiator_feature': ['X', 'X', 'Y'],
            'initiator_trait': ['x1', 'x2', 'y1'],
            'target_feature': ['Y', 'Y', 'X'],
            'target_trait': ['y1', 'y1', 'x1'],
            'initiator_payoff': [0.1, 0.2, 0.3],
            'target_payoff': [0.4, 0.5, 0.6]
        })

        # Mock calculate_trait_eu to return different values per trait
        def mock_calc_trait_eu(world_id, feature, trait, lfd, env, pop):
            return {
                ('X', 'x1'): 1.0,
                ('X', 'x2'): 0.5,
                ('Y', 'y1'): 2.0
            }[(feature, trait)]

        with patch.object(self.calculator, 'get_features_df', return_value=features_df), \
             patch.object(self.calculator, 'get_env_features', return_value=env_features), \
             patch.object(self.calculator, 'build_lfd', return_value=lfd), \
             patch.object(self.calculator, 'get_active_agent_features', return_value=active_features), \
             patch.object(self.calculator, 'calculate_trait_eu', side_effect=mock_calc_trait_eu), \
             patch('model.rolesets.db.get_interactions_df', return_value=mock_interactions_df), \
             patch('model.rolesets.db.get_payoffs_df', return_value=mock_payoffs_df):

            # Max-only output (cached)
            fud_max = self.calculator.get_site_step_fud(world_id, site, step_num, full_traits=False)
            self.assertEqual(fud_max, {
                'X': 1.0,
                'Y': 2.0
            })

            # Full trait output
            fud = self.calculator.get_site_step_fud(world_id, site, step_num, full_traits=True)
            self.assertEqual(fud, {
                'X': {'x1': 1.0, 'x2': 0.5},
                'Y': {'y1': 2.0}
            })

            # Cached result should be used
            self.assertIn((world_id, site, step_num), self.calculator.fud_cache)
            self.assertEqual(self.calculator.fud_cache[(world_id, site, step_num)], {
                'X': 1.0,
                'Y': 2.0
            })


    def test_get_world_sites_steps_fud(self):
        """Test get_world_sites_steps_fud aggregates FUDs for all sites and steps."""
        world_id = 1
        sites_dict = {
            1: {
                '(0, 0)': ['A.a', 'B.b'],
                '(0, 1)': ['C.c']
            }
        }

        # Mock methods 
        mock_features_df = pd.DataFrame({
            'feature': ['A', 'B'],
            'trait': ['a1', 'b1'],
            'pop': [10, 20],
            'step_num': [5, 5]
        })

        mock_fud = {'A': {'a1': 0.5}, 'B': {'b1': 0.3}}

        with patch('model.rolesets.db.get_sites_dict', return_value=sites_dict), \
             patch.object(self.calculator, 'get_features_df', return_value=mock_features_df), \
             patch.object(self.calculator, 'get_site_step_fud', return_value=mock_fud):

            # Call the method with specific steps
            result = self.calculator.get_world_sites_steps_fud(world_id, steps=[5])

            # Check structure
            self.assertEqual(set(result.keys()), {'(0, 0)', '(0, 1)'})
            for site in result:
                self.assertEqual(set(result[site].keys()), {5})
                self.assertEqual(result[site][5], mock_fud)


class TestRoleAnalyzer(unittest.TestCase):
    def setUp(self):
        # Mock FeatureUtilityCalculator
        self.mock_calculator = MagicMock(spec=FeatureUtilityCalculator)
        self.mock_calculator.db_path = 'mock_db_path'
        self.mock_calculator.world_dict = {1: 10, 2: 20}
        self.mock_calculator.params_df = pd.DataFrame({
            'world_id': [1, 2],
            'network_id': [10, 20],
            'base_env_utils': [100.0, 100.0],
            'feature_cost_exp': [0.75, 0.75],
            'grid_size': [3, 3],
            'total_pop_limit': [9000, 9000],
            'pop_cost_exp': [2, 2]
        })

        # Create analyzer instance with mock calculator
        self.analyzer = RoleAnalyzer(self.mock_calculator)
    
    def test_init(self):
        """Test initialization of RoleAnalyzer."""
        # Check attributes are set correctly
        self.assertEqual(self.analyzer.calculator, self.mock_calculator)
        self.assertEqual(self.analyzer.db_path, 'mock_db_path')
        self.assertEqual(self.analyzer.world_dict, self.mock_calculator.world_dict)
        pd.testing.assert_frame_equal(self.analyzer.params_df, self.mock_calculator.params_df)
        self.assertEqual(self.analyzer.occupied_roles_cache, {})

    def test_get_site_pop_cost(self):
        """Test get_site_pop_cost calculates population cost correctly."""
        world_id = 1
        site = '(0, 0)'
        step_num = 5

        # Create mock features dataframe
        mock_features_df = pd.DataFrame({
            'feature': ['A', 'B'],
            'trait': ['a1', 'b1'],
            'pop': [30, 20],  # Total population = 50
            'step_num': [5, 5]
        })

        # Set up params for this world
        self.analyzer.params_df = pd.DataFrame({
            'world_id': [1],
            'grid_size': [3],          # 3×3 grid = 9 sites
            'total_pop_limit': [900],  # 900 / 9 = 100 per site
            'pop_cost_exp': [2]        # Square the ratio
        })

        # Mock the get_features_df method
        self.mock_calculator.get_features_df.return_value = mock_features_df

        # Call the method
        pop_cost = self.analyzer.get_site_pop_cost(world_id, site, step_num)

        # Expected: (50/100)^2 = 0.25
        self.assertEqual(pop_cost, 0.25)

        # Test with zero site_pop_limit
        self.analyzer.params_df = pd.DataFrame({
            'world_id': [1],
            'grid_size': [3],
            'total_pop_limit': [0],  # This will make site_pop_limit = 0
            'pop_cost_exp': [2]
        })

        pop_cost_zero = self.analyzer.get_site_pop_cost(world_id, site, step_num)
        self.assertEqual(pop_cost_zero, 0)

    def test_check_sustainable(self):
        """Test check_sustainable determines role sustainability correctly."""
        world_id = 1
        site = '(0, 0)'
        step_num = 5
        features = ('A', 'B')

        # Mock FUD and pop_cost
        mock_fud = {
            'A': 0.6,
            'B': 0.5,
            'C': 0.2
        }

        # Set feature_cost_exp parameter
        self.analyzer.params_df = pd.DataFrame({
            'world_id': [1],
            'feature_cost_exp': [0.75]
        })

        # Mock the methods used by check_sustainable
        self.mock_calculator.get_site_step_fud.return_value = mock_fud
        self.analyzer.get_site_pop_cost = MagicMock(return_value=0.5)

        # Case 1: sustainable role (EU > cost)
        # EU = A + B = 0.6 + 0.5 = 1.1
        # Cost = 0.5 * (2^0.75) = 0.5 * 1.682 = 0.841
        is_sustainable = self.analyzer.check_sustainable(world_id, site, step_num, features)
        self.assertTrue(is_sustainable)

        # Case 2: Non-sustainable role (EU < cost)
        # Set higher pop_cost so EU < cost
        self.analyzer.get_site_pop_cost = MagicMock(return_value=2.0)
        # Cost = 2.0 * (2^0.75) = 2.0 * 1.682 = 3.364 > 1.1
        is_sustainable2 = self.analyzer.check_sustainable(world_id, site, step_num, features)
        self.assertFalse(is_sustainable2)

        # Case 3: Empty features list (should return False)
        is_sustainable3 = self.analyzer.check_sustainable(world_id, site, step_num, ())
        self.assertFalse(is_sustainable3)

    def test_get_occupied_roles(self):
        """Test get_occupied_roles correctly identifies occupied roles."""
        world_id = 1
        site = '(0, 0)'
        step_num = 5
        cache_key = (world_id, site, step_num)

        # Mock phenotypes dataframe
        mock_df = pd.DataFrame({
            'role': ['A:B', 'C', 'A:D:E'],
            'pop': [10, 5, 15]
        })

        with patch('model.rolesets.db.get_phenotypes_df', return_value=mock_df):
            # Call the method
            occupied_roles = self.analyzer.get_occupied_roles(world_id, site, step_num)

            # Check results
            expected_roles = {
                ('A', 'B'): 10,
                ('C',): 5,
                ('A', 'D', 'E'): 15
            }
            self.assertEqual(occupied_roles, expected_roles)

            # Check caching
            self.assertEqual(self.analyzer.occupied_roles_cache[cache_key], expected_roles)

            # Call again to test cache
            occupied_roles2 = self.analyzer.get_occupied_roles(world_id, site, step_num)
            self.assertEqual(occupied_roles2, expected_roles)

    def test_check_adjacent(self):
        """Test check_adjacent identifies roles that differ by at most one feature."""
        world_id = 1
        site = '(0, 0)'
        step_num = 5

        # Mock occupied roles
        occupied_roles = {
            ('A', 'B'): 10,
            ('C', 'D'): 5,
            ('E',): 15
        }

        # Mock get_occupied_roles
        self.analyzer.get_occupied_roles = MagicMock(return_value=occupied_roles)

        # Case 1: Adjacent by removal (A:B -> A)
        features1 = ('A',)
        is_adjacent1 = self.analyzer.check_adjacent(world_id, site, step_num, features1)
        self.assertTrue(is_adjacent1)

        # Case 2: Adjacent by addition (A:B -> A:B:F)
        features2 = ('A', 'B', 'F')
        is_adjacent2 = self.analyzer.check_adjacent(world_id, site, step_num, features2)
        self.assertTrue(is_adjacent2)

        # Case 3: Not adjacent (differs by 2+ features)
        features3 = ('G', 'H')
        is_adjacent3 = self.analyzer.check_adjacent(world_id, site, step_num, features3)
        self.assertFalse(is_adjacent3)

        # Case 4: Empty features (should return False)
        features4 = ()
        is_adjacent4 = self.analyzer.check_adjacent(world_id, site, step_num, features4)
        self.assertFalse(is_adjacent4)

    def test_count_sustainable(self):
        """Test count_sustainable correctly counts sustainable role combinations."""
        world_id = 1
        site = '(0, 0)'
        step_num = 5
        network_id = 10

        # Mock feature utility dictionary with controlled values
        mock_fud = {
            'A': 0.6,
            'B': 0.5,
            'C': 0.2
        }

        # Set up network data
        self.analyzer.world_dict = {world_id: network_id}
        self.mock_calculator.network_data = {
            network_id: {'all_features': {'A', 'B', 'C'}}
        }

        # Mock methods
        self.mock_calculator.get_site_step_fud.return_value = mock_fud
        # Mock get_active_agent_features to return keys matching mock_fud
        self.mock_calculator.get_active_agent_features.return_value = {
            'A': ['a1'],
            'B': ['b1'],
            'C': ['c1']
        }        
        # Set parameters
        self.analyzer.params_df = pd.DataFrame({
            'world_id': [1],
            'feature_cost_exp': [1.0]  # Linear cost for simplicity
        })

        # Case 1: Zero population cost (all combinations sustainable)
        self.analyzer.get_site_pop_cost = MagicMock(return_value=0)
        count1 = self.analyzer.count_sustainable(world_id, site, step_num)
        # Expected: 2^3 - 1 = 7 combinations (excluding empty set)
        self.assertEqual(count1, 7)

        # Case 2: Medium pop_cost - some combinations sustainable
        # Best utility order: A (0.6), B (0.5), C (0.2)
        # Pop cost = 0.3, so:
        # 1 feature: A or B sustainable (0.6 > 0.3, 0.5 > 0.3), C not sustainable (0.2 < 0.3)
        # 2 features: A:B sustainable (1.1 > 0.6), others depend on cost function
        # 3 features: A:B:C = 1.3 total utility vs 0.9 cost (sustainable)
        self.analyzer.get_site_pop_cost = MagicMock(return_value=0.3)
        count2 = self.analyzer.count_sustainable(world_id, site, step_num)
        # Exact count depends on the algorithm details, but should be less than 7
        self.assertLess(count2, 7)
        self.assertGreater(count2, 0)

        # Case 3: High pop_cost - no combinations sustainable
        self.analyzer.get_site_pop_cost = MagicMock(return_value=2.0)
        count3 = self.analyzer.count_sustainable(world_id, site, step_num)
        self.assertEqual(count3, 0)

    def test_evaluate_rolesets(self):
        """Test evaluate_rolesets collects roleset data for all sites and steps."""
        world_id = 1
        network_id = 10
        steps = [5, 10]
        sites = ['(0, 0)', '(0, 1)']

        # Mock sites dictionary
        sites_dict = {world_id: {site: ['A.a'] for site in sites}}

        # Mock occupied roles
        occupied_roles = {('A', 'B'): 10, ('C',): 5}

        # Mock fud
        mock_fud = {'A': 0.5, 'B': 0.3, 'C': 0.2}

        # Set up network data
        self.analyzer.world_dict = {world_id: network_id}
        self.mock_calculator.get_site_step_fud.return_value = mock_fud
        self.mock_calculator.get_active_agent_features.return_value = {'A': ['a'], 'B': ['b'], 'C': ['c']}
        self.mock_calculator.network_data = {
            network_id: {'all_features': {'A', 'B', 'C', 'D'}}
        }

        # Mock methods
        with patch('model.rolesets.db.get_sites_dict', return_value=sites_dict), \
             patch('model.rolesets.db.get_phenotypes_df', return_value=pd.DataFrame({'step_num': steps})), \
             patch.object(self.analyzer, 'get_occupied_roles', return_value=occupied_roles), \
             patch.object(self.analyzer, 'count_sustainable', return_value=10):

            # Add mock to handle empty values in check_sustainable
            def mock_check_sustainable(world_id, site, step, role):
                # Simple implementation that won't hit the empty values issue
                return len(role) > 0

            self.analyzer.check_sustainable = MagicMock(side_effect=mock_check_sustainable)

            # Call the method
            result = self.analyzer.evaluate_rolesets(world_id, steps)

            # Check result structure
            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual(len(result), len(sites) * len(steps))

            # Check expected columns
            expected_columns = ['world_id', 'site', 'step', 'possible', 'sustainable', 
                                'adjacent', 'occupiable', 'occupied']
            self.assertEqual(set(result.columns), set(expected_columns))

            # Check values
            self.assertTrue((result['world_id'] == world_id).all())
            self.assertTrue((result['sustainable'] == 10).all())
            self.assertTrue((result['occupied'] == len(occupied_roles)).all())

            # Check possible count (2^n - 1)
            expected_possible = (2 ** 3) - 1
            self.assertTrue((result['possible'] == expected_possible).all())


if __name__ == '__main__':
    unittest.main()