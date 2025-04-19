import unittest
import pandas as pd
import numpy as np
from model.analysis import (add_evolutionary_activity_columns,
    shannon_entropy, identify_adaptive_entities, calculate_evolutionary_activity_stats)

class TestShannonEntropy(unittest.TestCase):

    def test_uniform_distribution(self):
        """Test shannon_entropy with a uniform distribution."""
        proportions = [0.25, 0.25, 0.25, 0.25]
        entropy = shannon_entropy(proportions)
        self.assertAlmostEqual(entropy, 2.0, places=6, 
                              msg="Uniform distribution should have maximum entropy")

    def test_skewed_distribution(self):
        """Test shannon_entropy with a completely skewed distribution."""
        proportions = [1.0, 0.0, 0.0]
        entropy = shannon_entropy(proportions)
        self.assertEqual(entropy, 0.0, 
                        msg="Distribution with single non-zero value should have zero entropy")

    def test_mixed_distribution(self):
        """Test shannon_entropy with a mixed distribution."""
        proportions = [0.5, 0.25, 0.25]
        # Expected entropy: -0.5*log2(0.5) - 0.25*log2(0.25) - 0.25*log2(0.25) = 0.5 + 0.5 + 0.5 = 1.5
        expected_entropy = 1.5
        entropy = shannon_entropy(proportions)
        self.assertAlmostEqual(entropy, expected_entropy, places=6,
                              msg="Mixed distribution entropy calculation is incorrect")

    def test_empty_list(self):
        """Test shannon_entropy with an empty list."""
        proportions = []
        entropy = shannon_entropy(proportions)
        self.assertEqual(entropy, 0.0, 
                        msg="Empty list should have zero entropy")

    def test_single_proportion(self):
        """Test shannon_entropy with a single proportion."""
        proportions = [1.0]
        entropy = shannon_entropy(proportions)
        self.assertEqual(entropy, 0.0, 
                        msg="Single proportion should have zero entropy")

    def test_small_values(self):
        """Test shannon_entropy with very small values for numerical stability."""
        # Use very small values that sum to 1
        proportions = [1e-10, 1 - 1e-10]
        # Expected entropy: -1e-10*log2(1e-10) - (1-1e-10)*log2(1-1e-10) ≈ 33.2 * 1e-10 ≈ 0
        entropy = shannon_entropy(proportions)
        self.assertGreaterEqual(entropy, 0.0, 
                               msg="Very small values should be handled stably")
        self.assertLess(entropy, 1e-8, 
                       msg="Very skewed distribution should have very small entropy")

    def test_zero_handling(self):
        """Test how zeros are handled in the distribution."""
        # Zeros should be ignored in entropy calculation
        proportions = [0.5, 0.5, 0.0, 0.0]
        expected_entropy = 1.0  # -0.5*log2(0.5) - 0.5*log2(0.5) = 0.5 + 0.5 = 1.0
        entropy = shannon_entropy(proportions)
        self.assertAlmostEqual(entropy, expected_entropy, places=6,
                              msg="Zeros should be ignored in entropy calculation")

    def test_numpy_array_input(self):
        """Test shannon_entropy with numpy array input."""
        proportions = np.array([0.25, 0.25, 0.25, 0.25])
        entropy = shannon_entropy(proportions)
        self.assertAlmostEqual(entropy, 2.0, places=6,
                              msg="Numpy array input should be handled correctly")

class TestEvolutionaryActivityColumns(unittest.TestCase):

    def test_add_evolutionary_activity_columns(self):
        """Test that add_evolutionary_activity_columns produces expected results for all columns"""

        # Create a controlled test dataframe with predictable values
        data = [
            # world_id, step_num, phenotype, role, pop
            # World 1, Step 1
            (1, 1, "P1", "R1", 10),
            (1, 1, "P2", "R1", 20),
            (1, 1, "P3", "R2", 30),
            
            # World 1, Step 2
            (1, 2, "P1", "R1", 15),  # P1 grew by 5
            (1, 2, "P2", "R1", 15),  # P2 decreased by 5
            (1, 2, "P3", "R2", 45),  # P3 grew by 15
            (1, 2, "P4", "R2", 5),   # New phenotype appeared
            
            # World 1, Step 3 
            (1, 3, "P1", "R1", 20),  # P1 continues to grow
            (1, 3, "P2", "R1", 10),  # P2 continues to decrease
            (1, 3, "P3", "R2", 60),  # P3 continues to grow
            (1, 3, "P4", "R2", 10),  # P4 doubles
            
            # World 2, Step 1
            (2, 1, "P5", "R3", 25),
            (2, 1, "P6", "R3", 15),
            
            # World 2, Step 2
            (2, 2, "P5", "R3", 20),  # P5 decreased
            (2, 2, "P6", "R3", 30)   # P6 doubled
        ]

        df = pd.DataFrame(data, columns=['world_id', 'step_num', 'phenotype', 'role', 'pop'])

        # Apply the function to test (no snap_interval for simplicity)
        result = add_evolutionary_activity_columns(df.copy())

        # Check that all expected columns exist, including intermediate ones
        expected_columns = [
            'phenotype_activity', 'phenotype_cum_pop', 'phenotype_growth_rate', 'phenotype_delta_N',
            'role_activity', 'role_cum_pop', 'role_growth_rate', 'role_delta_N'
        ]

        for col in expected_columns:
            self.assertIn(col, result.columns, f"Column {col} missing from result")

        # Test phenotype activity (persistence counter)
        # P1 should have activity 1, 2, 3 across steps 1, 2, 3
        p1_activity = result[result['phenotype'] == 'P1']['phenotype_activity'].values
        self.assertTrue(np.array_equal(p1_activity, [1, 2, 3]), 
                       f"Expected P1 activity [1, 2, 3], got {p1_activity}")

        # Test phenotype cumulative population
        # P1 should have cum_pop 10, 25, 45 across steps 1, 2, 3
        p1_cum_pop = result[result['phenotype'] == 'P1']['phenotype_cum_pop'].values
        self.assertTrue(np.array_equal(p1_cum_pop, [10, 25, 45]), 
                       f"Expected P1 cum_pop [10, 25, 45], got {p1_cum_pop}")

        # Test phenotype growth rate
        # P1 growth rate at step 2 should be (15-10)/10 = 0.5
        p1_growth_step2 = result[(result['phenotype'] == 'P1') & 
                                  (result['step_num'] == 2)]['phenotype_growth_rate'].values[0]
        self.assertAlmostEqual(p1_growth_step2, 0.5, places=6,
                             msg=f"Expected P1 step 2 growth rate 0.5, got {p1_growth_step2}")

        # Test phenotype delta_N (non-neutral selection)
        # Calculate expected delta_N for P3 at step 2:
        # Step 1 total pop for world 1 = 60, P3 pop = 30, so expected_prop = 0.5
        # Step 2 total pop for world 1 = 80, P3 pop = 45, so observed_prop = 0.5625
        # delta_N = 80 * (0.5625 - 0.5)^2 = 80 * 0.0039 = 0.312
        p3_delta_step2 = result[(result['phenotype'] == 'P3') & 
                               (result['step_num'] == 2)]['phenotype_delta_N'].values[0]
        self.assertAlmostEqual(p3_delta_step2, 0.3125, places=6, 
                             msg=f"Expected P3 step 2 delta_N ~0.312, got {p3_delta_step2}")

        # First check that all role metrics are consistent across phenotypes in the same role
        for step in [1, 2, 3]:
            for role in ['R1', 'R2']:
                role_rows = result[(result['role'] == role) & (result['step_num'] == step)]
                if len(role_rows) > 0:  # Only check if this role exists in this step
                    role_metrics = ['role_activity', 'role_growth_rate', 'role_delta_N', 'role_cum_pop']
                    for metric in role_metrics:
                        unique_vals = role_rows[metric].nunique()
                        self.assertEqual(unique_vals, 1, 
                                       f"{metric} should be identical for all phenotypes in role {role} at step {step}")

        # Test role activity
        # R1 should have activity 1, 2, 3 across steps 1, 2, 3
        r1_activity_s3 = result[(result['role'] == 'R1') & 
                              (result['step_num'] == 3)]['role_activity'].iloc[0]
        self.assertEqual(r1_activity_s3, 3, 
                       f"Expected R1 step 3 activity to be 3, got {r1_activity_s3}")

        # Test role cumulative population 
        # For R1: step 1 = 30, step 2 = 30, step 3 = 30
        # So at step 3, cum_pop should be 90
        r1_cum_pop_s3 = result[(result['role'] == 'R1') & 
                              (result['step_num'] == 3)]['role_cum_pop'].iloc[0]
        self.assertEqual(r1_cum_pop_s3, 90, 
                       f"Expected R1 step 3 cum_pop to be 90, got {r1_cum_pop_s3}")

        # Test role delta_N
        # For R2 at step 2:
        # Step 1 total pop = 60, R2 pop = 30, so expected_prop = 0.5
        # Step 2 total pop = 80, R2 pop = 50, so observed_prop = 0.625
        # delta_N = 80 * (0.625 - 0.5)^2 = 80 * 0.0156 = 1.25
        r2_delta_s2 = result[(result['role'] == 'R2') & 
                            (result['step_num'] == 2)]['role_delta_N'].iloc[0]
        self.assertAlmostEqual(r2_delta_s2, 1.25, places=5,
                             msg=f"Expected R2 step 2 delta_N ~1.25, got {r2_delta_s2}")

        # Test role_growth_rate
        # R1 growth rate at step 2: (30-30)/30 = 0.0
        r1_growth_s2 = result[(result['role'] == 'R1') & 
                             (result['step_num'] == 2)]['role_growth_rate'].iloc[0]
        self.assertAlmostEqual(r1_growth_s2, 0.0, places=6,
                            msg=f"Expected R1 step 2 growth rate to be 0.0, got {r1_growth_s2}")

        # Test snap_interval functionality
        snap_df = add_evolutionary_activity_columns(df.copy(), snap_interval=2)
        # Check that metrics reset at steps divisible by snap_interval
        self.assertEqual(snap_df[(snap_df['world_id'] == 1) & (snap_df['step_num'] == 2) & 
                                  (snap_df['phenotype'] == 'P3')]['phenotype_delta_N'].values[0], 0,
                         "Delta_N should be zero at snap interval steps")

        # Test zero population handling
        # Add a row with zero population to test data and rerun test
        zero_pop_df = df.copy()
        zero_pop_df = pd.concat([zero_pop_df, pd.DataFrame([
            (1, 3, "P5", "R3", 0)  # New phenotype with zero population
        ], columns=zero_pop_df.columns)])
        zero_result = add_evolutionary_activity_columns(zero_pop_df)
        # Verify no NaN values in result
        self.assertFalse(zero_result.isnull().any().any(), 
                        "Zero population handling should not produce NaN values")

        # Test first step metrics
        first_step_rows = result[result['step_num'] == 1]
        self.assertTrue((first_step_rows['phenotype_growth_rate'] == 0).all(), 
                       "First step growth rates should all be zero")
        self.assertTrue((first_step_rows['phenotype_delta_N'] == 0).all(), 
                       "First step delta_N should all be zero")

        # Test new phenotype handling - P4 first appears at step 2
        p4_step2 = result[(result['phenotype'] == 'P4') & (result['step_num'] == 2)]
        self.assertEqual(p4_step2['phenotype_activity'].values[0], 1,
                        "New phenotype should have activity = 1")
        self.assertEqual(p4_step2['phenotype_cum_pop'].values[0], 5,
                        "New phenotype cum_pop should match its population")
        self.assertEqual(p4_step2['phenotype_growth_rate'].values[0], 0,
                        "New phenotype growth_rate should be 0")

        # Test internal consistency between phenotype and role metrics
        # Role R1 contains phenotypes P1 and P2, so their sum should match
        for step in [1, 2, 3]:
            phenotype_sum = result[(result['step_num'] == step) & 
                                  ((result['phenotype'] == 'P1') | 
                                   (result['phenotype'] == 'P2'))]['pop'].sum()
            role_sum = result[(result['step_num'] == step) & 
                             (result['role'] == 'R1')]['pop'].sum()
            
            self.assertEqual(phenotype_sum, role_sum,
                            f"Sum of phenotype populations should match role population at step {step}")

        # Verify DataFrame integrity - same number of rows
        self.assertEqual(len(df), len(result),
                        "Result should have same number of rows as input")

        # Verify original columns are preserved
        for col in df.columns:
            self.assertIn(col, result.columns,
                         f"Original column {col} should be preserved in result")

    def test_multiple_worlds_with_overlapping_phenotypes(self):
        """Test that metrics are calculated correctly when multiple worlds have overlapping phenotypes."""
        from model.analysis import add_evolutionary_activity_columns

        # Create test data with two worlds sharing some phenotypes
        data = [
            # World 1
            (1, 1, "P1", "R1", 10),
            (1, 1, "P2", "R1", 20),

            # World 1, Step 2
            (1, 2, "P1", "R1", 15),
            (1, 2, "P2", "R1", 25),

            # World 2 (overlapping P1 phenotype)
            (2, 1, "P1", "R1", 30),
            (2, 1, "P3", "R2", 40),

            # World 2, Step 2
            (2, 2, "P1", "R1", 35),
            (2, 2, "P3", "R2", 45)
        ]
        
        df = pd.DataFrame(data, columns=['world_id', 'step_num', 'phenotype', 'role', 'pop'])
        result = add_evolutionary_activity_columns(df.copy())

        # Verify worlds are processed separately
        world1_p1 = result[(result['world_id'] == 1) & (result['phenotype'] == 'P1')]
        world2_p1 = result[(result['world_id'] == 2) & (result['phenotype'] == 'P1')]

        # Check that metrics for the same phenotype differ by world
        self.assertEqual(world1_p1['phenotype_cum_pop'].iloc[1], 25)  # World 1, P1, Step 2: 10 + 15
        self.assertEqual(world2_p1['phenotype_cum_pop'].iloc[1], 65)  # World 2, P1, Step 2: 30 + 35

        # Verify activity is counted per world
        self.assertEqual(world1_p1['phenotype_activity'].iloc[0], 1)  # World 1, P1, Step 1
        self.assertEqual(world1_p1['phenotype_activity'].iloc[1], 2)  # World 1, P1, Step 2
        self.assertEqual(world2_p1['phenotype_activity'].iloc[0], 1)  # World 2, P1, Step 1
        self.assertEqual(world2_p1['phenotype_activity'].iloc[1], 2)  # World 2, P1, Step 2

    def test_snap_interval_reset(self):
        """Test that selection metrics reset at snap intervals."""
        from model.analysis import add_evolutionary_activity_columns

        # Create data with step points at a snap interval boundary
        data = [
            # Step 1
            (1, 1, "P1", "R1", 10),
            (1, 1, "P2", "R1", 20),

            # Step 2
            (1, 2, "P1", "R1", 15),
            (1, 2, "P2", "R1", 15),

            # Step 3 (simulation reset point)
            (1, 3, "P1", "R1", 20),
            (1, 3, "P2", "R1", 10),

            # Step 4
            (1, 4, "P1", "R1", 25),
            (1, 4, "P2", "R1", 5)
        ]

        df = pd.DataFrame(data, columns=['world_id', 'step_num', 'phenotype', 'role', 'pop'])

        # First analyze without snap_interval
        result_no_snap = add_evolutionary_activity_columns(df.copy())

        # Then analyze with snap_interval=3
        result_with_snap = add_evolutionary_activity_columns(df.copy(), snap_interval=3)

        # At step 3 (snap boundary), delta_N and growth_rate should be 0
        step3_metrics = result_with_snap[result_with_snap['step_num'] == 3]
        self.assertTrue(all(step3_metrics['phenotype_delta_N'] == 0), 
                       "Delta_N should be reset to 0 at snap interval")
        self.assertTrue(all(step3_metrics['phenotype_growth_rate'] == 0), 
                       "Growth rate should be reset to 0 at snap interval")

        # At step 3 in no-snap version, these metrics should have non-zero values
        step3_no_snap = result_no_snap[result_no_snap['step_num'] == 3]
        # Since P1 increases and P2 decreases, at least one should show non-zero growth rate
        self.assertTrue(any(step3_no_snap['phenotype_growth_rate'] != 0), 
                       "Without snap interval, growth rate should not be reset")

    def test_phenotype_appearance_disappearance(self):
        """Test dynamics of phenotypes appearing and disappearing between steps."""
        from model.analysis import add_evolutionary_activity_columns

        # Create data with phenotypes appearing and disappearing
        data = [
            # Step 1
            (1, 1, "P1", "R1", 10),

            # Step 2 - P1 continues, P2 appears
            (1, 2, "P1", "R1", 15),
            (1, 2, "P2", "R1", 5),

            # Step 3 - P1 disappears, P2 continues, P3 appears
            (1, 3, "P2", "R1", 10),
            (1, 3, "P3", "R2", 20),

            # Step 4 - P1 reappears, others continue
            (1, 4, "P1", "R1", 5),
            (1, 4, "P2", "R1", 15),
            (1, 4, "P3", "R2", 25)
        ]

        df = pd.DataFrame(data, columns=['world_id', 'step_num', 'phenotype', 'role', 'pop'])
        result = add_evolutionary_activity_columns(df.copy())

        # Test new phenotype appearance (P2 at step 2)
        p2_step2 = result[(result['phenotype'] == 'P2') & (result['step_num'] == 2)]
        self.assertEqual(p2_step2['phenotype_activity'].iloc[0], 1,
                        "Activity counter should start at 1 for new phenotype")
        self.assertEqual(p2_step2['phenotype_growth_rate'].iloc[0], 0,
                        "Growth rate should be 0 for new phenotype")

        # Test when phenotype reappears (P1 at step 4)
        p1_step4 = result[(result['phenotype'] == 'P1') & (result['step_num'] == 4)]
        p1_activities = result[result['phenotype'] == 'P1']['phenotype_activity'].values
        self.assertTrue(np.array_equal(p1_activities, [1, 2, 3]), 
                       f"Expected P1 activities [1, 2, 3], got {p1_activities}")

        # Verify cum_pop includes the missing step
        p1_cum_pops = result[result['phenotype'] == 'P1']['phenotype_cum_pop'].values
        self.assertTrue(np.array_equal(p1_cum_pops, [10, 25, 30]), 
                      f"Expected P1 cum_pops [10, 25, 30], got {p1_cum_pops}")

class TestIdentifyAdaptiveEntities(unittest.TestCase):

    def setUp(self):
        # Create sample dataframes for selection and shadow models
        # Base data with different components (roles and phenotypes) and metrics
        self.selection_data = [
            # world_id, step_num, phenotype, role, pop, role_growth_rate, phenotype_growth_rate, 
            # role_activity, phenotype_activity, role_cum_pop, phenotype_cum_pop
            (1, 10, "P1", "R1", 20, 0.5, 0.4, 3, 2, 50, 40),
            (1, 10, "P2", "R1", 15, 0.5, 0.3, 3, 2, 50, 35),
            (1, 10, "P3", "R2", 25, 0.2, 0.1, 2, 1, 40, 25),

            (2, 10, "P4", "R3", 10, 0.1, 0.05, 1, 1, 20, 10),
            (2, 10, "P5", "R4", 30, 0.8, 0.7, 4, 3, 60, 55),
            (2, 10, "P6", "R4", 20, 0.8, 0.6, 4, 3, 60, 45),
        ]

        self.shadow_data = [
            # Same structure but with different values to test various comparison scenarios
            (1, 10, "P1", "R1", 15, 0.3, 0.2, 2, 1, 30, 20),
            (1, 10, "P2", "R1", 15, 0.3, 0.3, 2, 2, 30, 30),
            (1, 10, "P3", "R2", 30, 0.4, 0.3, 3, 2, 50, 35),
            
            (2, 10, "P4", "R3", 20, 0.2, 0.15, 2, 2, 40, 25),
            (2, 10, "P5", "R4", 10, 0.3, 0.2, 2, 1, 20, 15),
            (2, 10, "P6", "R4", 20, 0.3, 0.4, 2, 3, 20, 35),
        ]

        # Convert to DataFrames
        columns = ['world_id', 'step_num', 'phenotype', 'role', 'pop', 
                  'role_growth_rate', 'phenotype_growth_rate', 
                  'role_activity', 'phenotype_activity',
                  'role_cum_pop', 'phenotype_cum_pop']

        self.selection_df = pd.DataFrame(self.selection_data, columns=columns)
        self.shadow_df = pd.DataFrame(self.shadow_data, columns=columns)

    def test_role_growth_rate(self):
        """Test identification of adaptive entities with 'role' component and 'growth_rate' metric."""
        # Using a mid-range percentile threshold
        adaptive_entities, proportions = identify_adaptive_entities(
            self.selection_df, self.shadow_df, component='role', metric='role_growth_rate', percentile=0.5
        )

        # Verify the correct adaptive roles were identified
        # For world_id=1: R1 in selection has growth_rate=0.5 which exceeds the shadow threshold
        # For world_id=2: R4 in selection has growth_rate=0.8 which exceeds the shadow threshold
        self.assertIn('R1', adaptive_entities['selection'][1].index)
        self.assertIn('R4', adaptive_entities['selection'][2].index)

        # Check that roles below threshold are not included
        self.assertNotIn('R2', adaptive_entities['selection'][1].index)
        self.assertNotIn('R3', adaptive_entities['selection'][2].index)

        # Verify proportions (adaptive roles / total roles)
        # World 1: 1 adaptive role out of 2 total = 0.5
        # World 2: 1 adaptive role out of 2 total = 0.5
        self.assertEqual(proportions[1], 0.5)
        self.assertEqual(proportions[2], 0.5)

    def test_phenotype_activity(self):
        """Test identification of adaptive entities with 'phenotype' component and 'activity' metric."""
        # Using a high percentile threshold
        adaptive_entities, proportions = identify_adaptive_entities(
            self.selection_df, self.shadow_df, component='phenotype', metric='phenotype_activity', percentile=0.9
        )

        # At 0.9 percentile, only the highest activity phenotypes should be identified
        # For world_id=1, no phenotype should exceed the threshold
        # For world_id=2, P5 and P6 should exceed the threshold
        self.assertEqual(len(adaptive_entities['selection'][1]), 0)
        self.assertGreater(len(adaptive_entities['selection'][2]), 0)

        if len(adaptive_entities['selection'][2]) > 0:
            self.assertIn('P5', adaptive_entities['selection'][2].index)
            self.assertIn('P6', adaptive_entities['selection'][2].index)

        # Verify proportions
        # World 1: 0 adaptive phenotypes out of 3 total = 0
        # World 2: 2 adaptive phenotypes out of 3 total ≈ 0.67
        self.assertEqual(proportions[1], 0)
        self.assertAlmostEqual(proportions[2], 2/3, places=2)

    def test_role_cum_pop(self):
        """Test identification of adaptive entities with 'role' component and 'cum_pop' metric."""
        # Using a very high percentile threshold
        adaptive_entities, proportions = identify_adaptive_entities(
            self.selection_df, self.shadow_df, component='role', metric='role_cum_pop', percentile=0.99
        )

        # At 0.99 percentile, only extremely high values should exceed the threshold
        # In this case, likely no roles would be identified
        for world_id in [1, 2]:
            # Either no adaptive roles or very few
            self.assertLessEqual(len(adaptive_entities['selection'][world_id]), 1)

        # Using a low percentile threshold
        adaptive_entities, proportions = identify_adaptive_entities(
            self.selection_df, self.shadow_df, component='role', metric='role_cum_pop', percentile=0.1
        )

        # At 0.1 percentile, most roles should be identified as adaptive
        for world_id in [1, 2]:
            self.assertGreaterEqual(proportions[world_id], 0.5)

    def test_comparison_cases(self):
        """Test different comparison scenarios where selection > shadow, selection < shadow, selection = shadow."""
        # Create test data with controlled comparison cases
        equal_data = [
            (3, 10, "PE1", "RE1", 20, 0.5, 0.5, 3, 3, 50, 50),
            (3, 10, "PE2", "RE2", 20, 0.5, 0.5, 3, 3, 50, 50),
        ]

        # World 4: Selection clearly higher than shadow
        selection_greater_data = [
            (4, 10, "PG1", "RG1", 40, 0.9, 0.9, 6, 6, 90, 90),
            (4, 10, "PG2", "RG2", 10, 0.1, 0.1, 1, 1, 10, 10),
        ]

        shadow_greater_data = [
            (4, 10, "PG1", "RG1", 15, 0.2, 0.2, 2, 2, 20, 20),
            (4, 10, "PG2", "RG2", 15, 0.2, 0.2, 2, 2, 20, 20),
        ]

        # World 5: Shadow has higher values
        selection_data_w5 = [
            (5, 10, "PS1", "RS1", 10, 0.1, 0.1, 1, 1, 10, 10),
            (5, 10, "PS2", "RS2", 10, 0.1, 0.1, 1, 1, 10, 10),
        ]

        shadow_data_w5 = [
            (5, 10, "PS1", "RS1", 40, 0.8, 0.8, 5, 5, 80, 80),
            (5, 10, "PS2", "RS2", 30, 0.6, 0.6, 4, 4, 60, 60),
        ]

        # Create new test DataFrames
        test_selection_data = self.selection_data + equal_data + selection_greater_data + selection_data_w5
        test_shadow_data = self.shadow_data + equal_data + shadow_greater_data + shadow_data_w5

        columns = ['world_id', 'step_num', 'phenotype', 'role', 'pop', 
                  'role_growth_rate', 'phenotype_growth_rate', 
                  'role_activity', 'phenotype_activity',
                  'role_cum_pop', 'phenotype_cum_pop']

        test_selection_df = pd.DataFrame(test_selection_data, columns=columns)
        test_shadow_df = pd.DataFrame(test_shadow_data, columns=columns)

        # Test with role_growth_rate and percentile=0.5
        adaptive_entities, proportions = identify_adaptive_entities(
            test_selection_df, test_shadow_df, component='role', metric='role_growth_rate', percentile=0.5
        )

        # Equal case (world_id=3): Neither role should be identified since they match shadow
        self.assertEqual(len(adaptive_entities['selection'][3]), 0)

        # Selection > Shadow case (world_id=4): RG1 should be identified
        self.assertIn('RG1', adaptive_entities['selection'][4].index)
        self.assertNotIn('RG2', adaptive_entities['selection'][4].index)

        # Shadow > Selection case (world_id=5): Shadow has much higher values
        # At 0.5 percentile, the threshold from shadow would be around 0.7
        # Since all values in selection are 0.1, none should exceed this threshold
        self.assertEqual(len(adaptive_entities['selection'][5]), 0)

    def test_invalid_inputs(self):
        """Test behavior with invalid component or metric values."""
        # Test with invalid component
        with self.assertRaises(Exception):
            identify_adaptive_entities(
                self.selection_df, self.shadow_df, component='invalid', metric='role_growth_rate', percentile=0.5
            )

        # Test with invalid metric
        with self.assertRaises(Exception):
            identify_adaptive_entities(
                self.selection_df, self.shadow_df, component='role', metric='invalid_metric', percentile=0.5
            )

        # Test with invalid percentile
        with self.assertRaises(Exception):
            identify_adaptive_entities(
                self.selection_df, self.shadow_df, component='role', metric='role_growth_rate', percentile=1.5
            )


class TestCalculateEvolutionaryActivityStats(unittest.TestCase):

    def setUp(self):
        """Set up test data with various evolutionary metrics."""
        # Create a DataFrame with sample evolutionary activity data
        # This includes multiple worlds, steps, roles, and phenotypes
        data = [
            # world_id, step_num, phenotype, role, pop
            # World 1, Step 1 - even distribution
            (1, 1, "P1", "R1", 10),
            (1, 1, "P2", "R1", 10),
            (1, 1, "P3", "R2", 10),

            # World 1, Step 2 - uneven distribution
            (1, 2, "P1", "R1", 15),  # P1 grew by 5
            (1, 2, "P2", "R1", 5),   # P2 decreased by 5
            (1, 2, "P3", "R2", 20),  # P3 grew by 10

            # World 2, Step 1 - single role, multiple phenotypes
            (2, 1, "P4", "R3", 20),
            (2, 1, "P5", "R3", 10),

            # World 2, Step 2 - increased diversity
            (2, 2, "P4", "R3", 15),
            (2, 2, "P5", "R3", 15),
            (2, 2, "P6", "R4", 10)
        ]

        # Convert to DataFrame with evolutionary metrics
        self.df = pd.DataFrame(data, columns=["world_id", "step_num", "phenotype", "role", "pop"])

        # Add evolutionary activity metrics
        # For simplicity, we'll just add placeholder values that we can test against
        self.df["phenotype_activity"] = [1, 1, 1, 2, 2, 2, 1, 1, 2, 2, 1]
        self.df["phenotype_cum_pop"] = [10, 10, 10, 25, 15, 30, 20, 10, 35, 25, 10]
        self.df["phenotype_growth_rate"] = [0, 0, 0, 0.5, -0.5, 1.0, 0, 0, -0.25, 0.5, 0]
        self.df["phenotype_delta_N"] = [0, 0, 0, 0.3, 0, 0.8, 0, 0, 0, 0.2, 0.5]

        self.df["role_activity"] = [1, 1, 1, 2, 2, 2, 1, 1, 2, 2, 1]
        self.df["role_cum_pop"] = [20, 20, 10, 40, 40, 20, 30, 30, 30, 30, 10]
        self.df["role_growth_rate"] = [0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0]
        self.df["role_delta_N"] = [0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0.8]

    def test_output_columns_exist(self):
        """Test that all expected output columns exist in the result."""
        result = calculate_evolutionary_activity_stats(self.df)

        expected_columns = [
            'world_id', 'step_num',
            # Basic evolutionary activity metrics
            'phenotype_acum', 'phenotype_acum_mean', 'phenotype_pcum', 'phenotype_pcum_mean',
            'role_acum', 'role_acum_mean', 'role_pcum', 'role_pcum_mean',
            # Diversity metrics
            'phenotype_diversity', 'role_diversity', 'phenotype_entropy', 'role_entropy',
            # Selection metrics
            'phenotype_delta_N', 'role_delta_N'
        ]

        for column in expected_columns:
            self.assertIn(column, result.columns, f"Column {column} missing from result")

    def test_world_and_step_grouping(self):
        """Test that data is properly grouped by world_id and step_num."""
        result = calculate_evolutionary_activity_stats(self.df)

        # Check that the result has the correct number of rows
        expected_rows = 4  # 2 worlds × 2 steps
        self.assertEqual(len(result), expected_rows, f"Expected {expected_rows} rows, got {len(result)}")

        # Check that each world_id and step_num combination is present
        expected_combinations = [(1, 1), (1, 2), (2, 1), (2, 2)]
        actual_combinations = list(zip(result['world_id'], result['step_num']))

        for combo in expected_combinations:
            self.assertIn(combo, actual_combinations, f"Missing world_id/step_num combination: {combo}")
    
    def test_phenotype_metrics_calculation(self):
        """Test the calculation of phenotype-level metrics."""
        result = calculate_evolutionary_activity_stats(self.df)

        # Test for world 1, step 1
        world1_step1 = result[(result['world_id'] == 1) & (result['step_num'] == 1)]

        # Check phenotype accumulation (sum of activity values)
        expected_phenotype_acum = 3  # Three phenotypes, each with activity = 1
        self.assertEqual(world1_step1['phenotype_acum'].iloc[0], expected_phenotype_acum)

        # Check mean phenotype accumulation
        expected_phenotype_acum_mean = 1.0  # 3 total / 3 phenotypes
        self.assertEqual(world1_step1['phenotype_acum_mean'].iloc[0], expected_phenotype_acum_mean)

        # Check phenotype cumulative population
        expected_phenotype_pcum = 30  # 10 + 10 + 10
        self.assertEqual(world1_step1['phenotype_pcum'].iloc[0], expected_phenotype_pcum)

        # Check phenotype diversity
        expected_phenotype_diversity = 3  # Three unique phenotypes
        self.assertEqual(world1_step1['phenotype_diversity'].iloc[0], expected_phenotype_diversity)

        # Check phenotype delta_N
        expected_phenotype_delta_N = 0  # All delta_N values are 0 for step 1
        self.assertEqual(world1_step1['phenotype_delta_N'].iloc[0], expected_phenotype_delta_N)

        # Test for world 2, step 2 (more complex case)
        world2_step2 = result[(result['world_id'] == 2) & (result['step_num'] == 2)]

        # Check phenotype diversity
        expected_phenotype_diversity = 3  # P4, P5, P6
        self.assertEqual(world2_step2['phenotype_diversity'].iloc[0], expected_phenotype_diversity)

        # Check phenotype delta_N (sum of all delta_N values)
        expected_phenotype_delta_N = 0.7  # 0 + 0.2 + 0.5
        self.assertAlmostEqual(world2_step2['phenotype_delta_N'].iloc[0], expected_phenotype_delta_N, places=5)

    def test_role_metrics_calculation(self):
        """Test the calculation of role-level metrics."""
        result = calculate_evolutionary_activity_stats(self.df)

        # Test for world 1, step 2
        world1_step2 = result[(result['world_id'] == 1) & (result['step_num'] == 2)]

        # Check role accumulation
        expected_role_acum = 4  # R1: activity=2 (twice) + R2: activity=2
        self.assertEqual(world1_step2['role_acum'].iloc[0], expected_role_acum)

        # Check role diversity
        expected_role_diversity = 2  # Two roles: R1, R2
        self.assertEqual(world1_step2['role_diversity'].iloc[0], expected_role_diversity)

        # Check role delta_N
        expected_role_delta_N = 0.5  # Sum of role_delta_N values
        self.assertEqual(world1_step2['role_delta_N'].iloc[0], expected_role_delta_N)

        # Test for world 2, step 2
        world2_step2 = result[(result['world_id'] == 2) & (result['step_num'] == 2)]

        # Check role diversity
        expected_role_diversity = 2  # Two roles: R3, R4
        self.assertEqual(world2_step2['role_diversity'].iloc[0], expected_role_diversity)

        # Check role cumulative population
        expected_role_pcum = 40  # R3: 30 + R4: 10
        self.assertEqual(world2_step2['role_pcum'].iloc[0], expected_role_pcum)
    
    def test_entropy_calculation(self):
        """Test the calculation of Shannon entropy for phenotypes and roles."""
        result = calculate_evolutionary_activity_stats(self.df)

        # Test for world 1, step 1 (even distribution)
        world1_step1 = result[(result['world_id'] == 1) & (result['step_num'] == 1)]

        # Calculate expected entropy for even distribution
        # All phenotypes have equal population (10), so proportions are all 1/3
        # Shannon entropy = -sum(p_i * log2(p_i)) = -3 * (1/3 * log2(1/3))
        expected_phenotype_entropy = shannon_entropy([1/3, 1/3, 1/3])
        self.assertAlmostEqual(world1_step1['phenotype_entropy'].iloc[0], expected_phenotype_entropy, places=5)

        # Role entropy: R1 has 20/30 population, R2 has 10/30
        expected_role_entropy = shannon_entropy([2/3, 1/3])
        self.assertAlmostEqual(world1_step1['role_entropy'].iloc[0], expected_role_entropy, places=5)

        # Test for world 1, step 2 (uneven distribution)
        world1_step2 = result[(result['world_id'] == 1) & (result['step_num'] == 2)]

        # Phenotype entropy: P1 has 15/40, P2 has 5/40, P3 has 20/40
        expected_phenotype_entropy = shannon_entropy([15/40, 5/40, 20/40])
        self.assertAlmostEqual(world1_step2['phenotype_entropy'].iloc[0], expected_phenotype_entropy, places=5)

        # Role entropy: R1 has 20/40 population, R2 has 20/40
        expected_role_entropy = shannon_entropy([20/40, 20/40])
        self.assertAlmostEqual(world1_step2['role_entropy'].iloc[0], expected_role_entropy, places=5)


if __name__ == '__main__':
    unittest.main()