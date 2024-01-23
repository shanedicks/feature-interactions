import unittest
from unittest.mock import Mock, PropertyMock, patch
from model.features import name_from_number, Feature, Interaction, Role 


class TestNameFromNumber(unittest.TestCase):

    def test_name_from_number(self):
        self.assertEqual(name_from_number(1), 'a')
        self.assertEqual(name_from_number(27), 'aa')
        self.assertEqual(name_from_number(1, lower=False), 'A')
        self.assertEqual(name_from_number(27, lower=False), 'AA')


class TestFeature(unittest.TestCase):

    def setUp(self):
        # Mock the model
        self.mock_model = Mock()
        self.mock_model.network_id = 1
        self.mock_model.db_rows = {'features': [], 'payoffs': [], 'traits': [], 'trait_changes': []}
        self.mock_model.feature_interactions = Mock()
        self.mock_model.spacetime_dict = {"world": 1}
        self.mock_model.sites = [(0, 0), (0, 1), (1, 0), (1, 1)]  # Mock for site coordinates
        self.mock_model.trait_timeout = 10  # Mock for trait timeout

        # Create separate iterators for features and traits db_id generation
        self.feature_db_id_iter = iter(range(1, 10))
        self.trait_db_id_iter = iter(range(1, 100))

        # Mock next_db_id to return the next value from the appropriate iterator
        def next_db_id_mock(table_name):
            if table_name == 'features':
                return next(self.feature_db_id_iter)
            elif table_name == 'traits':
                return next(self.trait_db_id_iter)
            else:
                raise ValueError("Invalid table name")

        # Assign the side_effect to the mock next_db_id method
        self.mock_model.next_db_id.side_effect = next_db_id_mock
        
        # Mock the grid.coord_iter to return values in the form of ([[], (x, y)], ...)
        coord_iter_values = [([], 0, 0), ([], 0, 1), ([], 1, 0), ([], 1, 1)]
        self.mock_model.grid.coord_iter = Mock(return_value=iter(coord_iter_values))
        
        # Mock the database interactions
        self.mock_model.db.get_next_trait = Mock(return_value=None)  # No traits restored
        self.mock_model.db.get_interaction_payoffs = Mock(return_value=[])  # No payoffs restored
        
        # Mock the next_feature method
        self.mock_model.next_feature = Mock(return_value=Mock(name='NextFeature', values=['a', 'b', 'c']))

        # Mock the feature_interactions.nodes() to return an iterable
        self.mock_model.feature_interactions.nodes.return_value = []

        # Feature initialization parameters
        self.feature_id = 1
        self.env = True
        self.name = 'A'
        self.num_values = 5
        self.db_id = 1

        # Create a Feature instance
        self.feature = Feature(self.mock_model, self.feature_id, self.env, self.name, self.num_values, self.db_id)

        # Mock interactions
        self.mock_interaction1 = Mock(spec=Interaction)
        self.mock_interaction2 = Mock(spec=Interaction)

        # Setup necessary attributes for interactions
        self.mock_interaction1.db_id = 1
        self.mock_interaction1.initiator = self.feature
        self.mock_interaction1.target = Mock(values=['a', 'b'], trait_ids={'a': 11, 'b': 12})
        self.mock_interaction1.new_payoff = Mock(return_value=(0.5, 0.5))

        self.mock_interaction2.db_id = 2
        self.mock_interaction2.initiator = Mock(values=['a', 'b'], trait_ids={'a': 14, 'b': 15})
        self.mock_interaction2.target = self.feature
        self.mock_interaction2.new_payoff = Mock(return_value=(0.4, 0.4))

        # Mock methods in_edges and out_edges
        self.feature.in_edges = Mock(return_value=[self.mock_interaction2])
        self.feature.out_edges = Mock(return_value=[self.mock_interaction1])

    def test_init(self):
        feature = self.feature
        
        # Assert initialization
        self.assertEqual(feature.feature_id, self.feature_id)
        self.assertEqual(feature.current_value_id, self.num_values)
        self.assertEqual(feature.model, self.mock_model)
        self.assertEqual(feature.env, self.env)
        self.assertEqual(feature.name, self.name)
        self.assertEqual(feature.db_id, self.db_id)
        self.assertEqual(len(feature.model.db_rows['features']), 0)  # No new row added because db_id is provided
        self.assertEqual(feature.empty_steps, 0)
        self.assertIsNotNone(feature.empty_traits)
        self.assertIsNotNone(feature.trait_ids)
        self.assertIsNotNone(feature.values)
        self.assertEqual(feature.values, ['a', 'b', 'c', 'd', 'e'])
        self.assertEqual(feature.trait_ids, {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5})
        self.assertEqual(len(feature.traits_dict), 4)  # Number of coordinates in the grid

        # Test without provided name and db_id
        feature_without_name_db_id = Feature(self.mock_model, self.feature_id, self.env, num_values=self.num_values)
        self.assertEqual(feature_without_name_db_id.name, name_from_number(self.feature_id, lower=False))
        self.assertEqual(feature_without_name_db_id.db_id, 1)
        self.assertEqual(len(feature_without_name_db_id.model.db_rows['features']), 1)  # New row added because db_id is None


    def test_next_trait(self):
        # Mock set_payoffs
        self.feature.set_payoffs = Mock()

        # Test case: Trait is restored from the database
        restored_trait_data = {'trait_id': 10}
        self.mock_model.db.get_next_trait.return_value = restored_trait_data
        value = self.feature.next_trait()

        # Assertions for restored trait
        self.assertIn(value, self.feature.values)
        self.assertIn(value, self.feature.trait_ids)
        self.assertEqual(self.feature.trait_ids[value], restored_trait_data['trait_id'])
        self.assertNotIn((self.feature.trait_ids[value], value, self.feature.db_id), self.mock_model.db_rows['traits'])
        self.assertIn((self.mock_model.spacetime_dict['world'], self.feature.trait_ids[value], 'added'), self.mock_model.db_rows['trait_changes'])
        self.feature.set_payoffs.assert_not_called()  # Assuming feature not in feature_interactions.nodes

        # Test case: New trait is added
        self.mock_model.db.get_next_trait.return_value = None
        value = self.feature.next_trait()

        # Assertions for new trait added
        self.assertIn(value, self.feature.values)
        self.assertIn(value, self.feature.trait_ids)
        self.assertIn((self.feature.trait_ids[value], value, self.feature.db_id), self.mock_model.db_rows['traits'])
        self.assertIn((self.mock_model.spacetime_dict['world'], self.feature.trait_ids[value], 'added'), self.mock_model.db_rows['trait_changes'])
        self.feature.set_payoffs.assert_not_called()  # Assuming feature not in feature_interactions.nodes

        # Test case: Feature is in feature_interactions.nodes
        self.mock_model.feature_interactions.nodes.return_value = [self.feature]
        value = self.feature.next_trait()
        self.feature.set_payoffs.assert_called_once_with(value)  # set_payoffs called

    def test_set_payoffs(self):
        self.mock_interaction1.payoffs = {}
        self.mock_interaction2.payoffs = {'a': {}, 'b': {}}
        # Use 'g' as the new value for which payoffs are being set
        value = self.feature.next_trait()

        # Mock return values for get_interaction_payoffs for initiator and target interactions
        self.mock_model.db.get_interaction_payoffs.side_effect = lambda model, db_id, initiator_values, target_values: [
            payoff for payoff in (
                {'initiator': value, 'target': 'a', 'i_utils': 1.0, 't_utils': 2.0} if db_id == 1 else [],
                {'initiator': 'b', 'target': value, 'i_utils': 3.0, 't_utils': 4.0} if db_id == 2 else []
            ) if payoff
        ]

        # Call set_payoffs
        self.feature.set_payoffs(value)

        # Verify that payoffs are updated correctly
        self.assertIn(value, self.mock_interaction1.payoffs)
        self.assertIn('a', self.mock_interaction1.payoffs[value])
        self.assertIn('b', self.mock_interaction1.payoffs[value])
        self.assertEqual(self.mock_interaction1.payoffs[value]['a'], (1.0, 2.0))
        self.assertIn('a', self.mock_interaction2.payoffs)
        self.assertIn('b', self.mock_interaction2.payoffs)
        self.assertIn(value, self.mock_interaction2.payoffs['b'])
        self.assertIn(value, self.mock_interaction2.payoffs['a'])
        self.assertEqual(self.mock_interaction2.payoffs['b'][value], (3.0, 4.0))

        # Verify that new payoffs are calculated and added to the database
        self.mock_interaction1.new_payoff.assert_called_once_with()
        self.mock_interaction2.new_payoff.assert_called_once_with()
        self.assertIn((1, 6, 12, 0.5, 0.5), self.mock_model.db_rows['payoffs']) # feature.f,  mock1.b
        self.assertIn((2, 14, 6, 0.4, 0.4), self.mock_model.db_rows['payoffs']) # mock2.a, feature.f

        # Ensure the 'trait_changes' database is updated
        expected_trait_changes = (self.mock_model.spacetime_dict["world"], 6, "added")
        self.assertIn(expected_trait_changes, self.mock_model.db_rows['trait_changes'])

    def test_remove_trait(self):
        # Choose a trait value to remove
        value = 'a'

        # Mock the necessary structures
        self.feature.trait_ids = {'a': 11, 'b': 12}
        self.feature.values = ['a', 'b']
        self.feature.empty_traits = {'a': 0, 'b': 0}
        self.feature.traits_dict = {(0, 0): {'a': 1, 'b': 1}, (0, 1): {}, (1, 0): {}, (1, 1): {}}
        self.mock_interaction1.payoffs = {'a': {}, 'b': {}}
        self.mock_interaction2.payoffs = {'a': {'a': [], 'b': []}, 'b': {'a': [], 'b': []}}

        # Mock remove_feature method for when all traits are removed
        self.mock_model.remove_feature = Mock()

        # Call remove_trait
        self.feature.remove_trait(value)

        # Verify the trait is removed from internal feature structures
        self.assertNotIn(value, self.feature.values)
        self.assertNotIn(value, self.feature.trait_ids)
        self.assertNotIn(value, self.feature.empty_traits)

        # Verify the trait is removed from traits_dict for all sites
        for s in self.mock_model.sites:
            self.assertNotIn(value, self.feature.traits_dict[s])

        # Verify the trait removal is recorded in the trait_changes database
        expected_trait_changes = (self.mock_model.spacetime_dict["world"], 11, "removed")
        self.assertIn(expected_trait_changes, self.mock_model.db_rows['trait_changes'])

        # Verify payoffs are removed for targeted interactions
        self.assertNotIn(value, self.mock_interaction2.payoffs['a'])

        # Verify payoffs are removed for initiated interactions
        self.assertNotIn(value, self.mock_interaction1.payoffs)

        # Verify that remove_feature is called if no traits are left
        self.feature.values = ['b']  # Simulate all traits being removed
        self.feature.remove_trait('b')  # Call remove_trait again to trigger feature removal
        self.mock_model.remove_feature.assert_called_once_with(self.feature)

    def test_check_empty(self):
        # Setup the traits_dict to simulate traits presence in sites
        self.feature.traits_dict = {
            (0, 0): {'a': 1},
            (0, 1): {'b': 0},
            (1, 0): {},
            (1, 1): {'a': 0, 'b': 0}
        }
        self.feature.values = ['a', 'b']

        # Initial state
        self.feature.empty_traits = {'a': 0, 'b': 0}
        self.feature.empty_steps = 0

        # Call check_empty when traits are present in some sites
        self.feature.check_empty()

        # Verify that empty_steps is reset to 0
        self.assertEqual(self.feature.empty_steps, 0)

        # Verify that empty_traits is updated correctly
        self.assertEqual(self.feature.empty_traits['a'], 0)
        self.assertEqual(self.feature.empty_traits['b'], 1)

        # Modify traits_dict to simulate traits absence in all sites
        self.feature.traits_dict = {
            (0, 0): {'a': 0},
            (0, 1): {'b': 0},
            (1, 0): {},
            (1, 1): {'a': 0, 'b': 0}
        }

        # Call check_empty when traits are absent in all sites
        self.feature.check_empty()

        # Verify that empty_steps is incremented
        self.assertEqual(self.feature.empty_steps, 1)

        # Verify that empty_traits is updated correctly
        self.assertEqual(self.feature.empty_traits['a'], 1)
        self.assertEqual(self.feature.empty_traits['b'], 2)

    def test_prune_traits(self):
        # Setup the empty_traits to simulate trait emptiness duration
        self.feature.empty_traits = {'a': 9, 'b': 10, 'c': 11}

        # Mock the remove_trait method
        self.feature.remove_trait = Mock()

        # Call prune_traits
        self.feature.prune_traits()

        # Verify that remove_trait is called for traits with emptiness duration equal to or exceeding trait_timeout
        self.feature.remove_trait.assert_any_call('b')
        self.feature.remove_trait.assert_any_call('c')
        self.assertEqual(self.feature.remove_trait.call_count, 2)

        # Verify that 'a' is not pruned as its emptiness duration is less than trait_timeout
        calls = [call[0][0] for call in self.feature.remove_trait.call_args_list]
        self.assertNotIn('a', calls)


class TestInteraction(unittest.TestCase):

    def setUp(self):
        # Mock the model
        self.mock_model = Mock()
        self.mock_model.network_id = 1
        self.mock_model.db_rows = {'features': [], 'payoffs': [], 'traits': [], 'trait_changes': [], 'interactions': []}
        self.mock_model.trait_payoff_mod = 0.5
        self.mock_model.payoff_bias = 0.5
        self.mock_model.anchor_bias = 0.5
        self.mock_model.spacetime_dict = {"world": 1}
        self.mock_model.random = Mock()
        self.mock_model.random.triangular = Mock(return_value=0.5)
        self.mock_model.db.get_interaction_payoffs = Mock(return_value=[])

        # Create separate iterators for interaction db_id generation
        self.interaction_db_id_iter = iter(range(1, 10))

        # Mock next_db_id to return the next value from the iterator
        def next_db_id_mock(table_name):
            return next(self.interaction_db_id_iter)

        # Assign the side_effect to the mock next_db_id method
        self.mock_model.next_db_id.side_effect = next_db_id_mock

        # Mock Feature instances as initiator and target
        self.mock_initiator = Mock(spec=Feature)
        self.mock_target = Mock(spec=Feature)
        self.mock_initiator.name = 'Initiator'
        self.mock_target.name = 'Target'
        self.mock_initiator.db_id = 1
        self.mock_target.db_id = 2
        self.mock_initiator.values = ['a', 'b']
        self.mock_target.values = ['x', 'y']
        self.mock_initiator.trait_ids = {'a': 1, 'b': 2}
        self.mock_target.trait_ids = {'x': 3, 'y': 4}

        # Interaction initialization parameters
        self.db_id = 1
        self.payoffs = None
        self.anchors = None
        self.restored = False

        # Create an Interaction instance
        self.interaction = Interaction(
            model=self.mock_model,
            initiator=self.mock_initiator,
            target=self.mock_target,
            db_id=self.db_id,
            payoffs=self.payoffs,
            anchors=self.anchors,
            restored=self.restored
        )

    def test_init(self):
        # Patching these methods out to test __init__
        with patch.object(Interaction, 'construct_payoffs', return_value={}) as mock_construct_payoffs, \
             patch.object(Interaction, 'restore_payoffs', return_value={}) as mock_restore_payoffs:

            # Test case with provided db_id, no payoffs provided, not restored
            interaction = Interaction(self.mock_model, self.mock_initiator, self.mock_target, self.db_id)
            self.assertEqual(interaction.db_id, self.db_id)
            self.assertFalse(interaction.payoffs is None)
            interaction.construct_payoffs.assert_called_once()
            interaction.restore_payoffs.assert_not_called()
            mock_construct_payoffs.reset_mock()
            mock_restore_payoffs.reset_mock()

            # Test case with provided db_id and payoffs
            self.payoffs = {"a": {'b': []}}
            interaction = Interaction(self.mock_model, self.mock_initiator, self.mock_target, self.db_id, False, self.payoffs)
            self.assertEqual(interaction.db_id, self.db_id)
            self.assertEqual(interaction.payoffs, self.payoffs)
            interaction.construct_payoffs.assert_not_called()
            interaction.restore_payoffs.assert_not_called()
            mock_construct_payoffs.reset_mock()
            mock_restore_payoffs.reset_mock()

            # Test case with provided db_id, restored is True
            interaction = Interaction(self.mock_model, self.mock_initiator, self.mock_target, self.db_id, True)
            self.assertEqual(interaction.db_id, self.db_id)
            self.assertFalse(interaction.payoffs is None)
            interaction.construct_payoffs.assert_not_called()
            interaction.restore_payoffs.assert_called_once()
            mock_construct_payoffs.reset_mock()
            mock_restore_payoffs.reset_mock()

            # Test case without provided db_id
            interaction = Interaction(self.mock_model, self.mock_initiator, self.mock_target)
            self.assertIsNotNone(interaction.db_id)
            self.assertFalse(interaction.payoffs is None)
            interaction.construct_payoffs.assert_called_once()
            interaction.restore_payoffs.assert_not_called()

            # Ensure new row is added to db_rows['interactions'] with correct db_id
            added_row = self.mock_model.db_rows['interactions'][-1]  # Get the last added row
            self.assertEqual(added_row[0], interaction.db_id)  # Ensure the db_id in the added row matches the interaction's db_id
            expected_row = (interaction.db_id, self.mock_model.network_id, self.mock_initiator.db_id, self.mock_target.db_id, interaction.anchors['i'], interaction.anchors['t'])
            self.assertEqual(added_row, expected_row)

    def test_construct_payoffs(self):
        # Setup
        self.interaction.initiator.values = ['a', 'b']
        self.interaction.initiator.trait_ids = {'a': 1, 'b': 2}
        self.interaction.target.values = ['a', 'b']
        self.interaction.target.trait_ids = {'a': 3, 'b': 4}
        self.interaction.new_payoff = Mock(return_value=(0.5, -0.5))

        # Call construct_payoffs
        payoffs = self.interaction.construct_payoffs()

        # Assertions
        self.assertEqual(len(payoffs), len(self.interaction.initiator.values))  # Payoff dict should have a key for each initiator value
        for i_value in self.interaction.initiator.values:
            self.assertIn(i_value, payoffs)
            for t_value in self.interaction.target.values:
                self.assertIn(t_value, payoffs[i_value])
                self.assertEqual(payoffs[i_value][t_value], (0.5, -0.5))  # Assert the payoff values are as returned by new_payoff

        # Verify new_payoff was called the correct number of times and payoffs added to db
        self.assertEqual(self.interaction.new_payoff.call_count, len(self.interaction.initiator.values) * len(self.interaction.target.values))
        for i_value in self.interaction.initiator.values:
            for t_value in self.interaction.target.values:
                row = (self.interaction.db_id, int(self.interaction.initiator.trait_ids[i_value]), int(self.interaction.target.trait_ids[t_value]), 0.5, -0.5)
                self.assertIn(row, self.mock_model.db_rows['payoffs'])

    def test_restore_payoffs(self):
        # Setup
        self.interaction.initiator.values = ['a', 'b']
        self.interaction.initiator.trait_ids = {'a': 1, 'b': 2}
        self.interaction.target.values = ['a', 'b']
        self.interaction.target.trait_ids = {'a': 3, 'b': 4}
        self.interaction.new_payoff = Mock(return_value=(0.6, -0.6))

        # Mock get_interaction_payoffs to return restored payoffs for some pairs
        self.mock_model.db.get_interaction_payoffs.return_value = [
            {'initiator': 'a', 'target': 'a', 'i_utils': 0.1, 't_utils': -0.1},
            {'initiator': 'b', 'target': 'b', 'i_utils': 0.2, 't_utils': -0.2}
        ]

        # Call restore_payoffs
        payoffs = self.interaction.restore_payoffs()

        # Assertions
        self.assertEqual(len(payoffs), len(self.interaction.initiator.values))  # Payoff dict should have a key for each initiator value
        self.assertEqual(payoffs['a']['a'], (0.1, -0.1))  # Assert restored payoffs are used
        self.assertEqual(payoffs['b']['b'], (0.2, -0.2))  # Assert restored payoffs are used
        self.assertEqual(payoffs['a']['b'], (0.6, -0.6))  # Assert new payoffs are generated where needed
        self.assertEqual(payoffs['b']['a'], (0.6, -0.6))  # Assert new payoffs are generated where needed

        # Verify new_payoff was called for missing pairs and payoffs added to db
        self.assertEqual(self.interaction.new_payoff.call_count, 2)  # Only for missing pairs ('a', 'b') and ('b', 'a')
        new_payoffs_rows = [
            (self.interaction.db_id, 1, 4, 0.6, -0.6),
            (self.interaction.db_id, 2, 3, 0.6, -0.6)
        ]
        for row in new_payoffs_rows:
            self.assertIn(row, self.mock_model.db_rows['payoffs'])

        # Ensure the 'payoffs' database is updated with the correct new payoffs
        for i_value, t_value in [('a', 'b'), ('b', 'a')]:
            row = (self.interaction.db_id, int(self.interaction.initiator.trait_ids[i_value]), int(self.interaction.target.trait_ids[t_value]), 0.6, -0.6)
            self.assertIn(row, self.mock_model.db_rows['payoffs'])

class TestRole(unittest.TestCase):

    def setUp(self):
        # Mock the model
        self.mock_model = Mock()
        self.mock_model.roles_dict = {}
        self.mock_model.grid.coord_iter = Mock(
            return_value=iter([([], 0, 0), ([], 0, 1), ([], 1, 0), ([], 1, 1)])
        )
        self.mock_model.feature_interactions = Mock()

        # Mock features
        self.mock_feature1 = Mock(spec=Feature, env=False)
        name_F = PropertyMock(return_value='F')
        type(self.mock_feature1).name = name_F
        self.mock_feature2 = Mock(spec=Feature, env=False)
        name_G = PropertyMock(return_value='G')
        type(self.mock_feature2).name = name_G
        self.features = frozenset([self.mock_feature1, self.mock_feature2])

        # Mock interactions
        self.mock_interaction1 = Mock(spec=Interaction, initiator=self.mock_feature1, target=self.mock_feature2)
        self.mock_interaction2 = Mock(spec=Interaction, initiator=self.mock_feature2, target=self.mock_feature1)
        self.mock_model.feature_interactions.edges.return_value = [
            (self.mock_feature1, self.mock_feature2, self.mock_interaction1),
            (self.mock_feature2, self.mock_feature1, self.mock_interaction2)
        ]
        self.mock_model.feature_interactions.in_edges.return_value = [
            (self.mock_feature1, self.mock_feature2, self.mock_interaction2),
            (self.mock_feature2, self.mock_feature1, self.mock_interaction1)
        ]

        # Initialize Role instance
        self.role = Role(self.mock_model, self.features)

    @patch('model.features.Role.get_interactions', return_value={'initiator': [], 'target': []})
    @patch('model.features.Role.get_target_features', return_value=[])
    @patch('model.features.Role.get_neighbors', return_value={'initiators': [], 'targets': []})
    def test_init(self, mock_get_interactions, mock_get_target_features, mock_get_neighbors):
        # Initialize a new role to trigger the __init__ method
        new_role = Role(self.mock_model, self.features)

        self.assertEqual(new_role.model, self.mock_model)
        self.assertEqual(new_role.features, self.features)
        self.assertEqual(new_role.rolename, ":".join(sorted([f.name for f in self.features])))

        # Validate that the functions are called during initialization
        mock_get_interactions.assert_called_once()
        mock_get_target_features.assert_called_once()
        mock_get_neighbors.assert_called_once()

        # Assert types dictionary is initialized based on grid coordinates
        self.assertEqual(
            len(new_role.types),
            len(list(self.mock_model.grid.coord_iter()))
        )

    def test_get_target_features(self):
        # Mock interactions with target features
        mock_interaction1 = Mock()
        mock_interaction2 = Mock()
        mock_interaction3 = Mock()

        # Setup target features for interactions
        target_feature1 = Mock(env=False)
        target_feature2 = Mock(env=True)  # Environmental feature, should be excluded
        target_feature3 = Mock(env=False)

        mock_interaction1.target = target_feature1
        mock_interaction2.target = target_feature2
        mock_interaction3.target = target_feature3

        # Set interactions as initiating interactions of the role
        self.role.interactions = {'initiator': [mock_interaction1, mock_interaction2, mock_interaction3]}

        # Call get_target_features
        target_features = self.role.get_target_features()

        # Verify that only non-environmental features are returned
        self.assertIn(target_feature1, target_features)
        self.assertNotIn(target_feature2, target_features)
        self.assertIn(target_feature3, target_features)
        self.assertEqual(len(target_features), 2)  # Only two non-environmental features

    def test_get_interactions(self):
        # Mock feature interactions in the model
        mock_interaction1 = Mock(spec=Interaction)
        mock_interaction2 = Mock(spec=Interaction)
        mock_interaction3 = Mock(spec=Interaction)
        mock_interaction4 = Mock(spec=Interaction)
        
        # Set up edges and in_edges to return mock interactions
        self.mock_model.feature_interactions.edges.return_value = [(None, None, mock_interaction1), (None, None, mock_interaction2)]
        self.mock_model.feature_interactions.in_edges.return_value = [(None, None, mock_interaction3), (None, None, mock_interaction4)]
        
        # Call get_interactions
        interactions = self.role.get_interactions()

        # Verify that interactions are correctly categorized as 'initiator' and 'target'
        self.assertIn(mock_interaction1, interactions['initiator'])
        self.assertIn(mock_interaction2, interactions['initiator'])
        self.assertIn(mock_interaction3, interactions['target'])
        self.assertIn(mock_interaction4, interactions['target'])

    def test_get_neighbors(self):
        # Mock features
        mock_feature_A = Mock(name='A')
        mock_feature_B = Mock(name='B')
        mock_feature_C = Mock(name='C')

        # Mock roles in the model
        mock_roleA = Mock(spec=Role, features=frozenset([mock_feature_A]), rolename='RoleA')
        mock_roleB = Mock(spec=Role, features=frozenset([mock_feature_B]), rolename='RoleB')
        mock_roleC = Mock(spec=Role, features=frozenset([mock_feature_C]), rolename='RoleC')
        self.mock_model.roles_dict = {
            'RoleA': mock_roleA,
            'RoleB': mock_roleB,
            'RoleC': mock_roleC
        }

        # Set interactions of the role
        mock_interaction1 = Mock(spec=Interaction, initiator=mock_feature_A, target=self.mock_feature1)
        mock_interaction2 = Mock(spec=Interaction, initiator=self.mock_feature1, target=mock_feature_B)
        self.role.interactions = {
            'initiator': [mock_interaction2],
            'target': [mock_interaction1]
        }

        # Call get_neighbors
        neighbors = self.role.get_neighbors()

        # Verify that neighbors are correctly identified
        self.assertIn(mock_roleA, neighbors['initiators'])
        self.assertIn(mock_roleB, neighbors['targets'])
        self.assertNotIn(mock_roleC, neighbors['initiators'])
        self.assertNotIn(mock_roleC, neighbors['targets'])


if __name__ == '__main__':
    unittest.main()