import random
import unittest
from collections import defaultdict
from unittest.mock import Mock, MagicMock, patch, call
from model.agents import Agent, Site
from model.features import Interaction, Role

class TestSite(unittest.TestCase):
    
    def setUp(self):
        self.mock_model = Mock()
        self.mock_model.random = random.Random()
        self.mock_pos = (0, 0)
        mock_feature_A = MagicMock(name='A', values=['a'])
        mock_feature_B = MagicMock(name='B', values=['a'])
        mock_feature_C = MagicMock(name='C', values=['a'])
        self.mock_model.get_features_list.return_value = [mock_feature_A, mock_feature_B, mock_feature_C]
        self.mock_model.base_env_utils = 1.0
        self.mock_model.site_pop_limit = 10
        self.mock_model.pop_cost_exp = 2.0
        self.mock_model.grid = {0: {0: MagicMock(agents=[])}}
        self.site = Site(self.mock_model, self.mock_pos)

    def test_init(self):
        # Check Attributes
        self.assertEqual(self.site.pos, self.mock_pos)
        self.assertEqual(self.site.pop_cost, 0)
        self.assertEqual(self.site.born, 0)
        self.assertEqual(self.site.died, 0)
        self.assertEqual(self.site.moved_in, 0)
        self.assertEqual(self.site.moved_out, 0)

        # Check stats
        self.assertIsInstance(self.site.interaction_stats, defaultdict)

        # Check agents
        self.assertEqual(self.site.agents, [])

        # Check if keys of traits and utils are the same
        self.assertEqual(set(self.site.traits.keys()), set(self.site.utils.keys()))

        # Check if all keys in traits (and thus in utils) are in the list of environmental features
        env_features = self.mock_model.get_features_list(env=True)
        for key in self.site.traits:
            self.assertIn(key, env_features)

    def test_get_pop(self):
        self.mock_model.grid[0][0].agents.extend([Mock(), Mock()])
        self.assertEqual(self.site.get_pop(), 2)

    def test_get_pop_cost(self):
        self.assertEqual(self.site.get_pop_cost(), 0)
        self.site.get_pop = Mock(return_value=5)
        expected_cost = (5 / self.mock_model.site_pop_limit) ** self.mock_model.pop_cost_exp
        self.assertEqual(self.site.get_pop_cost(), expected_cost)

    def test_reset(self):
        self.site.born = 10
        self.site.died = 5
        self.site.moved_in = 3
        self.site.moved_out = 2
        self.site.reset()
        self.assertEqual(self.site.born, 0)
        self.assertEqual(self.site.died, 0)
        self.assertEqual(self.site.moved_in, 0)
        self.assertEqual(self.site.moved_out, 0)
        for feature in self.site.utils:
            self.assertEqual(self.site.utils[feature], self.mock_model.base_env_utils)

    def test_agent_sample(self):
        mock_agents = [Mock() for _ in range(5)]
        self.mock_model.grid[0][0].agents.extend(mock_agents)
        sample = self.site.agent_sample(3)
        self.assertTrue(all(agent in mock_agents for agent in sample))
        self.assertEqual(len(sample), 3)

    def tearDown(self):
        # Reset mock objects
        self.mock_model.reset_mock()
        self.mock_model.grid[0][0].reset_mock()

        # Clear references to mock objects and other state
        self.mock_model = None
        self.mock_pos = None
        self.site = None


class TestAgent(unittest.TestCase):

    def setUp(self):
        # Create mock model
        self.mock_model = Mock()
        self.mock_model.random = random.Random()
        self.mock_model.roles_dict = {}
        self.mock_model.feature_cost_exp = 0
        self.mock_model.cached_payoffs = {}
        self.mock_model.cached = 0
        self.mock_model.new = 0
        self.mock_model.env = 0
        self.mock_model.target_sample = 3
        self.mock_model.grid.place_agent = Mock(side_effect=lambda agent, pos: setattr(agent, 'pos', pos))
        
        # patch shuffle for consistent env interaction behavior
        self.shuffle_patch = patch.object(
                self.mock_model.random, 'shuffle', 
                lambda lst: lst.sort(key=lambda i: i.db_id)
            )
        self.shuffle_patch.start()
        
        # Create mock roles and role_dict
        self.mock_role1 = Mock(spec=Role, features=frozenset(['F', 'H']))
        self.mock_role2 = Mock(spec=Role, features=frozenset(['G']))
        self.mock_model.roles_dict[self.mock_role1.features] = self.mock_role1
        self.mock_model.roles_dict[self.mock_role2.features] = self.mock_role2
        
        # Create mock interactions
        self.mock_interaction1 = Mock(spec=Interaction, initiator='F', target='A', db_id=1)
        self.mock_interaction2 = Mock(spec=Interaction, initiator='F', target='B', db_id=2)
        self.mock_interaction3 = Mock(spec=Interaction, initiator='F', target='G', db_id=3)
        self.mock_interaction4 = Mock(spec=Interaction, initiator='F', target='I', db_id=4)
        self.mock_interaction5 = Mock(spec=Interaction, initiator='G', target='A', db_id=5)
        self.mock_interaction6 = Mock(spec=Interaction, initiator='G', target='B', db_id=6)
        self.mock_interaction7 = Mock(spec=Interaction, initiator='G', target='H', db_id=7)
        self.mock_interaction8 = Mock(spec=Interaction, initiator='G', target='I', db_id=7)

        # Define payoff matrices for the interactions
        self.mock_interaction1.payoffs = {'a': {'a': (-0.1, -0.05)}}
        self.mock_interaction2.payoffs = {'a': {'a': (0.2, 0.10)}}
        self.mock_interaction3.payoffs = {'a': {'a': (0.3, 0.15)}}
        self.mock_interaction4.payoffs = {'a': {'a': (0.4, 0.20)}}
        self.mock_interaction5.payoffs = {'a': {'a': (0.5, 0.25)}}
        self.mock_interaction6.payoffs = {'a': {'a': (0.6, 0.30)}}
        self.mock_interaction7.payoffs = {'a': {'a': (0.7, 0.35)}}
        self.mock_interaction8.payoffs = {'a': {'a': (0.8, 0.40)}}
        
        # Define interactions and neighbors for the roles
        self.mock_role1.interactions = {
            'initiator': [self.mock_interaction1, self.mock_interaction2, self.mock_interaction3, self.mock_interaction4],
            'target': [self.mock_interaction7]
        }
        self.mock_role1.neighbors = {
            'initiators': [self.mock_role2],
            'targets': [self.mock_role2]
        }
        self.mock_role2.interactions = {
            'initiator': [self.mock_interaction5, self.mock_interaction6, self.mock_interaction7, self.mock_interaction8],
            'target': [self.mock_interaction3]
        }
        self.mock_role2.neighbors = {
            'initiators': [self.mock_role1],
            'targets': [self.mock_role1]        
        }

        # Create mock agents and mock site
        self.agent1_traits = {'F': 'a', 'H': 'a'}
        self.agent2_traits = {'G': 'a'}
        self.agent1 = Agent(1, self.mock_model, self.agent1_traits, 0.0)
        self.agent2 = Agent(2, self.mock_model, self.agent2_traits, 0.0)
        self.mock_site = Mock(spec=Site)

        # Set agent and site attributes
        self.mock_site.traits = {'A': 'a', 'B': 'a'}
        self.mock_site.utils = {'A': 0.05, 'B': 0.5}
        self.mock_site.interaction_stats = defaultdict(lambda: [0, 0, 0])
        self.mock_pos = (0,0)
        self.mock_role1.types = {self.mock_pos: {}}
        self.agent1.pos = self.agent2.pos = self.mock_site.pos = self.mock_pos
        self.mock_site.agent_sample.return_value = [
            self.agent1, 
            self.agent2,
            Mock(utils=1.0, role=self.agent1.role, traits=self.agent1_traits)
        ]
        self.mock_model.sites = {}
        self.mock_model.sites[self.mock_pos] = self.mock_site
        self.agent1.site = self.mock_site
        self.agent2.site = self.mock_site

    def test_init(self):
        self.assertEqual(self.agent1.traits, self.agent1_traits)
        self.assertEqual(self.agent1.utils, 0.0)
        self.assertEqual(self.agent1.unique_id, 1)
        self.assertEqual(self.agent1.shadow, False)
        self.assertEqual(self.agent2.traits, self.agent2_traits)
        self.assertEqual(self.agent2.utils, 0.0)
        self.assertEqual(self.agent2.unique_id, 2)
        self.assertEqual(self.agent2.shadow, False)

    def test_phenotype_property(self):
        agent1_phenotype = 'F.a:H.a'
        agent2_phenotype = 'G.a'
        self.assertEqual(self.agent1.phenotype, agent1_phenotype)
        self.assertEqual(self.agent2.phenotype, agent2_phenotype)

    def test_increment_and_decrement(self):
        # Create mock feature and role with appropriate structures
        mock_feature = Mock()
        mock_feature.__str__ = Mock(side_effect=lambda: 'F')
        mock_feature.traits_dict = {self.mock_pos: {}}
        mock_role = Mock()
        mock_role.types = {self.mock_pos: {}}

        # Assign mock feature and role to agent1
        self.agent1.traits = {mock_feature: 'a'}
        self.agent1.role = mock_role

        # Test increment_phenotype
        self.agent1.increment_phenotype()
        self.assertEqual(mock_role.types[self.agent1.pos]['F.a'], 1, "Phenotype count should increment")
        self.assertEqual(mock_feature.traits_dict[self.agent1.pos]['a'], 1, "Trait count should increment")

        # Test increment_phenotype with shadow agent
        self.agent1.shadow = True
        self.agent1.increment_phenotype()
        self.assertEqual(mock_role.types[self.agent1.pos]['F.a'], 2, "Phenotype count should increment")
        self.assertEqual(mock_feature.traits_dict[self.agent1.pos]['a'], 1, "Trait count should not increment for shadow agent")

        # Test decrement_phenotype
        self.agent1.shadow = False
        self.agent1.decrement_phenotype()
        self.assertEqual(mock_role.types[self.agent1.pos]['F.a'], 1, "Phenotype count should decrement")
        self.assertEqual(mock_feature.traits_dict[self.agent1.pos]['a'], 0, "Trait count should decrement")

        # Test decrement_phenotype with shadow agent
        self.agent1.shadow = True
        self.agent1.decrement_phenotype()
        self.assertEqual(mock_role.types[self.agent1.pos]['F.a'], 0, "Phenotype count should decrement")
        self.assertEqual(mock_feature.traits_dict[self.agent1.pos]['a'], 0, "Trait count should not decrement for shadow agent")


    def test_get_site(self):
        self.agent1.increment_phenotype = Mock()
        site = self.agent1.get_site()
        self.assertEqual(site, self.mock_site, "Agent1 site should be mock_site")

    def test_get_role(self):
        self.assertEqual(self.agent1.get_role(), self.mock_role1)
        self.assertEqual(self.agent2.get_role(), self.mock_role2)

    def test_get_shadow_agent(self):
        mock_agent = Mock()
        self.mock_model.shadow.sites = {self.agent1.pos: Mock(agents=[mock_agent])}
        self.assertEqual(self.agent1.get_shadow_agent(), mock_agent)

    def test_get_agent_target(self):
        # Test that agent2 is selected as the target
        target = self.agent1.get_agent_target()
        self.assertEqual(target, self.agent2, "Agent2 should be the selected target")

        # Change conditions so agent2 is not a suitable target
        self.agent2.utils = -1  # Make agent2's utility negative
        target = self.agent1.get_agent_target()
        self.assertIsNone(target, "No suitable target should be found")

        # Test that agent1 does not target itself when own role is targetable
        self.agent1.role.neighbors['targets'].append(self.agent1.role)
        target = self.agent1.get_agent_target()
        self.assertNotEqual(target, self.agent1, "Agent1 should not select itself as a target")

    def test_do_env_interactions(self):
        # Process interactions for agent1
        self.agent1.do_env_interactions()

        # Check expected utilities for agent1 and site features A and B
        # Agent1 should do interaction 1 against A but not 2 against B
        self.assertEqual(self.agent1.utils, -0.1, "agent1 utility did not change as expected.")
        self.assertEqual(self.mock_site.utils['A'], 0, "Site Feature A utility did not change as expected after agent1 interaction")
        self.assertEqual(self.mock_site.utils['B'], 0.5, "Site Feature B utility changed unexpectedly after agent1 interaction")

        # Process interactions for agent2
        self.agent2.do_env_interactions()

        # Check expected utilities for agent2 and site features A and B
        # Agent2 should do interaction 6 against B but not 1 against A
        self.assertEqual(self.agent2.utils, 0.6, "agent2 utility did not change as expected.")
        self.assertEqual(self.mock_site.utils['A'], 0, "Site Feature A utility changed unexpectedly after agent2 interaction")
        self.assertEqual(self.mock_site.utils['B'], 0.8, "Site Feature B utility changed unexpectedly after agent2 interaction")

        # Verify that interactions 1 and 6 were recorded and that 2 and 5 were not
        self.assertEqual(self.mock_site.interaction_stats[1], [1, -.1, -.05], "Interaction 1 stats incorrect")
        self.assertEqual(self.mock_site.interaction_stats[2], [0, 0, 0], "Interaction 2 stats incorrect")
        self.assertEqual(self.mock_site.interaction_stats[5], [0, 0, 0], "Interaction 5 stats incorrect")
        self.assertEqual(self.mock_site.interaction_stats[6], [1, 0.6, 0.3], "Interaction 6 stats incorrect")

    def test_do_agent_interactions(self):
        # Test agent1 as initiator and agent2 as target
        i_payoff, t_payoff, stats = self.agent1.do_agent_interactions(self.agent1, self.agent2)
        interaction = self.mock_interaction3
        expected_i_payoff = interaction.payoffs['a']['a'][0]
        expected_t_payoff = interaction.payoffs['a']['a'][1]
        self.assertEqual(i_payoff, expected_i_payoff, "Incorrect initiator payoff for agent1 as initiator")
        self.assertEqual(t_payoff, expected_t_payoff, "Incorrect target payoff for agent2 as target")
        expected_stats = [(interaction.db_id, interaction.payoffs['a']['a'][0], interaction.payoffs['a']['a'][1])]
        self.assertEqual(stats, expected_stats, "Incorrect stats for agent1 initiating")

        # Test agent2 as initiator and agent1 as target
        i_payoff, t_payoff, stats = self.agent1.do_agent_interactions(self.agent2, self.agent1)
        interaction = self.mock_interaction7
        expected_i_payoff = interaction.payoffs['a']['a'][0]
        expected_t_payoff = interaction.payoffs['a']['a'][1]
        self.assertEqual(i_payoff, expected_i_payoff, "Incorrect target payoff for agent1 as target")
        self.assertEqual(t_payoff, expected_t_payoff, "Incorrect initiator payoff for agent2 as initiator")
        expected_stats = [(interaction.db_id, interaction.payoffs['a']['a'][0], interaction.payoffs['a']['a'][1])]
        self.assertEqual(stats, expected_stats, "Incorrect stats for agent2 initiating")

    def test_handle_cached_target(self):
        cache_result = (0.5, 0.3, [(1, 0.5, 0.3)])
        self.agent1.handle_cached_target(self.agent2, cache_result)
        self.assertEqual(self.agent1.utils, cache_result[0], "Agent1 utils not updated correctly")
        self.assertEqual(self.agent2.utils, cache_result[1], "Agent2 utils not updated correctly")
        self.assertEqual(self.mock_model.cached, len(cache_result[2]),  "Cached interactions count not updated correctly")

    def test_handle_new_target(self):
        # Call handle_new_target
        self.agent1.handle_new_target(self.agent2, self.mock_model.cached_payoffs)

        # Assertions
        self.assertIn(self.agent1.phenotype, self.mock_model.cached_payoffs)
        self.assertIn(self.agent2.phenotype, self.mock_model.cached_payoffs[self.agent1.phenotype])
        self.assertEqual(self.agent1.utils, self.mock_model.cached_payoffs[self.agent1.phenotype][self.agent2.phenotype][0])
        self.assertEqual(self.agent2.utils, self.mock_model.cached_payoffs[self.agent1.phenotype][self.agent2.phenotype][1])
        self.assertEqual(len(self.mock_model.cached_payoffs[self.agent1.phenotype][self.agent2.phenotype][2]), self.mock_model.new)

    def test_interact(self):
        # Test case 1: Call interact for agent1, expect env interaction only
        self.agent1.interact()
        self.assertEqual(self.mock_model.env, 1)
        self.assertEqual(self.mock_model.cached, 0)
        self.assertEqual(self.mock_model.new, 0)
        self.reset_model_counters()

        # Test case 2: Call interact for agent2, expect env interaction and new interaction
        self.agent2.interact()
        self.assertEqual(self.mock_model.env, 1)
        self.assertEqual(self.mock_model.cached, 0)
        self.assertEqual(self.mock_model.new, 2)
        self.reset_model_counters()

        # Test case 3: Call interact for agent2 again, expect env interaction and cached interaction
        self.agent2.interact()
        self.assertEqual(self.mock_model.env, 1)
        self.assertEqual(self.mock_model.cached, 2)
        self.assertEqual(self.mock_model.new, 0)
        self.reset_model_counters()

        # Test case 4: Empty agent2's target neighbors and call interact, expect only env interaction
        self.agent2.role.neighbors['targets'] = []
        self.agent2.interact()
        self.assertEqual(self.mock_model.env, 1)
        self.assertEqual(self.mock_model.cached, 0)
        self.assertEqual(self.mock_model.new, 0)
        self.reset_model_counters()

    def reset_model_counters(self):
        # Reset model counters after each test case
        self.mock_model.env = 0
        self.mock_model.cached = 0
        self.mock_model.new = 0

    def test_process_trait_mutation(self):
        # Mock a feature and set up child traits
        mock_feature_F = Mock(name='F', values=['a', 'b'])
        mock_feature_F.next_trait.return_value = 'c'
        child_traits1 = {mock_feature_F: 'a'}
        mock_feature_G = Mock(name='G', values=['b'])
        mock_feature_G.next_trait.return_value = 'c'
        child_traits2 = {mock_feature_G: 'a'}

        # Test case: No mutation chance, expect no change in trait
        self.mock_model.trait_mutate_chance = 0.0
        result_traits = self.agent1.process_trait_mutation(child_traits1.copy())
        self.assertEqual(result_traits[mock_feature_F], 'a', "Child traits should not mutate")

        # Test case: Mutation and Creation chance set to 1, ValueError handled, expect mutation to 'b'
        self.mock_model.trait_mutate_chance = 1.0
        self.mock_model.trait_create_chance = 0.0
        result_traits = self.agent1.process_trait_mutation(child_traits2.copy())
        self.assertEqual(result_traits[mock_feature_G], 'b', "Child traits should mutate to 'b'")

        # Test case: Mutation and Creation chance set to 1, IndexError handled, expect no change in trait'
        self.mock_model.trait_mutate_chance = 1.0
        self.mock_model.trait_create_chance = 0.0
        mock_feature_G.values.remove('b')
        result_traits = self.agent1.process_trait_mutation(child_traits2.copy())
        self.assertEqual(result_traits[mock_feature_G], 'a', "Child traits should not mutate")

        # Test case: Mutation chance set to 1, expect mutation to 'b'
        self.mock_model.trait_mutate_chance = 1.0
        self.mock_model.trait_create_chance = 0.0
        result_traits = self.agent1.process_trait_mutation(child_traits1.copy())
        self.assertEqual(result_traits[mock_feature_F], 'b', "Child traits should mutate to 'b'")

        # Test case: Mutation and Creation chance set to 1, expect mutation to 'c'
        self.mock_model.trait_mutate_chance = 1.0
        self.mock_model.trait_create_chance = 1.0
        result_traits = self.agent1.process_trait_mutation(child_traits1.copy())
        self.assertEqual(result_traits[mock_feature_F], 'c', "Child traits should mutate to 'c'")

        # Test case: Mutation and Creation chance set to 1, agent1.shadow = True, expect mutation to 'b'
        self.mock_model.trait_mutate_chance = 1.0
        self.mock_model.trait_create_chance = 1.0
        self.agent1.shadow = True
        result_traits = self.agent1.process_trait_mutation(child_traits1.copy())
        self.assertEqual(result_traits[mock_feature_F], 'b', "Child traits should mutate to 'b'")

    def test_process_feature_mutation(self):
        mock_feature_F = Mock(name='F', values=['a'], env=False)
        mock_feature_G = Mock(name='G', values=['a'], env=False)
        mock_feature_H = Mock(name='H', values=['a'], env=False)
        child_traits = {mock_feature_G: 'a'}
        self.mock_model.feature_interactions.nodes = [mock_feature_F, mock_feature_G]
        self.mock_model.next_feature.return_value = mock_feature_H

        # Test case: Feature creation chance is 1 expect new feature addition
        self.mock_model.feature_create_chance = 1.0
        result_traits = self.agent1.process_feature_mutation(child_traits.copy())
        self.assertIn(mock_feature_H, result_traits, "New feature H should be added")
        self.assertEqual(result_traits[mock_feature_H], 'a', "New feature H should have value 'a'")

        # Test case: Feature gain chance is high, expect existing feature mutation
        self.mock_model.feature_create_chance = 0.0
        self.mock_model.feature_gain_chance = 1.0
        result_traits = self.agent1.process_feature_mutation(child_traits.copy())
        self.assertIn(mock_feature_G, result_traits, "Feature G should be added")
        self.assertEqual(result_traits[mock_feature_G], 'a', "Feature G should have value 'a'")

        # Test case: Both chances are low, expect feature removal
        self.mock_model.feature_create_chance = 0.0
        self.mock_model.feature_gain_chance = 0.0
        child_traits = {mock_feature_F: 'a', mock_feature_G: 'a'}
        result_traits = self.agent1.process_feature_mutation(child_traits.copy())
        self.assertTrue(len(result_traits) < len(child_traits), "A feature should be removed")

        # Test case: Agent is shadow, ensure at least one feature remains
        self.agent1.shadow = True
        self.mock_model.feature_create_chance = 0.0
        self.mock_model.feature_gain_chance = 0.0
        child_traits = {mock_feature_F: 'a'}
        result_traits = self.agent1.process_feature_mutation(child_traits.copy())
        self.assertTrue(len(result_traits) > 0, "Shadow agent should retain at least one feature")

    def test_reproduce(self):
        # Mock necessary attributes and methods
        self.mock_model.next_id.return_value = 100
        self.mock_model.base_agent_utils = 0.0
        self.mock_model.trait_mutate_chance = 0.0
        self.mock_model.feature_mutate_chance = 0.0
        self.mock_model.schedule.add = Mock()
        self.agent1.get_shadow_agent = Mock()
        self.agent1.shadow = False
        self.agent1.traits = {'F': 'a', 'H': 'a'}
        self.agent1.pos = (0, 0)
        self.agent1.site.born = 0

        # Test reproduction for a non-shadow agent and its corresponding shadow agent
        with patch("model.agents.Agent") as MockAgent:
            MockAgent.return_value.get_site = Mock()
            self.agent1.reproduce()

            # Assertions for non-shadow agent
            self.mock_model.schedule.add.assert_called_once()  # Non-shadow agents are added to schedule
            self.agent1.get_shadow_agent().reproduce.assert_called_once()  # Corresponding shadow agent also reproduces
            self.mock_model.grid.place_agent.assert_called_once()  # New agent is placed on the grid
            MockAgent.return_value.get_site.assert_called_once()
            self.assertEqual(self.agent1.site.born, 1, "Birth count should increment for the non-shadow agent's site")
            self.assertEqual(self.mock_model.grid.place_agent.call_args[0][1], self.agent1.pos, "New agent should be placed at the parent agent's position")

            # Reset mocks for next test
            self.mock_model.schedule.add.reset_mock()
            self.agent1.get_shadow_agent().reproduce.reset_mock()
            self.mock_model.grid.place_agent.reset_mock()

            # Test reproduction for a shadow agent
            self.agent1.shadow = True
            shadow_site = Mock(born=0)  # Mock shadow site
            self.agent1.site = shadow_site  # Assign shadow site to shadow agent
            self.agent1.reproduce()

            # Assertions for shadow agent
            self.mock_model.schedule.add.assert_not_called()  # Shadow agents are not added to schedule
            self.agent1.get_shadow_agent().reproduce.assert_not_called()  # Shadow agent does not trigger another shadow agent to reproduce
            self.mock_model.grid.place_agent.assert_called_once()  # New shadow agent is placed on the grid
            self.assertEqual(shadow_site.born, 1, "Birth count should increment for shadow agent's shadow site")
            self.assertEqual(self.mock_model.grid.place_agent.call_args[0][1], self.agent1.pos, "New shadow agent should be placed at the parent agent's position")

    def test_move(self):
        # Mock the necessary components for the agent and its site
        self.agent1.decrement_phenotype = Mock()
        self.agent1.site = Mock(moved_out=0, moved_in=0)
        self.agent1.get_site = Mock(return_value=self.agent1.site)
        self.mock_model.grid.get_neighborhood = Mock(return_value=[(0, 1), (1, 0)])
        self.mock_model.random.choice = Mock(return_value=(0, 1))
        self.mock_model.grid.move_agent = Mock()
        
        # Mock shadow agent and its properties
        shadow_site = Mock(moved_out=0, moved_in=0)
        shadow_agent = Mock(model=self.mock_model, site=shadow_site, decrement_phenotype=Mock(), get_site=Mock(return_value=shadow_site))

        self.agent1.get_shadow_agent = Mock(return_value=shadow_agent)
        
        # Call move method
        self.agent1.move()
        
        # Assert that site counts are updated
        self.assertEqual(self.agent1.site.moved_out, 1)
        self.assertEqual(self.agent1.site.moved_in, 1)
        
        # Assert that decrement_phenotype was called
        self.agent1.decrement_phenotype.assert_called_once()
        
        # Assert that agent and shadow agent were moved to the new position
        # Check all calls made to move_agent and assert the expected calls are in the list
        expected_calls = [call(shadow_agent, (0, 1)), call(self.agent1, (0, 1))]
        actual_calls = self.mock_model.grid.move_agent.call_args_list
        self.assertEqual(actual_calls, expected_calls, "move_agent should be called with correct arguments for both agents")
        self.agent1.get_site.assert_called_once()
        
        # check shadow agent interactions
        shadow_agent.decrement_phenotype.assert_called_once()
        shadow_agent.get_site.assert_called_once()
        self.assertEqual(shadow_agent.site.moved_out, 1)
        self.assertEqual(shadow_agent.site.moved_in, 1)

    def test_die(self):
        # Mock the necessary components for the agent
        self.agent1.decrement_phenotype = Mock()
        self.agent1.site = Mock(died=0)
        self.mock_model.grid.remove_agent = Mock()
        self.mock_model.schedule = Mock()
        
        # Create a real Agent instance for the shadow agent
        shadow_site = Mock(died=0)
        shadow_agent = Agent(
            unique_id=999,
            model=self.mock_model,
            traits=self.agent1_traits,
            utils=10,
            shadow=True
        )
        shadow_agent.site = shadow_site  # Assign the mocked site to the shadow agent
        shadow_agent.decrement_phenotype = Mock()
        self.agent1.get_shadow_agent = Mock(return_value=shadow_agent)

        # Call die method for agent1 (regular agent, not a shadow)
        self.agent1.shadow = False
        self.agent1.die()

        # Assert that the regular agent is removed from the schedule
        self.mock_model.schedule.remove.assert_called_once_with(self.agent1)

        # Assert that decrement_phenotype was called for both agents
        self.agent1.decrement_phenotype.assert_called_once()
        shadow_agent.decrement_phenotype.assert_called_once()

        # Assert that both agents were removed from their respective grids
        self.mock_model.grid.remove_agent.assert_any_call(self.agent1)
        self.mock_model.grid.remove_agent.assert_any_call(shadow_agent)

        # Assert that the death count is incremented for both the agent's site and the shadow agent's site
        self.assertEqual(self.agent1.site.died, 1, "Death count should increment by 1 for regular agent's site")
        self.assertEqual(shadow_agent.site.died, 1, "Death count should increment by 1 for shadow agent's site")

    def test_step(self):
        # Mock the methods called within step
        self.agent1.interact = Mock(side_effect=lambda: setattr(self.agent1, 'utils', self.agent1.utils - 1))
        self.agent1.die = Mock()
        self.agent1.reproduce = Mock()
        self.agent1.move = Mock()
        self.agent1.feature_cost = 1  # Set feature cost
        self.agent1.age = 0  # Set initial age
        self.mock_model.sites[self.agent1.pos].pop_cost = .5  # Set population cost at the site
        self.mock_model.repr_multi = 1  # Set reproduction multiplier
        self.mock_model.move_chance = 0.01  # Set move chance
        self.mock_model.random.random = Mock(return_value=0.05)  # Mock random chance for moving

        # Test case: Agent has negative utility, doesn't interact, dies
        self.agent1.utils = -1
        self.agent1.step()
        self.agent1.interact.assert_not_called()
        self.agent1.die.assert_called_once()
        self.agent1.reproduce.assert_not_called()
        self.agent1.move.assert_not_called()
        self.reset_step_mock()

        # Test case: Agent has non-negative utility, interacts, and then dies due to negative utility
        self.agent1.utils = 0
        self.agent1.step()
        self.agent1.interact.assert_called_once()
        self.agent1.die.assert_called_once()
        self.agent1.reproduce.assert_not_called()
        self.agent1.move.assert_not_called()
        self.reset_step_mock()
        
        # Test case: Agent has non-negative utility, doesn't die, and doesn't reproduce or move
        self.agent1.utils = 2
        self.agent1.step()
        self.agent1.die.assert_not_called()
        self.agent1.reproduce.assert_not_called()
        self.agent1.move.assert_not_called()
        self.reset_step_mock()

        # Test case: Agent has non-negative utility, doesn't die, does reproduce, doesn't move
        self.agent1.utils = 5
        self.agent1.step()
        self.agent1.die.assert_not_called()
        self.agent1.reproduce.assert_called_once()
        self.agent1.move.assert_not_called()
        self.reset_step_mock()

        # Test case: Agent has non-negative utility, doesn't die, does reproduce and moves due to move chance
        self.agent1.utils = 5
        self.mock_model.move_chance = 0.1 # random < move_chance
        self.agent1.step()
        self.agent1.interact.assert_called_once()
        self.agent1.die.assert_not_called()
        self.agent1.reproduce.assert_called_once()
        self.agent1.move.assert_called_once()
        self.reset_step_mock()

        # Test case: Agent has non-negative utility, doesn't die, does reproduce and moves due to self.start = self.utils
        self.agent1.utils = 5
        self.agent1.interact = Mock() # no utility change from interactions
        self.mock_model.move_chance = 0.01 # random > move chance
        self.agent1.step()
        self.agent1.interact.assert_called_once()
        self.agent1.die.assert_not_called()
        self.agent1.reproduce.assert_called_once()
        self.agent1.move.assert_called_once()

    def reset_step_mock(self):
        self.agent1.interact.reset_mock()
        self.agent1.die.reset_mock()
        self.agent1.reproduce.reset_mock()
        self.agent1.move.reset_mock()     

    def tearDown(self):
        # Reset mock objects
        self.mock_model.reset_mock()
        self.mock_role1.reset_mock()
        self.mock_role2.reset_mock()
        self.mock_site.reset_mock()
        self.mock_interaction1.reset_mock()
        self.mock_interaction2.reset_mock()
        self.mock_interaction3.reset_mock()
        self.mock_interaction4.reset_mock()
        self.mock_interaction5.reset_mock()
        self.mock_interaction6.reset_mock()
        self.mock_interaction7.reset_mock()
        self.mock_interaction8.reset_mock()

        # Clear references to mock objects and other state
        self.mock_model = None
        self.mock_role1 = None
        self.mock_role2 = None
        self.mock_site = None
        self.agent1 = None
        self.agent2 = None
        self.mock_interaction1 = None
        self.mock_interaction2 = None
        self.mock_interaction3 = None
        self.mock_interaction4 = None
        self.mock_interaction5 = None
        self.mock_interaction6 = None
        self.mock_interaction7 = None
        self.mock_interaction8 = None
        self.agent1_traits = None
        self.agent2_traits = None
        self.mock_pos = None

        self.shuffle_patch.stop()


if __name__ == '__main__':
    unittest.main()
