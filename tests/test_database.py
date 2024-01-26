import sqlite3
import unittest
import pandas as pd
from unittest.mock import patch, MagicMock
from model.database import Manager

class TestManager(unittest.TestCase):

    def setUp(self):
        # Start patching 'sqlite3' and store the mock for use in tests
        self.sqlite3_patch = patch('model.database.sqlite3')
        self.mock_sqlite = self.sqlite3_patch.start()
        
        # Mock the connection and cursor
        self.mock_conn = MagicMock()
        self.mock_cursor = MagicMock()
        self.mock_conn.cursor.return_value = self.mock_cursor
        self.mock_sqlite.connect.return_value = self.mock_conn

        # Initialize the Manager instance
        self.path_to_db = 'mock_path'
        self.db_name = 'mock_db'
        self.manager = Manager(path_to_db=self.path_to_db, db_name=self.db_name)

        self.mock_world = MagicMock()
        self.mock_world.current_feature_db_id = 1  # Example value
        self.mock_world.network_dfs = {
            'features': pd.DataFrame({
                'feature_id': [1, 2, 3],
                'name': ['A', 'B', 'C'],
                'env': [0, 1, 1]
            }),
            'traits': pd.DataFrame({
                'trait_id': [1, 2, 3],
                'name': ['a', 'b', 'c'],
                'feature_id': [1, 2, 3]
            }),
            'interactions': pd.DataFrame({
                'interaction_id': [1, 2, 3],
                'initiator': [1, 2, 3],
                'target': [3, 2, 1],
            }),
            'payoffs': pd.DataFrame({
                'interaction_id': [1, 2, 3],
                'initiator': [1, 2, 3],
                'target': [3, 3, 1],
                'initiator_utils': [0.1, 0.2, 0.3],
                'target_utils': [0.3, 0.2, 0.1]
            })
        }

    def tearDown(self):
        self.sqlite3_patch.stop()

    def test_init(self):
        self.assertEqual(self.manager.path_to_db, self.path_to_db)
        self.assertEqual(self.manager.db_name, self.db_name)
        self.assertEqual(self.manager.db_string, f"{self.path_to_db}/{self.db_name}")

    def test_get_connection(self):
        result_conn = self.manager.get_connection()
        self.mock_sqlite.connect.assert_called_once_with(self.manager.db_string, timeout=300)
        self.assertEqual(result_conn, self.mock_conn)

    def test_inspect_db(self):
        self.mock_cursor.execute.return_value = self.mock_cursor
        result = self.manager.inspect_db()
        self.mock_cursor.execute.assert_called_once_with("SELECT * FROM sqlite_master")
        self.assertEqual(result, self.mock_cursor)

    def test_get_network_dataframes(self):
        network_id = 1  # Example network_id

        # Mock the dependent methods and the connection
        self.manager.get_connection = MagicMock(return_value=self.mock_conn)
        self.manager.get_features_dataframe = MagicMock(return_value=pd.DataFrame({'dummy_column': [1, 2, 3]}))
        self.manager.get_interactions_dataframe = MagicMock(return_value=pd.DataFrame({'dummy_column': [4, 5, 6]}))
        self.manager.get_traits_dataframe = MagicMock(return_value=pd.DataFrame({'dummy_column': [7, 8, 9]}))
        self.manager.get_payoffs_dataframe = MagicMock(return_value=pd.DataFrame({'dummy_column': [10, 11, 12]}))

        # Call the method under test
        result_nd = self.manager.get_network_dataframes(network_id)

        # Assert the dependent methods were called with correct arguments
        self.manager.get_connection.assert_called_once()
        self.manager.get_features_dataframe.assert_called_once_with(self.mock_conn, network_id)
        self.manager.get_interactions_dataframe.assert_called_once_with(self.mock_conn, network_id)
        self.manager.get_traits_dataframe.assert_called_once_with(self.mock_conn, network_id)
        self.manager.get_payoffs_dataframe.assert_called_once_with(self.mock_conn, network_id)

        # Assert the method returns the correct dictionary structure
        self.assertIsInstance(result_nd, dict)
        self.assertListEqual(list(result_nd.keys()), ['features', 'interactions', 'traits', 'payoffs'])
        for key, df in result_nd.items():
            self.assertIsInstance(df, pd.DataFrame)

    def test_get_next_feature(self):
        # Scenario 1: next feature exists
        expected_result = {'feature_id': 2, 'name': 'B', 'env': True}

        # Call the method under test
        result = self.manager.get_next_feature(self.mock_world)

        self.assertEqual(result, expected_result)

        # Scenario 2: next feature does not exist
        self.mock_world.current_feature_db_id = 3 # highest db_id in mock dataframe

        # Call the method under test
        result = self.manager.get_next_feature(self.mock_world)

        self.assertIsNone(result)

    def test_get_next_trait_exists(self):
        # Scenario 1: next trait exists
        # parameters matching trait in mock dataframe
        feature_id = 2
        trait_name = 'b'
        expected_result = {'trait_id': 2, 'name': 'b', 'feature_id': 2}

        # Call the method under test
        result = self.manager.get_next_trait(self.mock_world, feature_id, trait_name)

        self.assertEqual(result, expected_result)

        # Scenario 2: next trait does not exist
        # no matching trait in mock dataframe
        feature_id = 2
        trait_name = 'f'

        # Call the method under test
        result = self.manager.get_next_trait(self.mock_world, feature_id, trait_name)

        self.assertIsNone(result)

    def test_get_feature_interactions(self):
        feature_id = 2  

        # Expected result after filtering by feature_id 2
        expected_result = [{'db_id': 2, 'initiator': 2, 'target': 2}]  # Example row after renaming and filtering

        # Call the method under test
        result = self.manager.get_feature_interactions(self.mock_world, feature_id)

        self.assertEqual(result, expected_result)

    def test_get_interaction_payoffs(self):
        # Setup parameters for the test
        interaction_id = 2
        i_traits = ['a', 'b']  # Initiator traits
        t_traits = ['c']  # Target traits

        # Expected result after filtering and renaming
        expected_result = [
            {
                'i_utils': 0.2,  # Initiator utils corresponding to interaction_id 2
                't_utils': 0.2,  # Target utils corresponding to interaction_id 2
                'initiator': 'b',  # Name of trait with trait_id 2
                'target': 'c'  # Name of trait with trait_id 3
            }
        ] # No payoff for i_trait 'a' with interaction_id 2

        # Call the method under test
        result = self.manager.get_interaction_payoffs(self.mock_world, interaction_id, i_traits, t_traits)

        # Assert the result is as expected
        self.assertEqual(result, expected_result)


class ConnectionWrapper:
    def __init__(self, connection):
        self.connection = connection
        self.close_called = False

    def __getattr__(self, name):
        # Delegate attribute access to the underlying connection object
        return getattr(self.connection, name)

    def close(self):
        # Set a flag when close is called instead of actually closing the connection
        self.close_called = True


class TestManagerIntegration(unittest.TestCase):

    def setUp(self):
        # Create an in-memory database
        raw_connection = sqlite3.connect(':memory:')
        self.connection = ConnectionWrapper(raw_connection)

        # Patch the close method of the connection
        self.patcher = patch.object(self.connection, 'close', MagicMock())
        self.patcher.start()

        # Initialize the Manager instance with the in-memory database
        self.manager = Manager(path_to_db='', db_name='')
        # Override get_connection to return the in-memory connection
        self.manager.get_connection = lambda: self.connection

        self.data = {
            'networks': [
                (1, 10, 10, 100, 1.5, 0.5, 0.5)
            ],
            'features': [
                (1, 1, 'A', 0),
                (2, 1, 'B', 1),
                (3, 1, 'C', 0),
                (4, 1, 'D', 1),
                (5, 1, 'E', 0)
            ],
            'traits': [
                (1, 'a', 1),
                (2, 'b', 2),
                (3, 'c', 3),
                (4, 'd', 4),
                (5, 'e', 5),
                (6, 'f', 1),
                (7, 'g', 2)
            ],
            'interactions': [
                (1, 1, 1, 2, 0.1, 0.2),
                (2, 1, 2, 3, 0.2, 0.3),
                (3, 1, 3, 4, 0.1, 0.3),
                (4, 1, 4, 5, 0.2, 0.4)
            ],
            'payoffs': [
                (1, 1, 1, 2, 0.10, 0.20),
                (2, 1, 2, 3, 0.20, 0.30),
                (3, 2, 3, 4, 0.15, 0.25),
                (4, 2, 4, 5, 0.20, 0.20),
                (5, 3, 1, 5, -0.10, 0.10),
                (6, 3, 2, 4, 0.15, -0.15),
                (7, 4, 3, 1, 0.20, -0.10)
            ]
        }


        # Initialize database and insert data
        self.manager.initialize_db()
        self.insert_data()

    def insert_data(self):
        # Insert data in the order of table dependencies
        for table in ['networks', 'features', 'traits', 'interactions', 'payoffs']:
            rows = self.data.get(table)
            if not rows:
                continue
            placeholders = ', '.join('?' * len(rows[0]))  # Create placeholders for each row
            sql = f"INSERT INTO {table} VALUES ({placeholders})"
            self.connection.executemany(sql, rows)
        self.connection.commit()

    def test_initialize_db(self):
        # Verify that the tables were created
        cursor = self.connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]

        # Define the expected table names
        expected_tables = ['networks', 'features', 'traits', 'interactions', 'payoffs', 'worlds',
                           'spacetime', 'feature_changes', 'trait_changes', 'interaction_stats',
                           'model_vars', 'phenotypes', 'demographics', 'environment']

        # Assert that all expected tables are created
        self.assertCountEqual(tables, expected_tables)  # assertCountEqual ignores order

    def test_write_row(self):
        # Setup parameters for the test - using 'networks' as an example
        table_name = 'networks'
        values_tuple = (10, 10, 100, 1.5, 0.5, 0.5)  # auto-incremented primary key handled automatically

        # Call the method under test
        row_id = self.manager.write_row(table_name, values_tuple)

        # Assert the row ID was returned correctly
        self.assertIsInstance(row_id, int)
        self.assertEqual(row_id, 2)  # 'networks' already has 1 row inserted in setUp

        # Assert the data was inserted correctly
        cursor = self.connection.cursor()
        cursor.execute(f"SELECT * FROM {table_name} WHERE network_id = ?", (row_id,))
        inserted_row = cursor.fetchone()

        # Create the expected row (replace None with the actual row_id for comparison)
        expected_row = (row_id,) + values_tuple

        # Assert the inserted row matches the expected row
        self.assertEqual(inserted_row, expected_row)

    def test_write_rows(self):
        # Setup parameters for the test
        rows_dict = {
            'traits': [
                (8, 'j', 4),  # ID provided by the model, not auto-incremented
                (9, 'k', 5)
            ],
            'payoffs': [  # No ID provided, should auto-increment
                (3, 2, 4, 0.25, 0.35),
                (3, 4, 5, 0.15, 0.45)
            ]
        }

        # Call the method under test
        self.manager.write_rows(rows_dict)

        # Assert the data was inserted correctly for each table in rows_dict
        for table_name, rows_list in rows_dict.items():
            cursor = self.connection.cursor()
            cursor.execute(f"SELECT * FROM {table_name} ORDER BY ROWID DESC LIMIT {len(rows_list)}")
            inserted_rows = cursor.fetchall()

            # Prepare the expected rows for comparison
            expected_rows = rows_list
            if table_name == 'payoffs':  # Adjust for auto-incrementing IDs
                # Payoffs table primary key starts after existing data
                num_existing_rows = len(self.data['payoffs'])
                expected_rows = [
                    (num_existing_rows + i + 1,) + row for i, row in enumerate(rows_list)
                ]

            # Reverse the order of inserted_rows for comparison as we fetched the last rows in descending order
            inserted_rows.reverse()

            # Assert the inserted rows match the expected rows
            self.assertEqual(inserted_rows, expected_rows)

    def test_get_features_dataframe(self):
        # Setup parameter for the test
        network_id = 1

        # Call the method under test
        df = self.manager.get_features_dataframe(self.connection, network_id)


        # Define the expected DataFrame
        expected_data = {
            'feature_id': [1, 2, 3, 4, 5],  # IDs of features
            'name': ['A', 'B', 'C', 'D', 'E'],  # Names of features
            'env': [False, True, False, True, False]  # 'env' column values
        }
        expected_df = pd.DataFrame(expected_data)

        # Assert the returned DataFrame matches the expected DataFrame
        pd.testing.assert_frame_equal(df, expected_df, check_dtype=True)

        # Test with a network_id that doesn't have features
        network_id = 999  # This network_id doesn't exist in the test data
        df = self.manager.get_features_dataframe(self.connection, network_id)
        self.assertTrue(df.empty, "DataFrame should be empty for a non-existent network_id")

    def test_get_interactions_dataframe(self):
        # Setup parameter for the test
        network_id = 1

        # Call the method under test
        df = self.manager.get_interactions_dataframe(self.connection, network_id)

        # Define the expected DataFrame
        expected_data = {
            'interaction_id': [1, 2, 3, 4],  # IDs of interactions
            'initiator': [1, 2, 3, 4],  # Initiator feature IDs
            'target': [2, 3, 4, 5],  # Target feature IDs
            'i_anchor': [0.1, 0.2, 0.1, 0.2],  # Initiator anchor values
            't_anchor': [0.2, 0.3, 0.3, 0.4]  # Target anchor values
        }
        expected_df = pd.DataFrame(expected_data)

        # Assert the returned DataFrame matches the expected DataFrame
        pd.testing.assert_frame_equal(df, expected_df, check_dtype=True)

        # Test with a network_id that doesn't have interactions
        network_id = 999  # This network_id doesn't exist in the test data
        df = self.manager.get_interactions_dataframe(self.connection, network_id)
        self.assertTrue(df.empty, "DataFrame should be empty for a non-existent network_id")

    def test_get_traits_dataframe(self):
        # Setup parameter for the test
        network_id = 1

        # Call the method under test
        df = self.manager.get_traits_dataframe(self.connection, network_id)

        # Define the expected DataFrame
        expected_data = {
            'trait_id': [1, 2, 3, 4, 5, 6, 7],  # IDs of traits
            'name': ['a', 'b', 'c', 'd', 'e', 'f', 'g'],  # Names of traits
            'feature_id': [1, 2, 3, 4, 5, 1, 2]  # Corresponding feature IDs
        }
        expected_df = pd.DataFrame(expected_data)

        # Assert the returned DataFrame matches the expected DataFrame
        pd.testing.assert_frame_equal(df, expected_df, check_dtype=True)

        # Test with a network_id that doesn't have traits
        network_id = 999  # This network_id doesn't exist in the test data
        df = self.manager.get_traits_dataframe(self.connection, network_id)
        self.assertTrue(df.empty, "DataFrame should be empty for a non-existent network_id")

    def test_get_payoffs_dataframe(self):
        # Setup parameter for the test
        network_id = 1

        # Call the method under test
        df = self.manager.get_payoffs_dataframe(self.connection, network_id)
        print("Actual DataFrame:")
        print(df)

        # Define the expected DataFrame
        expected_data = {
            'interaction_id': [1, 1, 2, 2, 3, 3, 4],  # IDs of interactions
            'initiator': [1, 2, 3, 4, 1, 2, 3],  # Initiator trait IDs
            'target': [2, 3, 4, 5, 5, 4, 1],  # Target trait IDs
            'initiator_utils': [0.10, 0.20, 0.15, 0.20, -0.10, 0.15, 0.20],  # Initiator utility values
            'target_utils': [0.20, 0.30, 0.25, 0.20, 0.10, -0.15, -0.10]  # Target utility values
        }
        expected_df = pd.DataFrame(expected_data)

        # Assert the returned DataFrame matches the expected DataFrame
        pd.testing.assert_frame_equal(df, expected_df, check_dtype=True)

        # Test with a network_id that doesn't have payoffs
        network_id = 999  #This network_id doesn't exist in the test data
        df = self.manager.get_payoffs_dataframe(self.connection, network_id)
        self.assertTrue(df.empty, "DataFrame should be empty for a non-existent network_id")

    def tearDown(self):
        # Close the raw connection
        self.connection.connection.close()

if __name__ == '__main__':
    unittest.main()
