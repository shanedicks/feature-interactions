import sqlite3
from typing import Any, Dict, List, Tuple, Union

class Manager():

    def __init__(
        self,
        path_to_db: str,
        db_name: str
    ) -> None:
        self.path_to_db = path_to_db
        self.db_name = db_name
        self.db_string = "{0}/{1}".format(path_to_db, db_name)

    def inspect_db(self) -> sqlite3.Cursor:
        conn = sqlite3.connect(self.db_string)
        c = conn.cursor()
        results = c.execute("SELECT * FROM sqlite_master")
        return results

    def initialize_db(self) -> None:
        conn = sqlite3.connect(self.db_string)
        conn.execute("PRAGMA foreign_keys = ON")

        networks_table = """
            CREATE TABLE IF NOT EXISTS networks (
                network_id            INTEGER PRIMARY KEY,
                init_env_features     INTEGER NOT NULL,
                init_agent_features   INTEGER NOT NULL,
                max_interactions      INTEGER NOT NULL,
                trait_payoff_mod      REAL NOT NULL,
                anchor_bias           REAL NOT NULL,
                payoff_bias           REAL NOT NULL
            )"""

        features_table = """
            CREATE TABLE IF NOT EXISTS features (
                feature_id INTEGER PRIMARY KEY,
                network_id INTEGER NOT NULL
                               REFERENCES networks (network_id),
                name       TEXT NOT NULL,
                env        INTEGER NOT NULL
            )"""
        traits_table = """
            CREATE TABLE IF NOT EXISTS traits (
                trait_id   INTEGER PRIMARY KEY,
                name       TEXT NOT NULL,
                feature_id INTEGER NOT NULL
                               REFERENCES features (feature_id)
            )"""
        interactions_table = """
            CREATE TABLE IF NOT EXISTS interactions (
                interaction_id   INTEGER PRIMARY KEY,
                network_id       INTEGER NOT NULL
                                     REFERENCES networks (network_id),
                initiator        INTEGER NOT NULL
                                     REFERENCES features (feature_id),
                target           INTEGER NOT NULL
                                     REFERENCES features (feature_id),
                i_anchor         REAL NOT NULL,
                t_anchor         REAL NOT NULL
            )"""
        payoffs_table = """
            CREATE TABLE IF NOT EXISTS payoffs (
                payoff_id       INTEGER PRIMARY KEY,
                interaction_id  INTEGER NOT NULL,
                initiator       INTEGER NOT NULL
                                    REFERENCES traits (trait_id),
                target          INTEGER NOT NULL
                                    REFERENCES traits (trait_id),
                initiator_utils REAL NOT NULL,
                target_utils    REAL NOT NULL
            )"""
        worlds_table = """
            CREATE TABLE IF NOT EXISTS worlds (
                world_id              INTEGER PRIMARY KEY,
                trait_mutate_chance   REAL NOT NULL,
                trait_create_chance   REAL NOT NULL,
                feature_mutate_chance REAL NOT NULL,
                feature_create_chance REAL NOT NULL,
                feature_gain_chance   REAL NOT NULL,
                init_agents           INTEGER NOT NULL,
                base_agent_utils      REAL NOT NULL,
                base_env_utils        REAL NOT NULL,
                total_pop_limit       INTEGER NOT NULL,
                pop_cost_exp          REAL NOT NULL,
                feature_cost_exp      REAL NOT NULL,
                grid_size             INTEGER NOT NULL,
                repr_multi            INTEGER NOT NULL,
                mortality             REAL NOT NULL,
                move_chance           REAL NOT NULL,
                snap_interval         INTEGER NOT NULL,
                feature_timeout       INTEGER NOT NULL,
                trait_timeout         INTEGER NOT NULL,
                target_sample         INTEGER NOT NULL,
                network_id            INTEGER NOT NULL
                                          REFERENCES networks (network_id)
            )"""
        spacetime_table = """
            CREATE TABLE IF NOT EXISTS spacetime (
                spacetime_id INTEGER PRIMARY KEY,
                world_id     INTEGER NOT NULL
                                 REFERENCES worlds (world_id),
                step_num     INTEGER NOT NULL,
                site_pos     TEXT NOT NULL
            )"""
        feature_changes_table = """ 
            CREATE TABLE IF NOT EXISTS feature_changes (
                feature_changes_id INTEGER PRIMARY KEY,
                spacetime_id       INTEGER NOT NULL
                                       REFERENCES spacetime (spacetime_id),
                feature_id         INTEGER NOT NULL
                                       REFERENCES features (feature_id),
                change             TEXT NOT NULL
            )"""
        trait_changes_table = """ 
            CREATE TABLE IF NOT EXISTS trait_changes (
                trait_changes_id INTEGER PRIMARY KEY,
                spacetime_id     INTEGER NOT NULL
                                     REFERENCES spacetime (spacetime_id),
                trait_id         INTEGER NOT NULL
                                     REFERENCES traits (trait_id),
                change           TEXT NOT NULL
            )"""
        model_vars_table = """
            CREATE TABLE IF NOT EXISTS model_vars (
                record_id     INTEGER PRIMARY KEY,
                spacetime_id  INTEGER NOT NULL
                                  REFERENCES spacetime (spacetime_id),
                pop           INTEGER NOT NULL,
                total_utility REAL NOT NULL,
                mean_utility  REAL NOT NULL,
                med_utility   REAL NOT NULL,
                num_types     INTEGER NOT NULL,
                num_roles     INTEGER NOT NULL,
                num_features  INTEGER NOT NULL
            )"""
        phenotypes_table = """
            CREATE TABLE IF NOT EXISTS phenotypes (
                record_id    INTEGER PRIMARY KEY,
                spacetime_id INTEGER NOT NULL
                                 REFERENCES spacetime (spacetime_id),
                shadow       INTEGER NOT NULL,
                phenotype    TEXT NOT NULL,
                pop          INTEGER NOT NULL
            )"""
        demographics_table = """
            CREATE TABLE IF NOT EXISTS demographics (
                record_id    INTEGER PRIMARY KEY,
                spacetime_id INTEGER NOT NULL
                                 REFERENCES spacetime (spacetime_id),
                pop          INTEGER NOT NULL,
                born         INTEGER NOT NULL,
                died         INTEGER NOT NULL,
                moved_in     INTEGER NOT NULL,
                moved_out    INTEGER NOT NULL
            )"""
        environment_table = """
            CREATE TABLE IF NOT EXISTS environment (
            record_id    INTEGER PRIMARY KEY,
            spacetime_id INTEGER NOT NULL
                             REFERENCES spacetime (spacetime_id),
            trait_id     INTEGER NOT NULL
                             REFERENCES traits (trait_id),
            utils        REAL NOT NULL
        )"""
        tables = [
            networks_table,
            features_table,
            traits_table, 
            interactions_table,
            payoffs_table,
            worlds_table,
            spacetime_table,
            feature_changes_table,
            trait_changes_table,
            model_vars_table,
            phenotypes_table,
            demographics_table,
            environment_table,
        ]
        for table in tables:
            conn.execute(table)
        conn.close()

    def write_row(self, table_name: str, values_tuple: Tuple[Any]) -> int:
        sql_params = ",".join(['?'] * len(values_tuple))
        sql = "INSERT INTO {0} VALUES (Null, {1})".format(table_name, sql_params)
        conn = sqlite3.connect(self.db_string)
        c = conn.cursor()
        c.execute(sql, values_tuple)
        row_id = c.lastrowid
        conn.commit()
        conn.close()
        return row_id

    def write_rows(self, rows_dict: Dict[str, List[Tuple[Any]]]) -> None:
        conn = sqlite3.connect(self.db_string)
        for table_name, rows_list in rows_dict.items():
            sql_params = ",".join(['?'] * len(rows_list[0]))
            sql = "INSERT INTO {0} VALUES (Null, {1})".format(table_name, sql_params)
            conn.executemany(sql, rows_list)
        conn.commit()
        conn.close()

    def get_next_record(self, sql: str, keys: List[str]) -> Union[Dict[str, Any], None]:
        conn = sqlite3.connect(self.db_string)
        record = conn.execute(sql).fetchone()
        if record:
            record = dict(zip(keys, record))
        conn.close()
        return record

    def get_next_feature(
        self,
        network_id: int,
        feature_id: int
    ) -> Union[Dict[str, Any], None]:
        keys = ["feature_id", "name", "env"]
        sql = """
            SELECT {0}
            FROM features
            WHERE network_id = {1}
            AND feature_id > {2}
        """.format(", ".join(keys), network_id, feature_id)

        record = self.get_next_record(sql, keys)
        if record:
            record['env'] = bool(record['env'])
        return record

    def get_next_trait(
        self,
        feature_id: int,
        trait_name: str,
    ) -> Union[Dict[str, Any], None]:
        keys = ["trait_id"]
        sql = """
            SELECT {0}
            FROM traits
            WHERE feature_id = {1}
            AND name = '{2}'
        """.format(", ".join(keys), feature_id, str(trait_name))
        return self.get_next_record(sql, keys)

    def get_records(self, sql: str, keys: List[str]) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_string)
        records = conn.execute(sql).fetchall()
        conn.close()
        return [dict(zip(keys, record)) for record in records]

    def get_feature_interactions(self, feature_id) -> List[Dict[str, Any]]:
        sql = """
            SELECT interactions.interaction_id,
                   i.name,
                   t.name,
                   interactions.i_anchor,
                   interactions.t_anchor
            FROM interactions
            LEFT JOIN features AS i ON initiator = i.feature_id
            LEFT JOIN features AS t ON target = t.feature_id
            WHERE initiator = {0}
        """.format(feature_id)
        keys = [
            "db_id",
            "initiator",
            "target",
            "i_anchor",
            "t_anchor"
        ]
        return self.get_records(sql, keys)

    def get_interaction_payoffs(
        self,
        interaction_id: int,
        i_traits: List[str],
        t_traits: List[str]
    ) -> List[Dict[str, Any]]:
        sql = """
            SELECT i.name,
                   t.name,
                   payoffs.initiator_utils,
                   payoffs.target_utils
            FROM payoffs
            LEFT JOIN traits AS i ON initiator = i.trait_id
            LEFT JOIN traits AS t ON target = t.trait_id
            WHERE interaction_id = {0}
        """.format(interaction_id, tuple(t_traits))
        if len(i_traits) > 1:
            sql = sql + " AND i.name IN {0}".format(tuple(i_traits))
        elif len(i_traits) == 1:
            sql = sql + " AND i.name = '{0}'".format(i_traits[0])
        if len(t_traits) > 1:
            sql = sql + " AND t.name IN {0}".format(tuple(t_traits))
        elif len(t_traits) == 1:
            sql = sql + " AND t.name = '{0}'".format(t_traits[0])
        keys = [
            "initiator",
            "target",
            "i_utils",
            "t_utils"
        ]
        return self.get_records(sql, keys)

    def get_spacetimes(
        self,
        world_id,
        step_num
    ) -> List[Dict[str, Any]]:
        sql = """
            SELECT site_pos,
                   spacetime_id
            FROM spacetime
            WHERE world_id = {0}
            AND step_num = {1}
        """.format(world_id, step_num)
        keys = ["pos", "id"]
        return self.get_records(sql, keys)
