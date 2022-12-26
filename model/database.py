import sqlite3
import pandas as pd
from typing import Any, Dict, List, Tuple, Union
from output import get_feature_by_name

class Manager():

    def __init__(
        self,
        path_to_db: str,
        db_name: str
    ) -> None:
        self.path_to_db = path_to_db
        self.db_name = db_name
        self.db_string = f"{path_to_db}/{db_name}"

    def get_connection(self):
        conn = sqlite3.connect(self.db_string, timeout=10)
        return conn

    def inspect_db(self) -> sqlite3.Cursor:
        conn = self.get_connection()
        c = conn.cursor()
        results = c.execute("SELECT * FROM sqlite_master")
        return results

    def initialize_db(self) -> None:
        conn = self.get_connection()
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
        sql = f"INSERT INTO {table_name} VALUES (Null, {sql_params})"
        conn = self.get_connection()
        c = conn.cursor()
        c.execute(sql, values_tuple)
        row_id = c.lastrowid
        conn.commit()
        conn.close()
        return row_id

    def write_rows(self, rows_dict: Dict[str, List[Tuple[Any]]]) -> None:
        conn = self.get_connection()
        for table_name, rows_list in rows_dict.items():
            sql_params = ",".join(['?'] * len(rows_list[0]))
            sql = f"INSERT INTO {table_name} VALUES (Null, {sql_params})"
            conn.executemany(sql, rows_list)
        conn.commit()
        conn.close()

    def get_features_dataframe(self, network_id: int) -> pd.DataFrame:
        sql = f"""
            SELECT feature_id, name, env
            FROM features
            WHERE network_id = {network_id}
        """
        conn = self.get_connection()
        df = pd.read_sql(sql, conn)
        df['env'] = df['env'].astype(bool)
        conn.close()
        return df

    def get_interactions_dataframe(self, network_id: int) -> pd.DataFrame:
        sql = f"""
            SELECT interaction_id, initiator, target, i_anchor, t_anchor
            FROM interactions
            WHERE network_id = {network_id}
        """
        conn = self.get_connection()
        df = pd.read_sql(sql, conn)
        conn.close()
        return df

    def get_traits_dataframe(self, network_id: int) -> pd.DataFrame:
        sql = f"""
            SELECT trait_id, traits.name, traits.feature_id
            FROM traits
            JOIN features
            ON traits.feature_id = features.feature_id
            WHERE features.network_id = {network_id}
        """
        conn = self.get_connection()
        df = pd.read_sql(sql, conn)
        conn.close()
        return df

    def get_payoffs_dataframe(self, network_id: int) -> pd.DataFrame:
        sql = f"""
            SELECT payoffs.interaction_id,
                   payoffs.initiator,
                   payoffs.target,
                   initiator_utils,
                   target_utils
            FROM payoffs
            JOIN interactions
            ON payoffs.interaction_id = interactions.interaction_id
            WHERE interactions.network_id = {network_id}
        """
        conn = self.get_connection()
        df = pd.read_sql(sql, conn)
        conn.close()
        return df

    def get_network_dataframes(self, network_id: int) -> Dict[str, pd.DataFrame]:
        nd = {}
        nd['features'] = self.get_features_dataframe(network_id)
        nd['interactions'] = self.get_interactions_dataframe(network_id)
        nd['traits'] = self.get_traits_dataframe(network_id)
        nd['payoffs'] = self.get_payoffs_dataframe(network_id)
        return nd

    def get_spacetime_dataframe(self, world: "World") -> pd.DataFrame:
        w_id = world.world_id
        max_steps = world.controller.max_steps
        sites = [(x,y) for _, x, y in world.grid.coord_iter()]
        sites.append('world')
        rows_list = [
            (w_id, step, str(site))
            for step in range(max_steps +1)
            for site in sites
        ]
        write_sql = """
            INSERT INTO spacetime
            VALUES (Null, ?, ?, ?)
        """
        read_sql = f"""
            SELECT spacetime_id, step_num, site_pos
            FROM spacetime
            WHERE world_id = {w_id}
        """
        conn = self.get_connection()
        conn.executemany(write_sql, rows_list)
        df = pd.read_sql(read_sql, conn)
        conn.close()
        return df

    def get_next_record(self, sql: str, keys: List[str]) -> Union[Dict[str, Any], None]:
        conn = self.get_connection()
        record = conn.execute(sql).fetchone()
        conn.close()
        if record:
            record = dict(zip(keys, record))
        return record

    def get_next_feature(
        self,
        world: "World"
    ) -> Union[Dict[str, Any], None]:
        db_id = world.current_feature_db_id
        df = world.network_dfs['features']
        df = df[df.feature_id>db_id]
        if len(df.index) > 0:
            f_dict = dict(df.iloc[0])
            f_dict['env'] = bool(f_dict['env'])
            return f_dict
        else:
            return None

    def get_next_trait(
        self,
        world: "World",
        feature_id: int,
        trait_name: str,
    ) -> Union[Dict[str, Any], None]:
        df = world.network_dfs['traits']
        df = df[(df.feature_id==feature_id)&(df.name==trait_name)]
        t_dict = dict(df.iloc[0]) if len(df.index) > 0 else None
        return t_dict

    def get_records(self, sql: str, keys: List[str]) -> List[Dict[str, Any]]:
        conn = self.get_connection()
        records = conn.execute(sql).fetchall()
        conn.close()
        return [dict(zip(keys, record)) for record in records]

    def get_feature_interactions(
            self,
            world: "World",
            feature_id: int
        ) -> List[Dict[str, Any]]:
        df = world.network_dfs['interactions']
        df = df[df.initiator==feature_id]
        df = df.rename(columns={'interaction_id': 'db_id'})
        return [dict(row) for i, row in df.iterrows()]

    def get_interaction_payoffs(
        self,
        world: "World",
        interaction_id: int,
        i_traits: List[str],
        t_traits: List[str]
    ) -> List[Dict[str, Any]]:
        p_df = world.network_dfs['payoffs']
        p_df = p_df[p_df.interaction_id==interaction_id]
        p_df = p_df.drop(columns=['interaction_id'])
        t_df = world.network_dfs['traits']
        t_df = t_df[t_df.trait_id.isin(pd.concat([p_df.initiator, p_df.target]))]
        df = p_df.replace(t_df.set_index('trait_id')['name'])
        df = df[(df.initiator.isin(i_traits) & df.target.isin(t_traits))]
        df = df.rename(
            columns={'initiator_utils': 'i_utils', 'target_utils': 't_utils'}
        )
        return [dict(row) for i, row in df.iterrows()]

    def handle_features(
        self,
        world: "World",
        c: sqlite3.Connection,
        f_list: List[Tuple[Any]]
    ) -> None:
        ready = []
        lookup = {}
        for f in f_list:
            env = 1 if f.env else 0
            lookup[f.name] = f
            ready.append((f.model.network_id, f.name, env))
        c.executemany("INSERT INTO features VALUES (Null, ?, ?, ?)", ready)
        inserted = c.execute(
            f"""
                SELECT feature_id, name
                FROM features
                ORDER BY feature_id DESC
                LIMIT {len(f_list)}
            """
        ).fetchall()
        world.current_feature_db_id = max([i for i,n in inserted])
        for i, n in inserted:
            lookup[n].db_id = i

    def handle_interactions(
        self,
        world: "World",
        c: sqlite3.Connection,
        i_list: List["Interaction"]
    ) -> None:
        ready = []
        lookup = {}
        for i in i_list:
            i_db = i.initiator.db_id
            t_db = i.target.db_id
            row = (
                world.network_id,
                i_db,
                t_db,
                i.anchors["i"],
                i.anchors["t"]
            )
            ready.append(row)
            lookup[(i_db, t_db)] = i
        c.executemany("INSERT INTO interactions VALUES (Null, ?, ?, ?, ?, ?)", ready)
        inserted = c.execute(
            f"""
                SELECT interaction_id, initiator, target
                FROM interactions
                ORDER BY interaction_id DESC
                LIMIT {len(ready)}
            """
        ).fetchall()
        for d, i, t in inserted:
            interaction = lookup[(i,t)]
            interaction.db_id = d

    def handle_traits(
        self,
        world: "World",
        c: sqlite3.Connection,
        t_dict: Dict[str, List[Tuple[Any]]]
    ) -> None:
        ready = t_dict['r']
        for t, f in t_dict['p']:
            ready.append((t, f.db_id))
        c.executemany("INSERT INTO traits VALUES (Null, ?, ?)", ready)
        inserted = c.execute(
            f"""
                SELECT trait_id, name, feature_id
                FROM traits
                ORDER BY trait_id DESC
                LIMIT {len(ready)}
            """
        ).fetchall()
        lookup = {f.db_id: f for f in world.feature_interactions.nodes}
        for td, n, fd in inserted:
            f = lookup[fd]
            f.trait_ids[n] = td

    def handle_feature_changes(
        self,
        c: sqlite3.Connection,
        f_dict: Dict[str, List[Tuple[Any]]]
    ) -> None:
        ready = f_dict['r']
        for s, f, t in f_dict['p']:
            ready.append((s, f.db_id, t))
        c.executemany("INSERT INTO feature_changes VALUES (Null, ?, ?, ?)", ready)

    def handle_payoffs(
        self,
        c: sqlite3.Connection,
        p_list: List[Tuple[Any]]
    ) -> None:
        ready = []
        for i, i_t, t_t, i_p, t_p in p_list:
            row = (
                i.db_id,
                i.initiator.trait_ids[i_t],
                i.target.trait_ids[t_t],
                i_p,
                t_p
            )
            ready.append(row)
        c.executemany("INSERT INTO payoffs VALUES (Null, ?, ?, ?, ?, ?)", ready)

    def handle_trait_changes(
        self,
        c: sqlite3.Connection,
        t_dict: Dict[str, List[Tuple[Any]]]
    ) -> None:
        ready = t_dict['r']
        for s, (f,v), t in t_dict['p']:
            ready.append((s, f.trait_ids[v], t))
        c.executemany("INSERT INTO trait_changes VALUES (Null, ?, ?, ?)", ready)

    def write_network_rows(self, world: "World"):
        rd = world.network_rows
        conn = self.get_connection()
        if len(rd['features']) > 0:
            self.handle_features(world, conn, rd['features'])
        if len(rd['interactions']) > 0:
            self.handle_interactions(world, conn, rd['interactions'])
        if len(rd['traits']['p']) > 0 or len(rd['traits']['r']) > 0:
            self.handle_traits(world, conn, rd['traits'])
        if len(rd['feature_changes']['p']) > 0 or len(rd['feature_changes']['r']) > 0:
            self.handle_feature_changes(conn, rd['feature_changes'])
        if len(rd['trait_changes']['p']) > 0 or len(rd['trait_changes']['r']) > 0:
            self.handle_trait_changes(conn, rd['trait_changes'])
        if len(rd['payoffs']) > 0:
            self.handle_payoffs(conn, rd['payoffs'])
        conn.commit()
        conn.close()
