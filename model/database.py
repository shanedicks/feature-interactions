import sqlite3
import pandas as pd
from typing import Any, Dict, List, Tuple, Union, Optional
from datetime import datetime

class Manager():

    def __init__(
        self,
        path_to_db: str,
        db_name: str,
    ) -> None:
        self.path_to_db = path_to_db
        self.db_name = db_name
        self.db_string = f"{path_to_db}/{db_name}"

    def get_connection(self):
        conn = sqlite3.connect(self.db_string, timeout=300)
        return conn

    def inspect_db(self) -> sqlite3.Cursor:
        conn = self.get_connection()
        c = conn.cursor()
        results = c.execute("SELECT * FROM sqlite_master")
        return results

    def initialize_db(self) -> None:
        conn = self.get_connection()
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")

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
                num_features  INTEGER NOT NULL,
                agent_int     INTEGER NOT NULL,
                env_int       INTEGER NOT NULL
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
            print(f"Writing {table_name} {datetime.now()}")
            sql_params = ",".join(['?'] * len(rows_list[0]))
            if table_name in ['features', 'interactions', 'traits', 'spacetime']:
                sql = f"INSERT INTO {table_name} VALUES ({sql_params});"
            else:
                sql = f"INSERT INTO {table_name} VALUES (Null, {sql_params});"
            conn.executemany(sql, rows_list)
            conn.commit()
            print(datetime.now())
        conn.close()

    def get_features_dataframe(self, conn: sqlite3.Connection, network_id: int) -> pd.DataFrame:
        sql = f"""
            SELECT feature_id, name, env
            FROM features
            WHERE network_id = {network_id};
        """
        df = pd.read_sql(sql, conn)
        df['env'] = df['env'].astype(bool)
        return df

    def get_interactions_dataframe(self, conn: sqlite3.Connection, network_id: int) -> pd.DataFrame:
        sql = f"""
            SELECT interaction_id, initiator, target, i_anchor, t_anchor
            FROM interactions
            WHERE network_id = {network_id};
        """
        df = pd.read_sql(sql, conn)
        return df

    def get_traits_dataframe(self, conn: sqlite3.Connection, network_id: int) -> pd.DataFrame:
        sql = f"""
            SELECT trait_id, traits.name, traits.feature_id
            FROM traits
            JOIN features
            ON traits.feature_id = features.feature_id
            WHERE features.network_id = {network_id};
        """
        df = pd.read_sql(sql, conn)
        return df

    def get_payoffs_dataframe(self, conn: sqlite3.Connection, network_id: int) -> pd.DataFrame:
        sql = f"""
            SELECT payoffs.interaction_id,
                   payoffs.initiator,
                   payoffs.target,
                   initiator_utils,
                   target_utils
            FROM payoffs
            JOIN interactions
            ON payoffs.interaction_id = interactions.interaction_id
            WHERE interactions.network_id = {network_id};
        """
        df = pd.read_sql(sql, conn)
        return df

    def get_network_dataframes(self, network_id: int) -> Dict[str, pd.DataFrame]:
        nd = {}
        conn = self.get_connection()
        nd['features'] = self.get_features_dataframe(conn, network_id)
        nd['interactions'] = self.get_interactions_dataframe(conn, network_id)
        nd['traits'] = self.get_traits_dataframe(conn, network_id)
        nd['payoffs'] = self.get_payoffs_dataframe(conn, network_id)
        conn.close()
        return nd

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
        # Get stored interaction payoffs
        p_df = world.network_dfs['payoffs']
        p_df = p_df.query('interaction_id == @interaction_id')
        p_df = p_df.drop(columns=['interaction_id'])
        # Get names for relevant traits
        t_df = world.network_dfs['traits']
        relevant_trait_ids = pd.concat([p_df.initiator, p_df.target]).unique()
        t_df = t_df[t_df.trait_id.isin(relevant_trait_ids)]
        t_df.set_index('trait_id', inplace=True) # Prepare for joins
        t_df_names = t_df[['name']]
        # Replace initiator and target trait ids with names
        p_df = p_df.join(t_df_names, on='initiator').rename(columns={'name': 'initiator'})
        p_df = p_df.join(t_df_names, on='target').rename(columns={'name': 'target'})
        # Filter by i_traits and t_traits parameters
        p_df = p_df.query('initiator in @i_traits and target in @t_traits')
        p_df = p_df.rename(columns={'initiator_utils': 'i_utils', 'target_utils': 't_utils'})
        return p_df.to_dict('records')

#Database Output
def get_connection(db_path: str) -> sqlite3.Connection:
    return sqlite3.connect(db_path)

def get_df(db_path: str, sql: str, dtype: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    conn = get_connection(db_path)
    if dtype is None:
        df = pd.read_sql(sql, conn)
    else:
        df = pd.read_sql_query(sql, conn, dtype=dtype)
    conn.close()
    return df

def get_world_dict(db_path) -> Dict[int, int]:
    return dict(get_params_df(db_path)[['world_id', 'network_id']].itertuples(index=False))

def get_sites_dict(db_path) -> Dict[int, Dict[Tuple[int, int], List[str]]]:
    sql = """
        SELECT 
            spacetime.site_pos,
            spacetime.world_id,
            traits.name AS trait_name,
            features.name AS feature_name
        FROM 
            environment
        JOIN 
            spacetime ON environment.spacetime_id = spacetime.spacetime_id
        JOIN 
            traits ON environment.trait_id = traits.trait_id
        JOIN 
            features ON traits.feature_id = features.feature_id
        GROUP BY
            spacetime.site_pos, spacetime.world_id, traits.name, features.name;
    """

    df = get_df(db_path, sql)
    sites_dict = {}

    for index, row in df.iterrows():
        world = row['world_id']
        site = row['site_pos']
        feature_trait = f"{row['feature_name']}.{row['trait_name']}"

        if world not in sites_dict:
            sites_dict[world] = {}

        if site not in sites_dict[world]:
            sites_dict[world][site] = []

        sites_dict[world][site].append(feature_trait)
    return sites_dict

def get_params_df(db_path: str) -> pd.DataFrame:
    columns = [
        "world_id",
        "worlds.network_id",
        "trait_mutate_chance",
        "trait_create_chance",
        "feature_mutate_chance",
        "feature_create_chance",
        "feature_gain_chance",
        "init_agents",
        "base_agent_utils",
        "base_env_utils",
        "total_pop_limit",
        "pop_cost_exp",
        "feature_cost_exp",
        "grid_size",
        "repr_multi",
        "mortality",
        "move_chance",
        "snap_interval",
        "feature_timeout",
        "trait_timeout",
        "target_sample",
        "init_env_features",
        "init_agent_features",
        "max_interactions",
        "trait_payoff_mod",
        "anchor_bias",
        "payoff_bias",
    ]
    sql = f"""SELECT {', '.join(columns)}
              FROM worlds
              JOIN networks
              ON worlds.network_id = networks.network_id;"""
    return get_df(db_path, sql)

def get_model_vars_df(db_path: str) -> pd.DataFrame:
    sql = f"""SELECT *
              FROM model_vars
              JOIN spacetime
              ON model_vars.spacetime_id = spacetime.spacetime_id;"""
    return get_df(db_path, sql)

def get_phenotypes_df(
    db_path: str,
    shadow: bool,
    sites: [Union[Tuple[int], int]] = None,
    worlds: [Union[Tuple[int], int]] = None,
    phenotypes: Union[Tuple[str], str] = None,
    steps: Union[Tuple[int], int] = None
    ) -> pd.DataFrame:
    arg_names = [
        (worlds, 'worlds'),
        (phenotypes, 'phenotypes'),
        (steps, 'steps'),
        (sites, 'sites')
    ]
    for arg, name in arg_names:
        if isinstance(arg, list):
            raise TypeError(f"A list was passed to '{name}'. Please pass a tuple or a single value instead.")

    columns = [
        "phenotype",
        "SUM(pop) as pop",
        "world_id",
        "step_num"
    ]
    dtype = {
        'phenotype': 'category',
    }
    conditions = [f"shadow = {int(shadow)}"]
    groupby = " GROUP BY world_id, step_num, phenotype"
    if sites is not None:
        columns[1] = "pop"
        columns.append("site_pos")
        dtype['site_pos'] = 'category'
        groupby = ""
        if isinstance(sites, tuple):
            conditions.append(f"site_pos in {sites}")
        else:
            conditions.append(f"site_pos = '{sites}'")
    if worlds is not None:
        if isinstance(worlds, tuple):
            conditions.append(f"world_id in {tuple(worlds)}")
        else:
            conditions.append(f"world_id = {worlds}")
    if phenotypes is not None:
        if isinstance(phenotypes, tuple):
            conditions.append(f"phenotype in {phenotypes}")
        else:
            conditions.append(f"phenotype = '{phenotypes}'")
    if steps is not None:
        if isinstance(steps, tuple):
            conditions.append(f"step_num in {steps}")
        else:
            conditions.append(f"step_num = '{steps}'")
    where = " AND ".join(conditions)
    sql = f"""SELECT {', '.join(columns)}
              FROM phenotypes
              JOIN spacetime
              ON phenotypes.spacetime_id = spacetime.spacetime_id
              WHERE {where}{groupby};"""
    df = get_df(db_path, sql, dtype=dtype)
    df['role'] = pd.Categorical(df['phenotype'].str.replace('([.a-z])', '', regex=True))
    return df

def get_feature_changes_df(
        db_path: str,
    ) -> pd.DataFrame:
    sql = """SELECT world_id, step_num, FC.feature_id, name, env, change
             FROM feature_changes AS FC
             JOIN features AS F ON F.feature_id = FC.feature_id
             JOIN spacetime AS S ON S.spacetime_id = FC.spacetime_id;
    """
    return get_df(db_path, sql)

def get_interactions_df(
        db_path: str,
    ) -> pd.DataFrame:
    sql = """SELECT interaction_id,
                    interactions.network_id,
                    I.name AS initiator,
                    T.name AS target,
                    i_anchor,
                    t_anchor
             FROM interactions
             JOIN features AS I ON I.feature_id = interactions.initiator
             JOIN features AS T ON T.feature_id = interactions.target
    """
    return get_df(db_path, sql)
