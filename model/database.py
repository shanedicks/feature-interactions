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
                interaction_id  INTEGER NOT NULL
                                    REFERENCES interactions (interaction_id),
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
        interaction_stats_table = """
            CREATE TABLE IF NOT EXISTS interaction_stats (
                interaction_stats_id INTEGER PRIMARY KEY,
                spacetime_id         INTEGER NOT NULL
                                         REFERENCES spacetime (spacetime_id),
                interaction_id       INTEGER NOT NULL
                                         REFERENCES interactions (interaction_id),
                num_interactions     INTEGER NOT NULL,
                initiator_utils      REAL NOT NULL,
                target_utils         REAL NOT NULL
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
            interaction_stats_table,
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

    def get_feature_interactions(
            self,
            world: "World",
            feature_id: int
        ) -> List[Dict[str, Any]]:
        df = world.network_dfs['interactions']
        df = df[df.initiator==feature_id]
        df = df.rename(columns={'interaction_id': 'db_id'})
        return df.to_dict('records')

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
        t_df = t_df.query("trait_id in @relevant_trait_ids")
        t_df.set_index('trait_id', inplace=True) # Prepare for joins
        t_df_names = t_df[['name']]
        # Replace initiator and target trait ids with names
        p_df = p_df.join(t_df_names, on='initiator').rename(columns={'name': 'i_name'})
        p_df = p_df.join(t_df_names, on='target').rename(columns={'name': 't_name'})
        p_df = p_df.drop(columns=['initiator', 'target'])
        p_df = p_df.rename(columns={'i_name': 'initiator', 't_name': 'target'})    
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

def create_indexes(db_path: str) -> None:
    index_queries = [
        "CREATE INDEX IF NOT EXISTS idx_phenotypes_spacetime_id ON phenotypes(spacetime_id);",
        "CREATE INDEX IF NOT EXISTS idx_spacetimes_site_pos ON spacetime(site_pos);",
        "CREATE INDEX IF NOT EXISTS idx_spacetimes_step_num ON spacetime(step_num);",
        "CREATE INDEX IF NOT EXISTS idx_spacetimes_world_id ON spacetime(world_id);"
    ]

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    for query in index_queries:
        cursor.execute(query)

    conn.commit()
    conn.close()

def get_steps_list(db_path: str) -> List[int]:
    sql = """
        SELECT DISTINCT spacetime.step_num
        FROM spacetime
        JOIN phenotypes ON spacetime.spacetime_id = phenotypes.spacetime_id
        ORDER BY spacetime.step_num ASC;
    """
    df = get_df(db_path, sql)
    return df['step_num'].tolist()

def get_world_dict(db_path) -> Dict[int, int]:
    return dict(get_params_df(db_path)[['world_id', 'network_id']].itertuples(index=False))

def get_sites_dict(db_path) -> Dict[int, Dict[Tuple[int, int], List[str]]]:
    sql = """
        SELECT DISTINCT spacetime.site_pos,
               spacetime.world_id,
               traits.name AS trait_name,
               features.name AS feature_name
        FROM spacetime
        LEFT JOIN environment ON environment.spacetime_id = spacetime.spacetime_id
        LEFT JOIN traits ON environment.trait_id = traits.trait_id
        LEFT JOIN features ON traits.feature_id = features.feature_id
        WHERE spacetime.site_pos != 'world'
   """
    df = get_df(db_path, sql)
    df['feature_trait'] = df['feature_name'] + '.' + df['trait_name']
    sites_dict = {}
    grouped = df.groupby(['world_id', 'site_pos'])
    for (world_id, site_pos), group in grouped:
        if world_id not in sites_dict:
            sites_dict[world_id] = {}
        traits_list = group['feature_trait'].dropna().tolist()
        sites_dict[world_id][site_pos] = traits_list

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
    sql = """
    SELECT world_id, step_num, FC.feature_id, name, env, change
    FROM feature_changes AS FC
    JOIN features AS F ON F.feature_id = FC.feature_id
    JOIN spacetime AS S ON S.spacetime_id = FC.spacetime_id;
    """
    return get_df(db_path, sql)

def get_trait_changes_df(db_path: str) -> pd.DataFrame:
    sql = """
    SELECT S.world_id, S.step_num, TC.trait_id, T.name AS trait_name, T.feature_id,
           F.name AS feature_name, F.env, TC.change
    FROM trait_changes AS TC
    JOIN traits AS T ON T.trait_id = TC.trait_id
    JOIN features AS F ON F.feature_id = T.feature_id
    JOIN spacetime AS S ON S.spacetime_id = TC.spacetime_id;
    """
    return get_df(db_path, sql)

def get_interactions_df(
        db_path: str,
    ) -> pd.DataFrame:
    sql = """
    SELECT interaction_id,
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

def get_local_features_df(db_path: str, shadow: bool, world_id: int, site: str) -> pd.DataFrame:
    # Fetch the phenotypes DataFrame for the specified parameters, but for all steps
    phenotypes_df = get_phenotypes_df(
        db_path=db_path,
        shadow=shadow,
        sites=site,
        worlds=world_id
    )
    phenotypes_df['phenotype'] = phenotypes_df['phenotype'].str.split(':')
    expanded = phenotypes_df.explode('phenotype')

    expanded[['feature', 'trait']] = expanded['phenotype'].str.extract(r'(?P<feature>[^.]+)\.(?P<trait>[^.]+)')
    aggregated = expanded.groupby(['feature', 'trait', 'step_num'], as_index=False)['pop'].sum()

    return aggregated

def get_payoffs_df(db_path: str, network_id: int) -> pd.DataFrame:
    sql = f"""
    SELECT p.interaction_id,
           ife.name AS initiator_feature,
           it.name AS initiator_trait,
           tfe.name AS target_feature,
           tt.name AS target_trait,
           p.initiator_utils AS initiator_payoff,
           p.target_utils AS target_payoff
    FROM payoffs p
    JOIN traits it ON p.initiator = it.trait_id
    JOIN traits tt ON p.target = tt.trait_id
    JOIN interactions i ON p.interaction_id = i.interaction_id
    JOIN features ife ON i.initiator = ife.feature_id
    JOIN features tfe ON i.target = tfe.feature_id
    WHERE i.network_id = {network_id}
    """
    df = get_df(db_path, sql)
    return df

def migrate_broken_tables(db_path):
    conn = get_connection(db_path)
    cursor = conn.cursor()
    # Create temporary table with correct column order
    cursor.execute('''
    CREATE TABLE worlds_temp (
        world_id INTEGER PRIMARY KEY,
        trait_mutate_chance REAL NOT NULL,
        trait_create_chance REAL NOT NULL,
        feature_mutate_chance REAL NOT NULL,
        feature_create_chance REAL NOT NULL,
        feature_gain_chance REAL NOT NULL,
        init_agents INTEGER NOT NULL,
        base_agent_utils REAL NOT NULL,
        base_env_utils REAL NOT NULL,
        total_pop_limit INTEGER NOT NULL,
        pop_cost_exp REAL NOT NULL,
        feature_cost_exp REAL NOT NULL,
        grid_size INTEGER NOT NULL,
        repr_multi INTEGER NOT NULL,
        mortality REAL NOT NULL,
        move_chance REAL NOT NULL,
        snap_interval INTEGER NOT NULL,
        feature_timeout INTEGER NOT NULL,
        trait_timeout INTEGER NOT NULL,
        target_sample INTEGER NOT NULL,
        network_id INTEGER NOT NULL
    )
    ''')
    # Copy data from the original table to the temporary table
    cursor.execute('''
    INSERT INTO worlds_temp (
        world_id, trait_mutate_chance, trait_create_chance, 
        feature_mutate_chance, feature_create_chance, feature_gain_chance,
        feature_timeout, trait_timeout,
        init_agents, base_agent_utils, base_env_utils, total_pop_limit, 
        pop_cost_exp, feature_cost_exp, grid_size, repr_multi, 
        mortality, move_chance, snap_interval, 
        target_sample, network_id
    ) 
    SELECT * FROM worlds
    ''')
    # Drop the original table
    cursor.execute('DROP TABLE worlds')
    # Rename the temporary table
    cursor.execute('ALTER TABLE worlds_temp RENAME TO worlds')
    # Increment step_num
    cursor.execute('UPDATE spacetime SET step_num = step_num + 1')
    conn.commit()
    conn.close()
