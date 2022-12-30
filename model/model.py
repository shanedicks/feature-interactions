from typing import Any, Dict, Iterator, List, Union
from mesa import Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from agents import Agent, Site
from features import Feature, Interaction
from output import *


class Shadow(Model):

    def __init__(
        self,
        model: "World"
    ) -> None:
        super().__init__()
        self.model = model
        self.random = model.random
        self.feature_interactions = self.model.feature_interactions
        self.base_agent_utils = self.model.base_agent_utils
        self.base_env_utils = self.model.base_env_utils
        self.repr_multi = self.model.repr_multi
        self.mortality = self.model.mortality
        self.move_chance = self.model.move_chance
        self.trait_mutate_chance = self.model.trait_mutate_chance
        self.feature_mutate_chance = self.model.feature_mutate_chance
        self.trait_create_chance = self.model.trait_create_chance
        self.feature_create_chance = self.model.feature_create_chance
        self.feature_gain_chance = self.model.feature_gain_chance
        self.site_pop_limit = self.model.site_pop_limit
        self.pop_cost_exp = self.model.pop_cost_exp
        self.grid = MultiGrid(self.model.grid_size, self.model.grid_size, True)
        self.sites = {}
        self.roles_dict = {}
        for pos, site in self.model.sites.items():
            new_site = Site(
                    model = self,
                    pos = pos,
                    traits = site.traits
                )
            self.sites[pos] = new_site
            for agent in site.agents():
                new_agent = Agent(
                    unique_id = self.next_id(),
                    model = self,
                    shadow = True,
                    utils = agent.utils,
                    traits = agent.traits 
                )
                self.grid.place_agent(new_agent, pos)
                new_agent.site = new_agent.get_site()

    def get_features_list(self, env: bool = False):
        return self.model.get_features_list(env=env)

    def agents(self) -> Iterator['Agent']:
        return self.grid.iter_cell_list_contents([pos for pos in self.sites])

class World(Model):

    def __init__(
        self,
        controller: "Controller",
        world_id,
        network_id,
        # features network variables
        init_env_features: int,
        init_agent_features: int,
        max_feature_interactions: int,
        trait_payoff_mod: float,
        anchor_bias: float,
        payoff_bias: float,
        # mutation variables
        trait_mutate_chance: float,
        trait_create_chance: float,
        feature_mutate_chance: float,
        feature_create_chance: float,
        feature_gain_chance: float,
        feature_timeout: int,
        trait_timeout: int,
        #other variables
        init_agents: int,
        base_agent_utils: float,
        base_env_utils: float,
        total_pop_limit: int,
        pop_cost_exp: int,
        feature_cost_exp: float,
        grid_size: int,
        repr_multi: int,
        mortality: float,
        move_chance: float,
        snap_interval: int,
        target_sample: int,
    ) -> None:
        super().__init__()
        assert trait_payoff_mod <= 1 and trait_payoff_mod >= 0
        assert anchor_bias <= 1 and anchor_bias >= -1
        self.controller = controller
        self.db = controller.db_manager
        self.world_id = world_id
        self.network_id = network_id
        self.base_agent_utils = base_agent_utils
        self.base_env_utils = base_env_utils
        self.max_feature_interactions = max_feature_interactions
        self.trait_payoff_mod = trait_payoff_mod
        self.anchor_bias = anchor_bias
        self.payoff_bias = payoff_bias
        self.trait_mutate_chance = trait_mutate_chance
        self.feature_mutate_chance = feature_mutate_chance
        self.trait_create_chance = trait_create_chance
        self.feature_create_chance = feature_create_chance
        self.feature_gain_chance = feature_gain_chance
        self.grid_size = grid_size
        self.repr_multi = repr_multi
        self.mortality = mortality
        self.move_chance = move_chance
        self.site_pop_limit = total_pop_limit / (grid_size ** 2)
        self.pop_cost_exp = pop_cost_exp
        self.feature_cost_exp = feature_cost_exp
        self.grid = MultiGrid(grid_size, grid_size, True)
        self.schedule = RandomActivation(self)
        self.cached_payoffs = {}
        self.roles_dict = {}
        self.site_roles_dict = {}
        self.sites = {}
        self.feature_timeout = feature_timeout
        self.trait_timeout = trait_timeout
        self.target_sample = target_sample
        self.spacetime_df = self.db.get_spacetime_dataframe(self)
        self.spacetime_dict = self.get_spacetime_dict()
        self.network_dfs = self.db.get_network_dataframes(self.network_id)
        self.db_ids = self.get_db_ids_dict()
        self.db_rows = self.get_db_rows_dict()
        self.get_or_create_init_features_network(
            init_env_features,
            init_agent_features
        )
        for _, x, y in self.grid.coord_iter():
            pos = (x,y)
            site = Site(model = self, pos = pos)
            self.sites[pos] = site
            self.site_roles_dict[pos] = {}
        t_list = list(range(1, init_agent_features+1))
        for i in range(init_agents):
            num_traits = self.random.choices(t_list, t_list[::-1])[0]
            agent = self.create_agent(num_traits = num_traits)
            self.schedule.add(agent)
            self.grid.place_agent(agent, agent.pos)
        for agent in self.schedule.agents:
            agent.site = agent.get_site()
        self.shadow = Shadow(model=self)
        self.running = True
        print("Environment -------------------")
        env_report(self)
        print("Roles Distribution ------------")
        print(role_dist(self))
        print("Interaction Report ------------")
        interaction_report(self)

    def next_feature_id(self) -> int:
        self.current_feature_id += 1
        return self.current_feature_id

    def next_db_id(self, table_name: str):
        db_id = self.db_ids[table_name]
        self.db_ids[table_name] += self.controller.total_networks
        return db_id

    def get_features_list(self, env: bool = False) -> List[Feature]:
        return [f for f in self.feature_interactions.nodes if f.env is env]

    def get_or_create_init_features_network(
        self,
        num_env: int,
        num_agent:int
    ) -> None:
        self.current_feature_id = 0
        self.current_feature_db_id = 0
        self.feature_interactions = nx.DiGraph()
        for i in range(num_env):
            self.next_feature(env = True)
        for i in range(num_agent):
            self.next_feature()
        print(f"Env Features {self.get_features_list(env=True)}")
        print(f"Agent Features {self.get_features_list()}")

    def next_feature(self, env: bool = False) -> Feature:
        restored = self.db.get_next_feature(self)
        if restored:
            feature = self.restore_feature(restored)
            self.next_db_id('features')
        else:
            feature = self.create_feature(env=env)
        feature_changes_row = (
            self.spacetime_dict["world"],
            feature.db_id,
            "added"
        )
        self.db_rows['feature_changes'].append(feature_changes_row)
        return feature

    def restore_feature(self, feature_dict: Dict[str, Any]):
        feature_db_id = int(feature_dict['feature_id'])
        self.current_feature_db_id = feature_db_id
        feature_id = self.next_feature_id()
        feature = Feature(
            feature_id = feature_id,
            db_id = feature_db_id,
            model = self,
            env = feature_dict['env']
        )
        self.feature_interactions.add_node(feature)
        print(f"restored feature {feature.db_id} {feature}")
        self.restore_interactions(feature)
        return feature

    def create_feature(self, env: bool = False) -> Feature:
        feature_id = self.next_feature_id()
        feature = Feature(
            feature_id = feature_id,
            model = self,
            env = env
        )
        self.feature_interactions.add_node(feature)
        if feature.env is False:
            num_features = self.feature_interactions.number_of_nodes()
            max_ints = min(self.max_feature_interactions, num_features)
            num_ints = self.random.randrange(1, max_ints)
            for i in range(num_ints):
                self.create_interaction(feature)
        print(f"New feature {feature.db_id} {feature}")
        return feature

    def restore_interactions(self, initiator: Feature) -> None:
        interactions = self.db.get_feature_interactions(self, initiator.db_id)
        for i in interactions:
            target = get_feature_by_id(self, i['target'])
            if target:
                interaction = Interaction(
                    model = self,
                    initiator = initiator,
                    target = target,
                    db_id = int(i['db_id']),
                    restored = True,
                    anchors = {"i": i['i_anchor'], "t": i['t_anchor']}
                )
                self.feature_interactions.add_edge(
                    initiator, target, interaction = interaction
                )
                print(f"restored interaction {interaction.db_id} {interaction}")
                affected_roles = [
                    role for role in self.roles_dict.values()
                    if any(f in role.features for f in [initiator, target])
                ]
                for role in affected_roles:
                    role.interactions = role.get_interactions()

    def create_interaction(self, initiator: Feature) -> None:
        extant_targets = list(self.feature_interactions.neighbors(initiator))
        target_choices = [
            x for x in self.feature_interactions.nodes if x not in extant_targets
        ]
        if len(target_choices) > 0:
            target = self.random.choice(target_choices)
            interaction = Interaction(
                model = self,
                initiator = initiator,
                target = target,
            )
            self.feature_interactions.add_edge(
                initiator, target, interaction = interaction
            )
            print(f"new interaction {interaction.db_id} {interaction}")
            affected_roles = [
                role for role in self.roles_dict.values()
                if any(f in role.features for f in [initiator, target])
            ]
            for role in affected_roles:
                role.interactions = role.get_interactions()

    def remove_feature(self, feature: Feature) -> None:
        print(f"Removing feature {feature.db_id} {feature}")
        feature_changes_row = (
            self.spacetime_dict["world"],
            feature.db_id,
            "removed"
        )
        self.db_rows['feature_changes'].append(feature_changes_row)
        in_edges = feature.in_edges()
        out_edges = feature.out_edges()
        self.feature_interactions.remove_node(feature)
        affected_features = [x.initiator for x in in_edges]
        affected_features.extend([x.target for x in out_edges])
        affected_roles = [
            role for role in self.roles_dict.values()
            if any(f in role.features for f in affected_features)
        ]
        roles_to_remove = [
            role for role in self.roles_dict.values()
            if feature in role.features
        ]
        sd = self.sites
        cache = self.cached_payoffs
        for role in roles_to_remove:
            pl = [p for s in sd for p in role.types[s].keys()]
            for phenotype in pl:
                keys = [k for k, v in cache.items() if v == phenotype]
                if phenotype in cache:
                    del cache[phenotype]
                for k in keys:
                    del cache[k][phenotype]
            del self.roles_dict[role.features]
        for role in affected_roles:
            role.update()

    def prune_features(self):
        for feature in self.get_features_list():
            feature.check_empty()
            feature.prune_traits()
        pruneable = [
            f for f in self.get_features_list()
            if f.empty_steps >= self.feature_timeout
        ]
        for feature in pruneable:
            self.remove_feature(feature)

    def create_agent(self, num_traits: int, utils:float = None) -> Agent:
        if utils is None:
            utils = self.base_agent_utils
        traits = {}
        agent_features = self.get_features_list()
        features = self.random.sample(agent_features, num_traits)
        for feature in features:
            value = self.random.choice(feature.values)
            traits[feature] = value
        x = self.random.randrange(self.grid_size)
        y = self.random.randrange(self.grid_size)
        agent = Agent(
            unique_id = self.next_id(),
            model = self,
            utils = utils,
            traits = traits,
        )
        agent.pos = (x,y)
        return agent

    def verify_shadow(self):
        shadow = self.shadow
        for pos in shadow.sites:
            shadow_site = shadow.sites[pos]
            live = self.sites[pos]
            assert shadow_site.died == live.died
            assert shadow_site.moved_in == live.moved_in
            assert shadow_site.moved_out == live.moved_out
            assert shadow_site.born == live.born

    def get_spacetime_dict(self) -> Dict[Union[str,tuple], int]:
        step = self.schedule.time
        sdf = self.spacetime_df
        spacetimes = sdf[sdf.step_num==step].itertuples()
        spacetime_dict = {}
        for s in spacetimes:
            if s.site_pos == "world":
                spacetime_dict["world"] = s.spacetime_id
            else:
                spacetime_dict[eval(s.site_pos)] = s.spacetime_id
        return spacetime_dict

    def get_db_ids_dict(self):
        db_ids = {}
        for table in ['features', 'interactions', 'traits']:
            df = self.network_dfs[table]
            db_ids[table] = self.network_id
            db_ids[table] += self.controller.total_networks * df.shape[0]

        return db_ids


    def get_db_rows_dict(self) -> Dict[str, List[Tuple[Any]]]:
        return {
            'features': [],
            'interactions': [],
            'traits': [],
            'feature_changes': [],
            'trait_changes': [],
            'payoffs': [],
            'model_vars': [],
            'phenotypes': [],
            'demographics': [],
            'environment': []
        }

    def database_update(self) -> None:
        sd = self.spacetime_dict
        rd = self.db_rows
        get_model_vars_row(self, sd, rd)
        get_phenotypes_rows(self, sd, rd)
        get_phenotypes_rows(self.shadow, sd, rd, True)
        get_sites_rows(self, sd, rd)
        for k in [k for k,v in rd.items() if len(v) == 0]:
            del rd[k]
        self.db.write_rows(rd)
        self.db_rows = self.get_db_rows_dict()


    def step(self):
        self.new = 0
        self.cached = 0
        for site in self.sites.values():
            site.reset()
        for site in self.shadow.sites.values():
            site.reset()
        self.schedule.step()
        self.verify_shadow()
        print(
            "N:{1}/{5}, W:{3}/{4} id={2}, Step:{0}/{6}".format(
                self.schedule.time,
                self.network_id,
                self.world_id,
                self.controller.world_num,
                self.controller.network_worlds,
                self.controller.total_networks,
                self.controller.max_steps
            )
        )
        self.prune_features()
        self.database_update()
        self.spacetime_dict = self.get_spacetime_dict()
        print(role_dist(self))
        if self.schedule.get_agent_count == 0:
            self.running = False
