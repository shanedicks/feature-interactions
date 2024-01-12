from typing import Dict, List, Tuple, FrozenSet
from itertools import product

Payoff = Tuple[float, float]
PayoffDict = Dict[str, Dict[str, Payoff]]


def name_from_number(num: int, lower: bool = True):
    # Set starting ASCII character based on case preference: 'a' (97) for lowercase, 'A' (65) for uppercase.
    if lower:
        char = 97
    else:
        char = 65
    letters = ''
    # Convert the given number to a string of letters (like Excel columns).
    while num:
        mod = (num - 1) % 26  # Find the remainder to determine the current letter.
        letters += chr(mod + char)  # Convert the remainder to a letter and add to the string.
        num = (num - 1) // 26  # Reduce the number for the next iteration.
    return ''.join(reversed(letters))  # Reverse and join the letters to form the final string.


class Feature:

    def __init__(
        self,
        model: "World",
        feature_id: int,
        env: bool,
        name: str = None,
        num_values: int = 5,
        db_id: int = None
    ) -> None:
        # Initialize feature identifiers and basic properties.
        self.feature_id = feature_id
        self.current_value_id = 0
        self.current_value_db_id = 0
        self.model = model
        self.env = env

        # Assign a name to the feature, generate based on feature_id if not provided.
        if name:
            self.name = name
        else:
            self.name = name_from_number(feature_id, lower=False)

        # Assign a database ID and add feature to the model's database if not already present.
        self.db_id = db_id
        if self.db_id is None:
            self.db_id = model.next_db_id('features')
            row = (self.db_id, model.network_id, self.name, int(env))
            self.model.db_rows['features'].append(row)

        # Initialize properties for trait management.
        self.empty_steps = 0
        self.empty_traits = {}
        self.trait_ids = {}
        self.values = []

        # Generate initial traits for the feature.
        for i in range(num_values):
            self.next_trait()

        # Prepare a dictionary to track traits across grid coordinates.
        self.traits_dict = {(x, y): {} for _, x, y in self.model.grid.coord_iter()}

    def new_value(self):
        self.current_value_id += 1
        return name_from_number(self.current_value_id)

    def in_edges(self) -> List['Interaction']:
        fi = self.model.feature_interactions
        return [x[2] for x in fi.in_edges(nbunch=self, data='interaction')]

    def out_edges(self) -> List['Interaction']:
        fi = self.model.feature_interactions
        return [x[2] for x in fi.edges(nbunch=self, data='interaction')]

    def next_trait(self) -> str:
        # Get the next trait value and add it to the feature's list of values.
        value = self.new_value()
        self.values.append(value)

        # Check if the trait is already in the database and restore it if it is.
        # Otherwise, assign a new database ID and add it to the model's database.
        restored = self.model.db.get_next_trait(world=self.model, feature_id=self.db_id, trait_name=value)
        if restored:
            self.trait_ids[value] = int(restored['trait_id'])
            print(f"restored {self.trait_ids[value]} {value} to {self.db_id} {self}")
        else:
            self.trait_ids[value] = self.model.next_db_id('traits')
            row = (self.trait_ids[value], value, self.db_id)
            self.model.db_rows['traits'].append(row)
            print(f"added {self.trait_ids[value]} {value} to {self.db_id} {self}")

        # Record the addition of the new trait in the 'trait_changes' database.
        trait_changes_row = (self.model.spacetime_dict["world"], int(self.trait_ids[value]), "added")
        self.model.db_rows['trait_changes'].append(trait_changes_row)

        # Update payoffs for interactions involving this feature if it's part of any.
        if self in self.model.feature_interactions.nodes():
            self.set_payoffs(value)

        return value

    def set_payoffs(self, value: str) -> None:
        # Retrieve interactions initiated by and targeting this feature.
        initated = self.out_edges()
        targeted = self.in_edges()
        new_payoffs = set()  # Set to store new payoffs for interactions.

        # Update payoffs for each initiated interaction.
        for i in initated:
            i.payoffs[value] = {}  # Initialize payoff structure for new value.
            target_values = set(i.target.values)  # Potential target values for payoffs.

            # Retrieve existing payoffs from the database and update.
            restored = self.model.db.get_interaction_payoffs(
                self.model,
                i.db_id,
                [value],
                i.target.values
            )
            for p in restored:
                i.payoffs[p['initiator']][p['target']] = (p['i_utils'], p['t_utils'])
                target_values.remove(p['target'])

            # Calculate and store new payoffs for remaining targets.
            i_d, t_d = i.initiator.trait_ids, i.target.trait_ids
            for t_value in target_values:
                if (i, value, t_value) not in new_payoffs:
                    new_payoffs.add((i, value, t_value))
                    np = i.new_payoff()
                    i.payoffs[value][t_value] = np
                    row = (i.db_id, int(i_d[value]), int(t_d[t_value]), np[0], np[1])
                    self.model.db_rows['payoffs'].append(row)

        # Repeat similar process for incoming (targeted) interactions.
        for t in targeted:
            initiator_values = set(t.initiator.values)  # Initiator values for payoffs.

            # Retrieve and apply existing payoffs from the database.
            restored = self.model.db.get_interaction_payoffs(
                self.model,
                t.db_id,
                t.initiator.values,
                [value]
            )
            for p in restored:
                t.payoffs[p['initiator']][p['target']] = (p['i_utils'], p['t_utils'])
                initiator_values.remove(p['initiator'])

            # Calculate and store new payoffs for remaining initiators.
            i_d, t_d = t.initiator.trait_ids, t.target.trait_ids
            for i_value in initiator_values:
                if (t, i_value, value) not in new_payoffs:
                    new_payoffs.add((t, i_value, value))
                    np = t.new_payoff()
                    t.payoffs[i_value][value] = np
                    row = (t.db_id, int(i_d[i_value]), int(t_d[value]), np[0], np[1])
                    self.model.db_rows['payoffs'].append(row)

    def remove_trait(self, value: str) -> None:
        # Gather interactions where the feature is involved either as initiator or target.
        initiated = self.out_edges()
        targeted = self.in_edges()

        # Record the removal of the trait in the model's 'trait_changes' database.
        trait_changes_row = (self.model.spacetime_dict["world"], self.trait_ids[value], "removed")
        self.model.db_rows['trait_changes'].append(trait_changes_row)

        # Remove the trait from payoffs in targeted interactions, handling any key errors.
        for t in targeted:
            for i_value in t.initiator.values:
                try:
                    del t.payoffs[i_value][value]
                except KeyError as e:
                    print(e)

        # Remove the trait from payoffs in initiated interactions.
        for i in initiated:
            del i.payoffs[value]

        # Remove the trait from the feature's records, get trait_id before deletion, and update internal trait tracking.
        self.values.remove(value)
        trait_id = self.trait_ids[value]
        del self.empty_traits[value]
        del self.trait_ids[value]

        # Remove the trait from the traits_dict for all model sites.
        for s in self.model.sites:
            if value in self.traits_dict[s]:
                del self.traits_dict[s][value]

        # Log the removal of the trait and check if the feature has any traits left.
        print(f"Trait {trait_id} {value} removed from {self.db_id} {self}")
        if len(self.values) == 0:
            self.model.remove_feature(self)  # Remove the feature from the model if no traits are left.

    def check_empty(self) -> None:
        # Retrieve the sites dictionary from the model.
        sd = self.model.sites

        # Check if the feature is empty (i.e., has no active traits) in all sites.
        # Increment empty_steps if empty, reset to 0 otherwise.
        if sum([v for s in sd for v in self.traits_dict[s].values()]) == 0:
            self.empty_steps += 1
        else:
            self.empty_steps = 0

        # Check and update the status of each trait for the feature.
        td = self.traits_dict
        for v in self.values:
            # If a trait is absent in all sites, increment or initialize its count in empty_traits.
            if sum([td[s][v] for s in sd if v in td[s]]) == 0:
                try:
                    self.empty_traits[v] += 1
                except KeyError:
                    self.empty_traits[v] = 1
            # If the trait is present in any site, reset its count in empty_traits.
            else:
                self.empty_traits[v] = 0

    def prune_traits(self) -> None:
        # Identify traits eligible for pruning based on the model's trait timeout.
        # Traits are considered prunable if they have been empty for a duration equal to or exceeding the trait timeout.
        prunable = [t for t, c in self.empty_traits.items() if c >= self.model.trait_timeout]

        # Remove each prunable trait from the feature.
        for trait in prunable:
            self.remove_trait(trait)

    def cleanup(self):
        # Clear internal structures and references
        self.model = None
        self.values.clear()
        self.empty_traits.clear()
        self.trait_ids.clear()
        self.traits_dict.clear()

    def __repr__(self) -> str:
        return self.name


class Interaction:

    def __init__(
        self,
        model: "World",
        initiator: Feature,
        target: Feature,
        db_id: int = None,
        restored: bool = False,
        payoffs: PayoffDict = None,
        anchors = None,
    ) -> None:
        # Initialize an Interaction instance linking two features in the world model.
        self.model = model
        self.random = model.random  # Random number generator from the model.
        self.initiator = initiator  # Feature initiating the interaction.
        self.target = target  # Feature targeted in the interaction.

        # Set or calculate interaction anchors.
        self.anchors = self.set_anchors() if anchors is None else anchors

        # Assign a database ID to the interaction; create a new record if not existing.
        self.db_id = db_id
        if self.db_id is None:
            self.db_id = model.next_db_id('interactions')
            row = (
                self.db_id,
                self.model.network_id,
                self.initiator.db_id,
                self.target.db_id,
                self.anchors['i'],
                self.anchors['t']
            )
            self.model.db_rows['interactions'].append(row)

        # Initialize or restore payoffs for the interaction.
        if payoffs is None and restored is False:
            self.payoffs = self.construct_payoffs()
        elif restored is True:
            self.payoffs = self.restore_payoffs()
        else:
            self.payoffs = payoffs

    def set_anchors(self):
        # Calculate the anchor points for the interaction based on the model's trait payoff modifier and anchor bias.
        anchor = 1 - self.model.trait_payoff_mod  # Base anchor value.
        mode = self.model.anchor_bias * anchor  # Mode for the triangular distribution.
        # Determine initiator and target anchors using a triangular distribution.
        i_anchor = round(self.random.triangular(-anchor, anchor, mode), 2)
        t_anchor = round(self.random.triangular(-anchor, anchor, mode), 2)
        return {"i": i_anchor, "t": t_anchor}

    def construct_payoffs(self) -> PayoffDict:
        # Initialize a payoff dictionary for each initiator value.
        payoff_dict = {i: {} for i in self.initiator.values}
        i_d, t_d = self.initiator.trait_ids, self.target.trait_ids  # Trait IDs for initiator and target.
        # Calculate payoffs for every combination of initiator and target trait values.
        for i_value in self.initiator.values:
            for t_value in self.target.values:
                np = self.new_payoff()  # Calculate a new payoff.
                payoff_dict[i_value][t_value] = np  # Assign the payoff to the dictionary.
                # Record the new payoff in the model's database.
                row = (self.db_id, int(i_d[i_value]), int(t_d[t_value]), np[0], np[1])
                self.model.db_rows['payoffs'].append(row)
        return payoff_dict

    def restore_payoffs(self) -> PayoffDict:
        # Initialize a payoff dictionary for each initiator value.
        payoff_dict = {i: {} for i in self.initiator.values}
        # Create a set of all possible initiator-target value pairs.
        payoff_set = set(product(self.initiator.values, self.target.values))

        # Retrieve and restore payoffs from the database.
        restored = self.model.db.get_interaction_payoffs(
            self.model,
            self.db_id,
            self.initiator.values,
            self.target.values
        )
        for p in restored:
            # Populate the payoff dictionary with restored values.
            payoff_dict[p['initiator']][p['target']] = (p['i_utils'], p['t_utils'])
            # Remove the restored pair from the set of pairs to be processed.
            payoff_set.remove((p['initiator'], p['target']))

        # For any remaining pairs, calculate new payoffs and add them to the dictionary and database.
        if len(payoff_set) > 0:
            i_d, t_d = self.initiator.trait_ids, self.target.trait_ids
            for i_value, t_value in payoff_set:
                np = self.new_payoff()  # Generate a new payoff.
                payoff_dict[i_value][t_value] = np  # Update the payoff dictionary.
                # Record the new payoff in the model's database.
                row = (self.db_id, int(i_d[i_value]), int(t_d[t_value]), np[0], np[1])
                self.model.db_rows['payoffs'].append(row)
        return payoff_dict

    def new_payoff(self) -> Payoff:
        # Determine the payoff modifier and bias from the model.
        mod = self.model.trait_payoff_mod
        mode = self.model.payoff_bias * mod

        # Calculate each payoff using a triangular distribution around the anchor, ensuring it's within [-1.0, 1.0].
        i = round(self.anchors["i"] + self.random.triangular(-mod, mod, mode), 2)
        assert i <= 1.0 and i >= -1.0
        t = round(self.anchors["t"] + self.random.triangular(-mod, mod, mode), 2)
        assert t <= 1.0 and t >= -1.0
        return (i, t)

    def cleanup(self):
        # Clear internal structures and references
        self.model = None
        self.initiator = None
        self.target = None
        self.payoffs.clear()

    def __repr__(self) -> str:
        return "{0}→{1}".format(
            self.initiator.name, 
            self.target.name
        )


class Role:
    def __init__(
        self,
        model: "Model",
        features: FrozenSet["Feature"],
    ):
        # Initialize the Role with model context and its defining features.
        self.model = model
        self.features = features
        # Generate a unique rolename from the sorted names of the features.
        self.rolename = ":".join(sorted([f.name for f in features]))
        # Determine the role's interactions, target features, and neighbors.
        self.interactions = self.get_interactions()
        self.target_features = self.get_target_features()
        self.neighbors = self.get_neighbors()
        # Initialize a types dictionary for tracking types across model grid coordinates.
        self.types = {(x, y): {} for _, x, y in self.model.grid.coord_iter()}

    def get_target_features(self):
        initiating = self.interactions['initiator']
        return [x.target for x in initiating if not x.target.env]

    def get_interactions(self):
        # Access the feature interactions from the model.
        fi = self.model.feature_interactions
        features = [f for f in self.features]  # List of features associated with this role.

        interactions = {}
        # Gather interactions where the role's features act as initiators.
        interactions['initiator'] = [x[2] for x in fi.edges(nbunch=features, data='interaction')]
        # Gather interactions where the role's features are targets.
        interactions['target'] = [x[2] for x in fi.in_edges(nbunch=features, data='interaction')]

        return interactions  # Return the compiled list of interactions.

    def get_neighbors(self):
        # Retrieve all roles from the model.
        roles = self.model.roles_dict.values()

        neighbors = {}
        # Identify roles that initiate interactions with this role's features.
        neighbors["initiators"] = [
            r for r in roles if any(f in r.features for f in [i.initiator for i in self.interactions['target']])
        ]
        # Identify roles that are targets of interactions initiated by this role's features.
        neighbors["targets"] = [
            r for r in roles if any(f in r.features for f in [i.target for i in self.interactions['initiator']])
        ]

        return neighbors  # Return the identified neighboring roles.


    def update(self):
        # Update the interactions and neighbors dictionaries
        self.interactions = self.get_interactions()
        self.neighbors = self.get_neighbors()

    def cleanup(self):
        # Clear internal structures and references
        self.model = None
        self.features.clear()
        self.interactions.clear()
        self.neighbors.clear()
        self.types.clear()

    def __repr__(self) -> str:
        return self.rolename
