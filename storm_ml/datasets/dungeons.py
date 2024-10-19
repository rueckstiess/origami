import random
from typing import Tuple

colors = ["red", "blue", "green", "yellow", "purple", "orange", "black", "white", "brown", "gray"]
treasures = [
    "diamonds",
    "gold",
    "artifacts",
    "spellbooks",
    "gemstones",
    "weapons",
    "scrolls",
    "potions",
    "relics",
    "coins",
]
monsters = ["goblin", "orc", "troll", "dragon", "wolf"]


def generate_data(
    num_instances: int,
    num_doors_range: Tuple[int, int] = (5, 10),
    num_colors: int = 3,
    num_treasures: int = 5,
    with_monsters: bool = False,
    shuffle_doors: bool = False,
):
    """
    Generates a dungeon-themed synthetic dataset for supervised learning purposes.

    Each instance constains a corridor array with multiple doors. Each room has a door number
    and contains multiple treasure chests with different-colored keys. All but one of the treasures
    are fake though.

    The goal is to find the correct room number and key color in each dungeon and return the
    only non-fake treasure. For example, the following dictionary represents a puzzle instance:


        {
            "door": 0,                              // clue which door is the correct one
            "key_color": "blue",                    // clue which key is the correct one
            "corridor": [
                {
                    "monsters": ["troll", "wolf"],  // optional monsters in front of the door
                    "door_no": 0,                   // door number in the corridor
                    "red_key": "gemstones",         // different keys return different treasures,
                    "blue_key": "spells",           // but only one is real, the others are fake
                    "green_key": "artifacts"
                },
                // ... more doors ...
            ],
            "treasure": "spells"                    // correct treasure (target label)
        }

    The clues are given at the top-level of the object with keys "door", "key_color". The
    correct answer for this instance is "artifacts", because the correct door is 0 and the
    correct key color is "blue".


    Parameters
    ----------
    num_instances : int
        The number of puzzle instances to generate.
    num_doors_range : Tuple[int, int], optional
        The range of the number of doors that each puzzle instance should have.
        Defaults to (5, 10).
    num_colors : int, optional
        The number of colors that each puzzle instance should have. Defaults to 5.
    num_treasures: int, optional
        The number of treasures that each puzzle instance should have. Defaults to 5.
    with_monsters : bool, optional
        Whether to include 0-2 monsters in the puzzle instances. Defaults to False.

    Returns
    -------
    data : List[Dict]
        A list of dictionaries, where each dictionary represents a puzzle instance.
        The dictionary contains the door number, key color, corridor, and treasure.
    """

    def generate_instance(num_doors: int, num_colors: int):
        # generate a corridor of rooms
        corridor = []
        for i in range(num_doors):
            # create random booleans for each color key
            keys = {f"{c}_key": random.choice(treasures[:num_treasures]) for c in colors[:num_colors]}

            # optionally add monsters
            if with_monsters:
                monsters_in_room = random.sample(monsters, random.randint(0, 2))
                if monsters_in_room:
                    door = {"monsters": monsters_in_room, "door_no": i, **keys}
                else:
                    door = {"door_no": i, **keys}
            else:
                door = {"door_no": i, **keys}

            corridor.append(door)

        # door is integer between 0 and num_doors
        door = random.randint(0, num_doors - 1)

        # key_color is one of the color keys
        key_color = random.choice(colors[:num_colors])

        # get the correct treasure with the right key_color in the right door
        treasure = corridor[door][key_color + "_key"]

        # shuffle the doors optionally
        if shuffle_doors:
            random.shuffle(corridor)

        return {"door": door, "key_color": key_color, "corridor": corridor, "treasure": treasure}

    data = []
    for _ in range(num_instances):
        num_doors = random.randint(num_doors_range[0], num_doors_range[1])
        data.append(generate_instance(num_doors, num_colors))
    return data
