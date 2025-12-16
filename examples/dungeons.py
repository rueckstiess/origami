"""Dungeons synthetic dataset from ORIGAMI paper.

A dungeon-themed puzzle dataset for testing supervised learning on JSON data.
Each puzzle contains clues (door number, key color) that must be followed
to find the correct treasure.
"""

import random
from collections import OrderedDict

COLORS = ["red", "blue", "green", "yellow", "purple", "orange", "black", "white", "brown", "gray"]
TREASURES = [
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
MONSTERS = ["goblin", "orc", "troll", "dragon", "wolf"]


def generate_data(
    num_instances: int,
    num_doors_range: tuple[int, int] = (5, 10),
    num_colors: int = 3,
    num_treasures: int = 5,
    with_monsters: bool = False,
    shuffle_rooms: bool = False,
    shuffle_keys: bool = False,
    seed: int | None = None,
) -> list[dict]:
    """Generate dungeon puzzle instances.

    Each instance contains a corridor array with multiple doors. Each room has
    a door number and contains multiple treasure chests with different-colored
    keys. All but one of the treasures are fake.

    The goal is to find the correct room number and key color in each dungeon
    and return the only non-fake treasure.

    Example instance:
        {
            "door": 0,                              # clue: correct door number
            "key_color": "blue",                    # clue: correct key color
            "corridor": [
                {
                    "monsters": ["troll", "wolf"],  # optional monsters
                    "door_no": 0,                   # door number
                    "red_key": "gemstones",         # different keys -> treasures
                    "blue_key": "spells",
                    "green_key": "artifacts"
                },
                ...
            ],
            "treasure": "spells"                    # target label
        }

    Args:
        num_instances: Number of puzzle instances to generate
        num_doors_range: Range of doors per instance (min, max)
        num_colors: Number of key colors per door
        num_treasures: Number of possible treasures
        with_monsters: Include 0-2 monsters per room
        shuffle_rooms: Shuffle corridor order (harder)
        shuffle_keys: Shuffle key order in rooms (harder)
        seed: Random seed for reproducibility

    Returns:
        List of puzzle dictionaries
    """
    if seed is not None:
        random.seed(seed)

    def generate_instance(num_doors: int, num_colors: int) -> dict:
        corridor = []
        for i in range(num_doors):
            # Create random treasure for each color key
            keys = {f"{c}_key": random.choice(TREASURES[:num_treasures]) for c in COLORS[:num_colors]}

            # Optionally shuffle key order
            if shuffle_keys:
                key_items = list(keys.items())
                random.shuffle(key_items)
                keys = OrderedDict(key_items)

            # Build door dict
            if with_monsters:
                monsters_in_room = random.sample(MONSTERS, random.randint(0, 2))
                if monsters_in_room:
                    door = {"monsters": monsters_in_room, "door_no": i, **keys}
                else:
                    door = {"door_no": i, **keys}
            else:
                door = {"door_no": i, **keys}

            corridor.append(door)

        # Select random door and key color as clues
        door = random.randint(0, num_doors - 1)
        key_color = random.choice(COLORS[:num_colors])

        # Get correct treasure
        treasure = corridor[door][key_color + "_key"]

        # Optionally shuffle room order
        if shuffle_rooms:
            random.shuffle(corridor)

        return {
            "door": door,
            "key_color": key_color,
            "corridor": corridor,
            "treasure": treasure,
        }

    data = []
    for _ in range(num_instances):
        num_doors = random.randint(num_doors_range[0], num_doors_range[1])
        data.append(generate_instance(num_doors, num_colors))
    return data


def mask_target(obj: dict) -> dict:
    """Remove the treasure field from an instance (for inference)."""
    return {k: v for k, v in obj.items() if k != "treasure"}


if __name__ == "__main__":
    # Demo: generate and print a few instances
    import json

    data = generate_data(3, seed=42)
    for i, instance in enumerate(data):
        print(f"\n=== Instance {i + 1} ===")
        print(f"Clues: door={instance['door']}, key_color={instance['key_color']}")
        print(f"Answer: {instance['treasure']}")
        print(f"Full JSON:\n{json.dumps(instance, indent=2)}")
