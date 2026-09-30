"""Small headless example of the Engine API."""

from engine import Engine

engine = Engine(seed=42)

map_id = engine.create_map(
    4,
    4,
    [
        ["EARTH", "WATER", "EARTH", "EARTH"],
        ["EARTH", "WATER", "EARTH", "EARTH"],
        ["EARTH", "WATER", "WATER", "EARTH"],
        ["EARTH", "SAND", "EARTH", "EARTH"],
    ],
)

goblin_model = engine.create_entity({
    "Type": "NPC",
    "Name": "Goblin",
    "HPMax": 10,
    "HP": 10,
    "AC": 12,
    "XP": 20,
    "Attributes": {
        "STR": 2,
        "DEX": 3,
        "INT": 0,
        "CHA": 0,
        "CON": 1,
    },
})

goblin = engine.spawn_entity(goblin_model, 0, 0)

print("Map:", engine.get_current_map())
print("Goblin:", engine.get_basic_data(goblin))
print("Test:", engine.make_attribute_test(goblin, "DEX"))

# The same model can create another independent instance.
goblin_2 = engine.spawn_entity(goblin_model, 3, 3)
engine.modify_status(goblin, "HP", -5)

print("First HP:", engine.get_basic_field(goblin, "HP"))
print("Second HP:", engine.get_basic_field(goblin_2, "HP"))
