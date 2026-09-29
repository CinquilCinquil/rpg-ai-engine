import unittest

from engine import Engine, EngineError


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = Engine(seed=1)
        self.map_id = self.engine.create_map(
            3,
            3,
            [
                ["EARTH", "EARTH", "EARTH"],
                ["EARTH", "WATER", "EARTH"],
                ["EARTH", "EARTH", "SAND"],
            ],
        )

        self.goblin_model = self.engine.create_entity({
            "Type": "NPC",
            "Name": "Goblin",
            "HPMax": 10,
            "HP": 10,
            "AC": 12,
            "XP": 20,
            "Attributes": {"STR": 2, "DEX": 3, "INT": 0, "CHA": 0, "CON": 1},
        })

    def test_dice(self):
        self.assertTrue(1 <= self.engine.roll_dice(20) <= 20)
        self.assertEqual(self.engine.evaluate_expression("1 + 1"), 2)
        self.assertEqual(self.engine.evaluate_expression("2 * (5 + 2)"), 14)

    def test_manhattan_distance(self):
        self.assertEqual(self.engine.calculate_distance(1, 2, 4, 6), 7)

    def test_instance_state_is_independent(self):
        a = self.engine.spawn_entity(self.goblin_model, 0, 0)
        b = self.engine.spawn_entity(self.goblin_model, 2, 0)

        self.engine.modify_status(a, "HP", -5)

        self.assertEqual(self.engine.get_basic_field(a, "HP"), 5)
        self.assertEqual(self.engine.get_basic_field(b, "HP"), 10)

    def test_water_is_blocked_by_default(self):
        with self.assertRaises(EngineError):
            self.engine.spawn_entity(self.goblin_model, 1, 1)

    def test_inventory_uses_instances(self):
        entity = self.engine.spawn_entity(self.goblin_model, 0, 0)
        item_model = self.engine.create_item_or_ability({
            "Type": "ITEM",
            "Name": "Dagger",
            "Rarity": 1,
            "Functionality": "TARGET_EFFECT",
            "Uses": None,
            "Parameters": {
                "TargetAttribute": "HP",
                "Modifier": "-1d4",
            },
        })

        item_a = self.engine.add_item_to_inventory(entity, item_model)
        item_b = self.engine.add_item_to_inventory(entity, item_model)

        self.assertNotEqual(item_a, item_b)
        self.assertEqual(len(self.engine.list_inventory_items(entity)), 2)

    def test_condition_interaction(self):
        entity = self.engine.spawn_entity(self.goblin_model, 0, 0)
        self.engine.apply_condition(entity, "OnFire")
        result = self.engine.apply_condition(entity, "Frozen")

        self.assertEqual(result["Interaction"], "Frozen+OnFire removed both conditions.")
        self.assertEqual(self.engine.list_conditions(entity), [])

    def test_structured_command_errors(self):
        response = self.engine.command("set_current_map", map_id=999)
        self.assertFalse(response["Success"])
        self.assertEqual(response["Error"]["Code"], "MAP_NOT_FOUND")


if __name__ == "__main__":
    unittest.main()
