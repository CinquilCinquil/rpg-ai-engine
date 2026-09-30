"""
AI RPG Engine
=============

A deterministic/stateful game engine for an AI-driven tabletop RPG simulator.

The engine deliberately separates:
    - World state
    - Mechanical rules
    - Queries
    - Commands
    - Rendering/UI

Narrative interpretation is NOT performed here. The engine validates and
executes explicit mechanical operations and returns structured data that an
LLM acting as Game Master can consume.

IDs:
    - Database/model objects use positive integers.
    - Runtime instances use 8-character hexadecimal IDs

Coordinates:
    - (0, 0) is the top-left cell.
    - X = column, Y = row.
    - Matrices are indexed as matrix[Y][X].

"""

from __future__ import annotations

import ast
import copy
import operator
import random
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TERRAINS = {"EARTH", "SAND", "WATER"}
STRUCTURES = {"NONE", "HOUSE", "STORE", "STATUE"}
ENTITY_TYPES = {"PLAYER", "NPC", "STATIC"}
ATTRIBUTES = {"STR", "DEX", "INT", "CHA", "CON"}
STATUSES = {"HP", "HPMAX", "AC", "XP"}

FUNCTIONALITIES = {
    "PASSIVE",
    "TARGET_EFFECT",
    "POSITION_EFFECT",
    "SIMPLE_USE",
    "NARRATIVE",
}

CONDITION_DURATION_UNITS = {"ACTIONS", "TURNS", "ROUNDS", "INDEFINITE"}


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class EngineError(Exception):
    """Structured engine error.

    The exception itself is convenient for Python callers. The `to_dict`
    method exposes the same standardized representation intended for an LLM.
    """

    def __init__(self, code: str, message: str, details: Optional[dict] = None):
        self.code = code
        self.message = message
        self.details = details or {}
        super().__init__(f"{code}: {message}")

    def to_dict(self) -> dict:
        result = {
            "Success": False,
            "Error": {
                "Code": self.code,
                "Message": self.message,
            },
        }
        if self.details:
            result["Error"]["Details"] = copy.deepcopy(self.details)
        return result


def success(value: Any = None) -> dict:
    """Create a standardized successful command response."""
    result = {"Success": True}
    if value is not None:
        result["Result"] = value
    return result


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def new_runtime_id() -> str:
    """Return an 8-character runtime ID."""
    return uuid.uuid4().hex[:8]


def deep_copy(value: Any) -> Any:
    return copy.deepcopy(value)


# ---------------------------------------------------------------------------
# Expression evaluator
# ---------------------------------------------------------------------------

_DICE_RE = re.compile(r"(?<![A-Za-z0-9_])(\d+)[dD](\d+)(?![A-Za-z0-9_])")


class SafeExpressionEvaluator:
    """Evaluate arithmetic expressions and XdY dice without eval()."""

    BINARY_OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }

    UNARY_OPS = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    def __init__(self, dice_roller: Callable[[int], int]):
        self.dice_roller = dice_roller

    def replace_dice(self, expression: str) -> str:
        def replacement(match: re.Match) -> str:
            count = int(match.group(1))
            sides = int(match.group(2))
            if count <= 0 or sides <= 0:
                raise EngineError(
                    "INVALID_VALUE",
                    "Dice expressions require positive count and sides.",
                )
            if count > 10000:
                raise EngineError(
                    "INVALID_VALUE",
                    "A dice expression may contain at most 10000 dice.",
                )
            total = sum(self.dice_roller(sides) for _ in range(count))
            return str(total)

        return _DICE_RE.sub(replacement, expression)

    def evaluate(self, expression: str) -> int | float:
        if not isinstance(expression, str) or not expression.strip():
            raise EngineError(
                "INVALID_ARGUMENT",
                "Expression must be a non-empty string.",
            )

        expression = self.replace_dice(expression.strip())

        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise EngineError(
                "INVALID_ARGUMENT",
                f"Invalid expression: {expression}",
            ) from exc

        value = self._eval_node(tree.body)

        if isinstance(value, float) and value.is_integer():
            return int(value)
        return value

    def _eval_node(self, node: ast.AST) -> int | float:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise EngineError("INVALID_ARGUMENT", "Only numeric constants are allowed.")
            return node.value

        if isinstance(node, ast.BinOp) and type(node.op) in self.BINARY_OPS:
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)

            if isinstance(node.op, (ast.Pow,)):
                if abs(right) > 100:
                    raise EngineError("INVALID_VALUE", "Exponent is too large.")

            try:
                return self.BINARY_OPS[type(node.op)](left, right)
            except ZeroDivisionError as exc:
                raise EngineError("INVALID_VALUE", "Division by zero.") from exc

        if isinstance(node, ast.UnaryOp) and type(node.op) in self.UNARY_OPS:
            return self.UNARY_OPS[type(node.op)](self._eval_node(node.operand))

        raise EngineError(
            "INVALID_ARGUMENT",
            "Expression contains an unsupported operation.",
        )


# ---------------------------------------------------------------------------
# Data objects
# ---------------------------------------------------------------------------

@dataclass
class MapState:
    map_id: int
    width: int
    height: int
    terrain: list[list[str]]
    structures: list[list[str]]
    entity_instances: dict[str, "EntityInstance"] = field(default_factory=dict)


@dataclass
class EntityModel:
    entity_id: int
    data: dict


@dataclass
class ItemModel:
    item_id: int
    data: dict


@dataclass
class AbilityModel:
    ability_id : int
    data : dict


@dataclass
class EntityInstance:
    instance_id: str
    model_id: int
    data: dict
    x: int
    y: int
    inventory: dict[str, "ItemInstance"] = field(default_factory=dict)
    conditions: dict[str, "ConditionInstance"] = field(default_factory=dict)
    abilities: list[int] = field(default_factory=list)


@dataclass
class ItemInstance:
    instance_id: str
    model_id: int
    data: dict


@dataclass
class ConditionInstance:
    condition_id: str
    name: str
    duration: Optional[int]
    duration_unit: str
    effect: dict
    rules: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class Engine:
    """Main RPG simulation engine.

    The class exposes the public API while
    keeping internal state in separate model/instance stores.

    A typical LLM integration can call:
        engine.command(...)
    or directly call the documented methods.

    For robust integrations, exceptions should be converted with:
        try:
            ...
        except EngineError as error:
            response = error.to_dict()
    """

    def __init__(self, seed: Optional[int] = None):
        self.random = random.Random(seed)
        self._expression_evaluator = SafeExpressionEvaluator(self.roll_dice)

        self.maps: dict[int, MapState] = {}
        self.entity_models: dict[int, EntityModel] = {}
        self.item_models: dict[int, ItemModel] = {}
        self.ability_models: dict[int, AbilityModel] = {}

        self._next_map_id = 1
        self._next_entity_id = 1
        self._next_item_id = 1

        self.current_map_id: Optional[int] = None

        # Runtime indexes.
        self._instances: dict[str, EntityInstance] = {}
        self._item_instances: dict[str, tuple[str, ItemInstance]] = {}
        self._conditions: dict[str, tuple[str, ConditionInstance]] = {}

    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------

    @staticmethod
    def _require_int(value: Any, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise EngineError("INVALID_ARGUMENT", f"{name} must be an integer.")
        return value

    @staticmethod
    def _validate_matrix(matrix: Any, width: int, height: int, valid_values: set[str], name: str):
        if not isinstance(matrix, list) or len(matrix) != height:
            raise EngineError(
                "INVALID_VALUE",
                f"{name} must contain exactly {height} rows.",
            )
        for y, row in enumerate(matrix):
            if not isinstance(row, list) or len(row) != width:
                raise EngineError(
                    "INVALID_VALUE",
                    f"{name}[{y}] must contain exactly {width} columns.",
                )
            for x, value in enumerate(row):
                if value not in valid_values:
                    raise EngineError(
                        "INVALID_VALUE",
                        f"Invalid {name} value at ({x}, {y}): {value}.",
                    )

    def _require_current_map(self) -> MapState:
        if self.current_map_id is None or self.current_map_id not in self.maps:
            raise EngineError("MAP_NOT_FOUND", "No map is currently selected.")
        return self.maps[self.current_map_id]

    def _require_map(self, map_id: int) -> MapState:
        if map_id not in self.maps:
            raise EngineError("MAP_NOT_FOUND", f"Map {map_id} does not exist.")
        return self.maps[map_id]

    def _require_model(self, entity_id: int) -> EntityModel:
        if entity_id not in self.entity_models:
            raise EngineError("ENTITY_NOT_FOUND", f"Entity model {entity_id} does not exist.")
        return self.entity_models[entity_id]

    def _require_instance(self, instance_id: str) -> EntityInstance:
        if instance_id not in self._instances:
            raise EngineError("ID_NOT_FOUND", f"Entity instance {instance_id} does not exist.")
        return self._instances[instance_id]

    def _require_item_model(self, item_id: int) -> ItemModel:
        if item_id not in self.item_models:
            raise EngineError("ID_NOT_FOUND", f"Item model {item_id} does not exist.")
        return self.item_models[item_id]
    
    def _require_ability_model(self, ability_id: int) -> AbilityModel:
        if ability_id not in self.ability_models:
            raise EngineError("ID_NOT_FOUND", f"Ability model {ability_id} does not exist.")
        return self.ability_models[ability_id]
    
    def _require_item_or_ability_model(self, item_or_ability_id: int) -> ItemModel:
        if item_or_ability_id in self.item_models:
            return self.item_models[item_or_ability_id]
        if item_or_ability_id in self.ability_models:
            return self.ability_models[item_or_ability_id]
        raise EngineError("ID_NOT_FOUND", f"Item or Ability model {item_or_ability_id} does not exist.")

    def _require_position(self, x: int, y: int, map_state: Optional[MapState] = None):
        map_state = map_state or self._require_current_map()
        if not isinstance(x, int) or not isinstance(y, int):
            raise EngineError("INVALID_POSITION", "X and Y must be integers.")
        if not (0 <= x < map_state.width and 0 <= y < map_state.height):
            raise EngineError(
                "INVALID_POSITION",
                f"Position ({x}, {y}) is outside map {map_state.map_id}.",
            )

    # -----------------------------------------------------------------------
    # Dice and expressions
    # -----------------------------------------------------------------------

    def roll_dice(self, n: int) -> int:
        """Roll an N-sided die, uniformly from 1 through N."""
        n = self._require_int(n, "N")
        if n <= 0:
            raise EngineError("INVALID_VALUE", "N must be greater than zero.")
        return self.random.randint(1, n)

    def evaluate_expression(self, expression: str) -> int | float:
        """Evaluate arithmetic and XdY dice expressions."""
        return self._expression_evaluator.evaluate(expression)

    # -----------------------------------------------------------------------
    # Geometry
    # -----------------------------------------------------------------------

    def calculate_distance(self, x: int, y: int, w: int, z: int) -> int:
        for value, name in ((x, "X"), (y, "Y"), (w, "W"), (z, "Z")):
            self._require_int(value, name)
        return abs(x - w) + abs(y - z)

    # -----------------------------------------------------------------------
    # Maps
    # -----------------------------------------------------------------------

    def create_map(self, width: int, height: int, config: list[list[str]]) -> int:
        """ Creates a map with the specified parameters and saves it in the
        map database.

        :param width: The width of the map.
        :param height: The height of the map.
        :param config: The configuration of the map (what each cell contains).

        :return: The map id.
        """
        width = self._require_int(width, "W")
        height = self._require_int(height, "H")
        if width <= 0 or height <= 0:
            raise EngineError("INVALID_VALUE", "Map dimensions must be positive.")

        self._validate_matrix(config, width, height, TERRAINS, "Terrain")

        map_id = self._next_map_id
        self._next_map_id += 1

        terrain = deep_copy(config)
        structures = [["NONE" for _ in range(width)] for _ in range(height)]

        self.maps[map_id] = MapState(
            map_id=map_id,
            width=width,
            height=height,
            terrain=terrain,
            structures=structures,
        )

        if self.current_map_id is None:
            self.current_map_id = map_id

        return map_id

    def populate_map(self, width: int, height: int, config: list[list[str]]) -> dict:
        current = self._require_current_map()
        if width != current.width or height != current.height:
            raise EngineError(
                "INVALID_VALUE",
                "Structure dimensions must match the current map dimensions.",
            )

        self._validate_matrix(config, width, height, STRUCTURES, "Structure")
        current.structures = deep_copy(config)
        return success()

    def list_maps(self) -> list[int]:
        return list(self.maps.keys())

    def get_current_map(self) -> dict:
        current = self._require_current_map()
        return {
            "MapID": current.map_id,
            "Width": current.width,
            "Height": current.height,
            "Terrain": deep_copy(current.terrain),
            "Structures": deep_copy(current.structures),
            "Entities": [
                {
                    "InstanceID": instance.instance_id,
                    "EntityID": instance.model_id,
                    "X": instance.x,
                    "Y": instance.y,
                }
                for instance in current.entity_instances.values()
            ],
        }

    def get_current_map_id(self) -> int:
        return self._require_current_map().map_id

    def set_current_map(self, map_id: int) -> dict:
        self._require_map(map_id)
        self.current_map_id = map_id
        return success(map_id)

    # -----------------------------------------------------------------------
    # Entity database
    # -----------------------------------------------------------------------

    def create_entity(self, config: dict) -> int:
        if not isinstance(config, dict):
            raise EngineError("INVALID_ARGUMENT", "Entity CONFIG must be an object.")

        entity_type = config.get("Type")
        if entity_type not in ENTITY_TYPES:
            raise EngineError(
                "INVALID_VALUE",
                f"Type must be one of {sorted(ENTITY_TYPES)}.",
            )

        if not config.get("Name"):
            raise EngineError("INVALID_VALUE", "Entity must have a Name.")

        data = deep_copy(config)

        if entity_type in {"PLAYER", "NPC"}:
            data.setdefault("HPMax", data.get("HP", 1))
            data.setdefault("HP", data["HPMax"])
            data.setdefault("AC", 10)
            data.setdefault("XP", 0)
            data.setdefault("Attributes", {})
            data["Attributes"] = {
                attribute: int(data["Attributes"].get(attribute, 0))
                for attribute in ATTRIBUTES
            }
            data.setdefault("Proficiencies", {})

        entity_id = self._next_entity_id
        self._next_entity_id += 1
        self.entity_models[entity_id] = EntityModel(entity_id, data)
        return entity_id

    # -----------------------------------------------------------------------
    # Entity instances
    # -----------------------------------------------------------------------

    def spawn_entity(self, entity_id: int, x: int, y: int) -> str:
        model = self._require_model(entity_id)
        current = self._require_current_map()
        self._require_position(x, y, current)

        if not self._terrain_allows_movement(current.terrain[y][x], model.data["Type"]):
            raise EngineError(
                "INVALID_POSITION",
                f"Terrain {current.terrain[y][x]} does not allow this entity.",
            )

        instance_id = new_runtime_id()
        instance = EntityInstance(
            instance_id=instance_id,
            model_id=entity_id,
            data=deep_copy(model.data),
            x=x,
            y=y,
        )
        current.entity_instances[instance_id] = instance
        self._instances[instance_id] = instance
        return instance_id

    def list_entity_models(self) -> list[int]:
        return list(self.entity_models.keys())

    def list_map_entities(self) -> list[str]:
        current = self._require_current_map()
        return list(current.entity_instances.keys())

    def remove_entity_from_map(self, instance_id: str) -> dict:
        instance = self._require_instance(instance_id)
        current = self._require_current_map()
        if instance_id not in current.entity_instances:
            raise EngineError(
                "ID_NOT_FOUND",
                f"Entity instance {instance_id} is not in the current map.",
            )

        for item_instance_id in list(instance.inventory):
            self._item_instances.pop(item_instance_id, None)

        for condition_id in list(instance.conditions):
            self._conditions.pop(condition_id, None)

        current.entity_instances.pop(instance_id)
        self._instances.pop(instance_id)
        return success()

    # -----------------------------------------------------------------------
    # Entity queries
    # -----------------------------------------------------------------------

    def get_entity(self, instance_id: str) -> dict:
        instance = self._require_instance(instance_id)
        return self._entity_snapshot(instance)

    def get_basic_data(self, instance_id: str) -> dict:
        instance = self._require_instance(instance_id)
        data = instance.data
        if data["Type"] == "STATIC":
            return {
                "Type": "STATIC",
                "Name": data["Name"],
                "X": instance.x,
                "Y": instance.y,
            }

        return {
            "Type": data["Type"],
            "Name": data["Name"],
            "HP": data["HP"],
            "HPMax": data["HPMax"],
            "AC": data["AC"],
            "XP": data["XP"],
            "Attributes": deep_copy(data["Attributes"]),
            "Proficiencies": deep_copy(data.get("Proficiencies", {})),
            "Conditions": self.list_conditions(instance_id),
        }

    def get_basic_field(self, instance_id: str, field_name: str) -> Any:
        instance = self._require_instance(instance_id)
        if field_name == "Conditions":
            return self.list_conditions(instance_id)
        if field_name not in instance.data:
            raise EngineError("ID_NOT_FOUND", f"Field {field_name} does not exist.")
        return deep_copy(instance.data[field_name])

    def _entity_snapshot(self, instance: EntityInstance) -> dict:
        snapshot = deep_copy(instance.data)
        snapshot.update({
            "InstanceID": instance.instance_id,
            "ModelID": instance.model_id,
            "X": instance.x,
            "Y": instance.y,
            "Inventory": list(instance.inventory.keys()),
            "Abilities": list(instance.abilities),
            "Conditions": self.list_conditions(instance.instance_id),
        })
        return snapshot

    # -----------------------------------------------------------------------
    # Movement / status / attributes
    # -----------------------------------------------------------------------

    def set_position(self, instance_id: str, x: int, y: int) -> dict:
        instance = self._require_instance(instance_id)
        current = self._require_current_map()

        self._require_position(x, y, current)

        if self._has_blocking_condition(instance):
            raise EngineError(
                "ACTION_UNAVAILABLE",
                "The entity cannot move because of an active condition.",
            )

        if not self._terrain_allows_movement(current.terrain[y][x], instance.data["Type"]):
            raise EngineError(
                "INVALID_POSITION",
                f"Terrain {current.terrain[y][x]} does not permit movement.",
            )

        instance.x = x
        instance.y = y
        return success()

    @staticmethod
    def _terrain_allows_movement(terrain: str, entity_type: str) -> bool:
        # The specification defines terrains but does not define exact movement
        # rules. This explicit mechanical default treats WATER as impassable.
        # Narrative exceptions can be represented by a future explicit rule.
        if terrain == "WATER":
            return False
        return True

    def modify_status(self, instance_id: str, status: str, modifier: int) -> int:
        instance = self._require_instance(instance_id)
        if instance.data["Type"] == "STATIC":
            raise EngineError("ACTION_UNAVAILABLE", "Static entities have no character status.")

        if status not in STATUSES:
            raise EngineError("INVALID_VALUE", f"Invalid status: {status}.")
        if not isinstance(modifier, int):
            raise EngineError("INVALID_ARGUMENT", "MOD must be an integer.")

        old = int(instance.data[status])
        new = old + modifier

        if status == "HPMax":
            new = max(1, new)
            instance.data["HPMax"] = new
            instance.data["HP"] = min(instance.data["HP"], new)
        elif status == "HP":
            new = max(0, min(new, int(instance.data["HPMax"])))
            instance.data["HP"] = new
        elif status == "AC":
            instance.data["AC"] = new
        elif status == "XP":
            instance.data["XP"] = max(0, new)

        return int(instance.data[status])

    def modify_attribute(self, instance_id: str, attribute: str, modifier: int) -> int:
        instance = self._require_instance(instance_id)
        if instance.data["Type"] == "STATIC":
            raise EngineError("ACTION_UNAVAILABLE", "Static entities have no attributes.")
        if attribute not in ATTRIBUTES:
            raise EngineError("INVALID_VALUE", f"Invalid attribute: {attribute}.")
        if not isinstance(modifier, int):
            raise EngineError("INVALID_ARGUMENT", "MOD must be an integer.")

        instance.data["Attributes"][attribute] += modifier
        return instance.data["Attributes"][attribute]

    # -----------------------------------------------------------------------
    # Entity actions
    # -----------------------------------------------------------------------

    def list_entity_actions(self, instance_id: str) -> dict:
        instance = self._require_instance(instance_id)

        actions = {
            "SetPosition": {
                "Args": {"X": "int", "Y": "int"},
                "Return": "void",
            }
        }

        if instance.data["Type"] in {"PLAYER", "NPC"}:
            actions.update({
                "ModifyStatus": {
                    "Args": {"Status": "HP|HPMax|AC|XP", "Modifier": "int"},
                    "Return": "int",
                },
                "ModifyAttribute": {
                    "Args": {"Attribute": "STR|DEX|INT|CHA|CON", "Modifier": "int"},
                    "Return": "int",
                },
                "MakeAttributeTest": {
                    "Args": {"Attribute": "STR|DEX|INT|CHA|CON", "Context": "string?"},
                    "Return": "object",
                },
                "UseItem": {
                    "Args": {"Item": "string", "Args": "object"},
                    "Return": "object",
                },
                "ApplyCondition": {
                    "Args": {"Condition": "object", "Config": "object?"},
                    "Return": "object",
                },
            })

        return actions

    def handle_entity_action(self, instance_id: str, action: str, args: dict) -> Any:
        self._require_instance(instance_id)
        if action not in self.list_entity_actions(instance_id):
            raise EngineError("ACTION_NOT_FOUND", f"Action {action} does not exist for this entity.")

        if not isinstance(args, dict):
            raise EngineError("INVALID_ARGUMENT", "Args must be an object.")

        if action == "SetPosition":
            return self.set_position(instance_id, args.get("X"), args.get("Y"))
        if action == "ModifyStatus":
            return self.modify_status(instance_id, args.get("Status"), args.get("Modifier"))
        if action == "ModifyAttribute":
            return self.modify_attribute(instance_id, args.get("Attribute"), args.get("Modifier"))
        if action == "MakeAttributeTest":
            return self.make_attribute_test(instance_id, args.get("Attribute"), args.get("Context"))
        if action == "UseItem":
            return self.use_item(instance_id, args.get("Item"), args)
        if action == "ApplyCondition":
            return self.apply_condition(instance_id, args.get("Condition"), args.get("Config"))

        raise EngineError("ACTION_NOT_FOUND", f"Unsupported action: {action}.")

    # -----------------------------------------------------------------------
    # Tests
    # -----------------------------------------------------------------------

    def make_attribute_test(self, instance_id: str, attribute: str, context: Optional[str] = None) -> dict:
        instance = self._require_instance(instance_id)
        if instance.data["Type"] == "STATIC":
            raise EngineError("ACTION_UNAVAILABLE", "Static entities cannot make attribute tests.")
        if attribute not in ATTRIBUTES:
            raise EngineError("INVALID_VALUE", f"Invalid attribute: {attribute}.")

        die = self.roll_dice(20)
        attribute_modifier = int(instance.data["Attributes"].get(attribute, 0))
        proficiency_modifier = 0
        proficiency_name = None

        if context:
            for name, proficiency in instance.data.get("Proficiencies", {}).items():
                if not isinstance(proficiency, dict):
                    continue
                if self._proficiency_matches(proficiency, context):
                    modifier = int(proficiency.get("Modifier", 0))
                    if modifier > proficiency_modifier:
                        proficiency_modifier = modifier
                        proficiency_name = name

        result = die + attribute_modifier + proficiency_modifier
        return {
            "Roll": die,
            "Attribute": attribute_modifier,
            "Proficiency": proficiency_modifier,
            "ProficiencyName": proficiency_name,
            "Result": result,
        }

    @staticmethod
    def _proficiency_matches(proficiency: dict, context: str) -> bool:
        keywords = set(re.findall(r"[A-Za-zÀ-ÿ0-9]+", str(proficiency.get("Context", "")).lower()))
        context_words = set(re.findall(r"[A-Za-zÀ-ÿ0-9]+", context.lower()))
        return bool(keywords & context_words)

    # -----------------------------------------------------------------------
    # Inventory
    # -----------------------------------------------------------------------

    def add_item_to_inventory(self, entity_instance_id: str, item_id: int) -> str:
        entity = self._require_instance(entity_instance_id)
        if entity.data["Type"] == "STATIC":
            raise EngineError("ACTION_UNAVAILABLE", "Static entities cannot have inventories.")

        model = self._require_item_model(item_id)

        instance_id = new_runtime_id()
        item = ItemInstance(
            instance_id=instance_id,
            model_id=item_id,
            data=deep_copy(model.data),
        )

        entity.inventory[instance_id] = item
        self._item_instances[instance_id] = (entity_instance_id, item)
        return instance_id

    def remove_item_from_inventory(self, entity_instance_id: str, item_instance_id: str) -> dict:
        entity = self._require_instance(entity_instance_id)
        if item_instance_id not in entity.inventory:
            raise EngineError("ID_NOT_FOUND", f"Item instance {item_instance_id} is not in the inventory.")

        entity.inventory.pop(item_instance_id)
        self._item_instances.pop(item_instance_id, None)
        return success()

    def list_inventory_items(self, entity_instance_id: str) -> list[str]:
        entity = self._require_instance(entity_instance_id)
        return list(entity.inventory.keys())

    def list_abilities(self, entity_instance_id: str) -> list[int]:
        entity = self._require_instance(entity_instance_id)
        return list(entity.abilities)

    # -----------------------------------------------------------------------
    # Items and abilities
    # -----------------------------------------------------------------------

    def list_item_models(self) -> list[int]:
        return list(self.item_models.keys())

    def list_ability_models(self) -> list[int]:
        return list(self.ability_models.keys())

    def get_item_or_ability(self, object_id: int) -> dict:
        model = self._require_item_or_ability_model(object_id)
        return {
            "ModelID": getattr(model, "item_id", None) or model.ability_id,
            **deep_copy(model.data),
        }

    def create_item_or_ability(self, config: dict) -> int:
        if not isinstance(config, dict):
            raise EngineError("INVALID_ARGUMENT", "CONFIG must be an object.")

        object_type = config.get("Type")
        if object_type not in {"ITEM", "ABILITY"}:
            raise EngineError("INVALID_VALUE", "Type must be ITEM or ABILITY.")

        if not config.get("Name"):
            raise EngineError("INVALID_VALUE", "The object must have a Name.")

        functionality = config.get("Functionality")
        if functionality not in FUNCTIONALITIES:
            raise EngineError(
                "INVALID_VALUE",
                f"Functionality must be one of {sorted(FUNCTIONALITIES)}.",
            )

        data = deep_copy(config)
        data.setdefault("Rarity", 1)
        data.setdefault("Uses", None)
        data.setdefault("Parameters", {})

        object_id = self._next_item_id
        self._next_item_id += 1
        if object_type == "ITEM":
            self.item_models[object_id] = ItemModel(object_id, data)
        else:
            self.ability_models[object_id] = AbilityModel(object_id, data)
        return object_id

    def assign_ability(self, entity_instance_id: str, ability_id: int) -> dict:
        entity = self._require_instance(entity_instance_id)
        model = self._require_ability_model(ability_id)

        if model.data.get("Type") != "ABILITY":
            raise EngineError("INVALID_VALUE", "The selected model is not a ABILITY.")

        if ability_id not in entity.abilities:
            entity.abilities.append(ability_id)
        return success()

    def use_item(
        self,
        entity_instance_id: str,
        item_instance_id: str,
        args: Optional[dict] = None,
    ) -> dict:
        """Use an inventory item instance.

        For abilities, use `use_ability` with the model ID.
        """
        args = args or {}
        entity = self._require_instance(entity_instance_id)

        if item_instance_id not in entity.inventory:
            raise EngineError("ID_NOT_FOUND", f"Item instance {item_instance_id} is not in the inventory.")

        item = entity.inventory[item_instance_id]
        return self._execute_effect(
            source_entity=entity,
            model_data=item.data,
            args=args,
            consumed_item=item,
        )

    def use_ability(
        self,
        entity_instance_id: str,
        ability_id: int,
        args: Optional[dict] = None,
    ) -> dict:
        entity = self._require_instance(entity_instance_id)
        model = self._require_ability_model(ability_id)
        if model.data.get("Type") != "ABILITY":
            raise EngineError("INVALID_VALUE", "The selected model is not a ABILITY.")
        if ability_id not in entity.abilities:
            raise EngineError("ACTION_UNAVAILABLE", "The entity does not possess this ability.")

        return self._execute_effect(
            source_entity=entity,
            model_data=model.data,
            args=args or {},
            consumed_item=None,
        )

    def _execute_effect(
        self,
        source_entity: EntityInstance,
        model_data: dict,
        args: dict,
        consumed_item: Optional[ItemInstance],
    ) -> dict:
        if self._has_blocking_condition(source_entity):
            raise EngineError(
                "ACTION_UNAVAILABLE",
                "The entity cannot perform actions because of an active condition.",
            )

        uses = model_data.get("Uses")
        if consumed_item is not None and uses is not None:
            if int(uses) <= 0:
                raise EngineError("ACTION_UNAVAILABLE", "This item has no uses remaining.")

        functionality = model_data["Functionality"]
        params = deep_copy(model_data.get("Parameters", {}))

        if functionality == "NARRATIVE":
            result = {
                "Success": True,
                "Type": "NARRATIVE",
                "Effect": params.get("Effect", ""),
                "ActivationContext": params.get("ActivationContext", ""),
            }
        elif functionality == "PASSIVE":
            result = self._execute_passive(source_entity, params)
        elif functionality == "SIMPLE_USE":
            result = self._execute_simple_use(source_entity, params)
        elif functionality == "TARGET_EFFECT":
            result = self._execute_target_effect(source_entity, params, args)
        elif functionality == "POSITION_EFFECT":
            result = self._execute_position_effect(source_entity, params, args)
        else:
            raise EngineError("INVALID_VALUE", "Unknown functionality.")

        if consumed_item is not None and uses is not None:
            consumed_item.data["Uses"] = int(uses) - 1

        self._advance_action(source_entity.instance_id)
        return result

    def _execute_passive(self, entity: EntityInstance, params: dict) -> dict:
        attribute = params.get("TargetAttribute")
        modifier = int(self.evaluate_expression(str(params.get("Modifier", 0))))
        if attribute not in ATTRIBUTES:
            raise EngineError("INVALID_VALUE", "Passive TargetAttribute must be an attribute.")
        old = entity.data["Attributes"][attribute]
        new_value = self.modify_attribute(entity.instance_id, attribute, modifier)
        return {
            "Success": True,
            "Type": "PASSIVE",
            "Attribute": attribute,
            "Modifier": modifier,
            "OldValue": old,
            "NewValue": new_value,
        }

    def _execute_simple_use(self, entity: EntityInstance, params: dict) -> dict:
        status = params.get("TargetAttribute")
        if status not in STATUSES:
            raise EngineError("INVALID_VALUE", "SIMPLE_USE requires a valid status target.")
        modifier = int(self.evaluate_expression(str(params.get("Modifier", 0))))
        old = entity.data[status]
        new = self.modify_status(entity.instance_id, status, modifier)
        return {
            "Success": True,
            "Type": "SIMPLE_USE",
            "TargetAttribute": status,
            "Modifier": modifier,
            "PreviousValue": old,
            "PreviousHP": old if status == "HP" else None,
            "NewValue": new,
        }

    def _execute_target_effect(self, source_entity: EntityInstance, params: dict, args: dict) -> dict:
        target_id = args.get("Target", params.get("Target"))
        if target_id is None:
            raise EngineError("INVALID_TARGET", "TARGET_EFFECT requires Target.")

        target = self._require_instance(str(target_id))
        target_status = params.get("TargetAttribute")
        if target_status not in STATUSES:
            raise EngineError("INVALID_VALUE", "Target effect requires a valid status target.")

        modifier = int(self.evaluate_expression(str(params.get("Modifier", 0))))
        previous = int(target.data[target_status])
        new = self.modify_status(target.instance_id, target_status, modifier)

        return {
            "Success": True,
            "Type": "TARGET_EFFECT",
            "Target": target.instance_id,
            "TargetAttribute": target_status,
            "Modifier": modifier,
            "PreviousValue": previous,
            "PreviousHP": previous if target_status == "HP" else None,
            "CurrentHP": new if target_status == "HP" else None,
            "NewValue": new,
        }

    def _execute_position_effect(self, source_entity: EntityInstance, params: dict, args: dict) -> dict:
        x = args.get("X", params.get("X"))
        y = args.get("Y", params.get("Y"))
        current = self._require_current_map()
        self._require_position(x, y, current)

        target_status = params.get("TargetAttribute")
        modifier_expression = str(params.get("Modifier", 0))
        affected = []

        for target in current.entity_instances.values():
            if target.x == x and target.y == y and target.data["Type"] in {"PLAYER", "NPC"}:
                if target_status not in STATUSES:
                    raise EngineError("INVALID_VALUE", "Position effect requires a valid status target.")
                modifier = int(self.evaluate_expression(modifier_expression))
                previous = int(target.data[target_status])
                new = self.modify_status(target.instance_id, target_status, modifier)
                affected.append({
                    "InstanceID": target.instance_id,
                    "TargetAttribute": target_status,
                    "PreviousValue": previous,
                    "CurrentValue": new,
                })

        return {
            "Success": True,
            "Type": "POSITION_EFFECT",
            "X": x,
            "Y": y,
            "Affected": affected,
        }

    # -----------------------------------------------------------------------
    # Conditions
    # -----------------------------------------------------------------------

    def apply_condition(
        self,
        instance_id: str,
        condition: str | dict,
        config: Optional[dict] = None,
    ) -> dict:
        entity = self._require_instance(instance_id)

        if isinstance(condition, dict):
            definition = deep_copy(condition)
        else:
            definition = {
                "Name": condition,
                **deep_copy(config or {}),
            }

        name = definition.get("Name")
        if not name:
            raise EngineError("INVALID_VALUE", "Condition must have a Name.")

        duration = definition.get("Duration")
        if duration is not None:
            if not isinstance(duration, int) or duration < 0:
                raise EngineError("INVALID_VALUE", "Condition duration must be a non-negative integer.")

        duration_unit = definition.get("DurationUnit", "ACTIONS")
        if duration_unit not in CONDITION_DURATION_UNITS:
            raise EngineError("INVALID_VALUE", "Invalid condition duration unit.")

        effect = deep_copy(definition.get("Effect", {}))
        rules = deep_copy(definition.get("Rules", {}))

        # Explicit condition definitions from the specification.
        defaults = {
            "OnFire": {"Duration": 5, "DurationUnit": "ACTIONS", "Effect": {"HP": "-1d4"}},
            "Frozen": {"Duration": None, "DurationUnit": "INDEFINITE", "Effect": {},
                          "Rules": {"BlocksMovement": True, "BlocksActions": True}},
            "Poisoned": {"Duration": 3, "DurationUnit": "ACTIONS", "Effect": {"HP": "-1d8"}},
            "Paralyzed": {"Duration": None, "DurationUnit": "INDEFINITE", "Effect": {},
                           "Rules": {"BlocksMovement": True, "BlocksActions": True}},
            "Unconscious": {"Duration": None, "DurationUnit": "INDEFINITE", "Effect": {},
                          "Rules": {"BlocksMovement": True, "BlocksActions": True}},
            "Insane": {"Duration": None, "DurationUnit": "INDEFINITE", "Effect": {},
                             "Rules": {"LosesControl": True}},
        }

        if name in defaults:
            default = defaults[name]
            if duration is None:
                duration = default["Duration"]
            if duration_unit == "ACTIONS" and default["DurationUnit"] != "ACTIONS":
                duration_unit = default["DurationUnit"]
            if not effect:
                effect = default["Effect"]
            merged_rules = deep_copy(default.get("Rules", {}))
            merged_rules.update(rules)
            rules = merged_rules

        condition_id = new_runtime_id()
        instance = ConditionInstance(
            condition_id=condition_id,
            name=name,
            duration=duration,
            duration_unit=duration_unit,
            effect=effect,
            rules=rules,
        )

        # Fire + Frozen explicitly cancel each other.
        if name in {"OnFire", "Frozen"}:
            opposite = "Frozen" if name == "OnFire" else "OnFire"
            for cid, active in list(entity.conditions.items()):
                if active.name == opposite:
                    self._remove_condition_internal(entity, cid)
                    return {
                        "Success": True,
                        "Condition": name,
                        "Interaction": f"{name}+{opposite} removed both conditions.",
                    }

        entity.conditions[condition_id] = instance
        self._conditions[condition_id] = (instance_id, instance)

        return {
            "Success": True,
            "ConditionID": condition_id,
            "Name": name,
            "Duration": duration,
            "DurationUnit": duration_unit,
        }

    def remove_condition(self, instance_id: str, condition_id: str) -> dict:
        entity = self._require_instance(instance_id)
        if condition_id not in entity.conditions:
            raise EngineError("ID_NOT_FOUND", f"Condition {condition_id} is not active.")
        self._remove_condition_internal(entity, condition_id)
        return success()

    def _remove_condition_internal(self, entity: EntityInstance, condition_id: str):
        entity.conditions.pop(condition_id, None)
        self._conditions.pop(condition_id, None)

    def list_conditions(self, instance_id: str) -> list[dict]:
        entity = self._require_instance(instance_id)
        return [
            {
                "ConditionID": c.condition_id,
                "Name": c.name,
                "Duration": c.duration,
                "DurationUnit": c.duration_unit,
                "Effect": deep_copy(c.effect),
                "Rules": deep_copy(c.rules),
            }
            for c in entity.conditions.values()
        ]

    def change_condition_duration(self, condition_id: str, duration: Optional[int]) -> dict:
        if condition_id not in self._conditions:
            raise EngineError("ID_NOT_FOUND", f"Condition {condition_id} does not exist.")

        if duration is not None and (not isinstance(duration, int) or duration < 0):
            raise EngineError("INVALID_VALUE", "Duration must be a non-negative integer or None.")

        _, condition = self._conditions[condition_id]
        condition.duration = duration
        return success(duration)

    def _has_blocking_condition(self, entity: EntityInstance) -> bool:
        return any(
            c.rules.get("BlocksActions") or c.rules.get("BlocksMovement")
            for c in entity.conditions.values()
        )

    def _advance_action(self, instance_id: str):
        """Process action-based condition effects and durations."""
        entity = self._require_instance(instance_id)

        for condition_id, condition in list(entity.conditions.items()):
            # Conditions with HP effects apply once per action.
            if condition.duration_unit == "ACTIONS":
                hp_expression = condition.effect.get("HP")
                if hp_expression:
                    damage_or_heal = int(self.evaluate_expression(str(hp_expression)))
                    self.modify_status(instance_id, "HP", damage_or_heal)

                if condition.duration is not None:
                    condition.duration -= 1
                    if condition.duration <= 0:
                        self._remove_condition_internal(entity, condition_id)

    # -----------------------------------------------------------------------
    # Convenience combat rule
    # -----------------------------------------------------------------------

    def attack(
        self,
        attacker_id: str,
        target_id: str,
        attack_modifier: int = 0,
        damage_expression: str = "1d4",
    ) -> dict:
        attacker = self._require_instance(attacker_id)
        target = self._require_instance(target_id)

        if attacker.data["Type"] == "STATIC" or target.data["Type"] == "STATIC":
            raise EngineError("INVALID_TARGET", "Static entities cannot participate in this combat attack.")

        if self._has_blocking_condition(attacker):
            raise EngineError("ACTION_UNAVAILABLE", "Attacker cannot act.")

        roll = self.roll_dice(20)
        total_attack = roll + attack_modifier
        hit = total_attack >= int(target.data["AC"])

        result = {
            "Hit": hit,
            "Roll": roll,
            "AttackModifier": attack_modifier,
            "AttackResult": total_attack,
            "TargetAC": target.data["AC"],
        }

        if hit:
            damage = int(self.evaluate_expression(damage_expression))
            hp_before = int(target.data["HP"])
            hp_after = self.modify_status(target_id, "HP", -damage)
            result.update({
                "Damage": damage,
                "PreviousHP": hp_before,
                "CurrentHP": hp_after,
            })
        else:
            result.update({
                "Damage": 0,
                "PreviousHP": target.data["HP"],
                "CurrentHP": target.data["HP"],
            })

        self._advance_action(attacker_id)
        return result

    # -----------------------------------------------------------------------
    # Generic command interface for an LLM
    # -----------------------------------------------------------------------

    def command(self, command_name: str, **kwargs) -> dict:
        """Dispatch a named engine command and normalize errors.

        This is intentionally string-based so an LLM can use a stable,
        documented command surface without knowing Python implementation
        details.
        """
        if not isinstance(command_name, str):
            return EngineError("INVALID_ARGUMENT", "Command name must be a string.").to_dict()

        commands = {
            "roll_dice": self.roll_dice,
            "evaluate_expression": self.evaluate_expression,
            "calculate_distance": self.calculate_distance,
            "create_map": self.create_map,
            "populate_map": self.populate_map,
            "list_maps": self.list_maps,
            "get_current_map": self.get_current_map,
            "get_current_map_id": self.get_current_map_id,
            "set_current_map": self.set_current_map,
            "create_entity": self.create_entity,
            "spawn_entity": self.spawn_entity,
            "list_entity_models": self.list_entity_models,
            "list_map_entities": self.list_map_entities,
            "remove_entity_from_map": self.remove_entity_from_map,
            "list_entity_actions": self.list_entity_actions,
            "handle_entity_action": self.handle_entity_action,
            "make_attribute_test": self.make_attribute_test,
            "get_basic_data": self.get_basic_data,
            "get_basic_field": self.get_basic_field,
            "set_position": self.set_position,
            "modify_status": self.modify_status,
            "modify_attribute": self.modify_attribute,
            "add_item_to_inventory": self.add_item_to_inventory,
            "remove_item_from_inventory": self.remove_item_from_inventory,
            "list_inventory_items": self.list_inventory_items,
            "list_abilities": self.list_abilities,
            "list_item_models": self.list_item_models,
            "list_ability_models": self.list_ability_models,
            "get_item_or_ability": self.get_item_or_ability,
            "create_item_or_ability": self.create_item_or_ability,
            "assign_ability": self.assign_ability,
            "use_item": self.use_item,
            "use_ability": self.use_ability,
            "apply_condition": self.apply_condition,
            "remove_condition": self.remove_condition,
            "list_conditions": self.list_conditions,
            "change_condition_duration": self.change_condition_duration,
            "attack": self.attack,
        }

        function = commands.get(command_name)
        if function is None:
            return EngineError("ACTION_NOT_FOUND", f"Unknown engine command: {command_name}.").to_dict()

        try:
            value = function(**kwargs)
            if isinstance(value, dict) and "Success" in value:
                return value
            return success(value)
        except EngineError as error:
            return error.to_dict()


# ---------------------------------------------------------------------------
# Demo world
# ---------------------------------------------------------------------------

def create_demo_engine() -> Engine:
    """Build a small world used by the Pygame editor and examples."""
    engine = Engine(seed=42)

    terrain = [
        ["EARTH", "EARTH", "EARTH", "SAND",  "SAND",  "EARTH", "EARTH", "EARTH", "EARTH", "EARTH"],
        ["EARTH", "EARTH", "EARTH", "SAND",  "SAND",  "EARTH", "WATER", "WATER", "EARTH", "EARTH"],
        ["EARTH", "EARTH", "SAND",  "SAND",  "EARTH", "EARTH", "WATER", "WATER", "EARTH", "EARTH"],
        ["EARTH", "EARTH", "EARTH", "EARTH", "EARTH", "EARTH", "WATER", "EARTH", "EARTH", "EARTH"],
        ["EARTH", "EARTH", "EARTH", "EARTH", "EARTH", "EARTH", "EARTH", "EARTH", "EARTH", "EARTH"],
        ["EARTH", "EARTH", "EARTH", "EARTH", "SAND",  "SAND",  "EARTH", "EARTH", "EARTH", "EARTH"],
        ["EARTH", "EARTH", "EARTH", "EARTH", "SAND",  "SAND",  "EARTH", "EARTH", "EARTH", "EARTH"],
        ["EARTH", "EARTH", "EARTH", "EARTH", "EARTH", "EARTH", "EARTH", "EARTH", "EARTH", "EARTH"],
    ]

    map_id = engine.create_map(10, 8, terrain)
    engine.populate_map(10, 8, [
        ["NONE", "NONE", "NONE", "NONE", "NONE", "NONE", "NONE", "NONE", "NONE", "NONE"],
        ["NONE", "HOUSE", "HOUSE", "NONE", "NONE", "NONE", "NONE", "NONE", "NONE", "NONE"],
        ["NONE", "HOUSE", "HOUSE", "NONE", "NONE", "NONE", "NONE", "NONE", "STORE", "NONE"],
        ["NONE", "NONE", "NONE", "NONE", "NONE", "NONE", "NONE", "NONE", "NONE", "NONE"],
        ["NONE", "NONE", "NONE", "STATUE", "NONE", "NONE", "NONE", "NONE", "NONE", "NONE"],
        ["NONE", "NONE", "NONE", "NONE", "NONE", "NONE", "NONE", "NONE", "NONE", "NONE"],
        ["NONE", "NONE", "NONE", "NONE", "NONE", "NONE", "NONE", "NONE", "NONE", "NONE"],
        ["NONE", "NONE", "NONE", "NONE", "NONE", "NONE", "NONE", "NONE", "NONE", "NONE"],
    ])

    goblin_id = engine.create_entity({
        "Type": "NPC",
        "Name": "Goblin",
        "HPMax": 10,
        "HP": 10,
        "AC": 12,
        "XP": 20,
        "Attributes": {"STR": 2, "DEX": 3, "INT": 0, "CHA": 0, "CON": 1},
        "Proficiencies": {
            "Survival": {
                "Modifier": 2,
                "Context": "situations related to survival",
            }
        },
    })

    hero_id = engine.create_entity({
        "Type": "PLAYER",
        "Name": "Adventurer",
        "HPMax": 30,
        "HP": 30,
        "AC": 14,
        "XP": 20,
        "Attributes": {"STR": 4, "DEX": 3, "INT": 2, "CHA": 1, "CON": 4},
        "Proficiencies": {
            "SleightOfHand": {
                "Modifier": 2,
                "Context": "manual manipulation thefts tricks",
            }
        },
    })

    statue_id = engine.create_entity({
        "Type": "STATIC",
        "Name": "Ancient Statue",
    })

    goblin = engine.spawn_entity(goblin_id, 7, 5)
    hero = engine.spawn_entity(hero_id, 2, 6)
    engine.spawn_entity(statue_id, 3, 4)

    dagger_id = engine.create_item_or_ability({
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

    potion_id = engine.create_item_or_ability({
        "Type": "ITEM",
        "Name": "Healing Potion",
        "Rarity": 1,
        "Functionality": "SIMPLE_USE",
        "Uses": 1,
        "Parameters": {
            "TargetAttribute": "HP",
            "Modifier": "2d4",
        },
    })

    fireball_id = engine.create_item_or_ability({
        "Type": "ABILITY",
        "Name": "Fire Bolt",
        "Rarity": 1,
        "Functionality": "TARGET_EFFECT",
        "Uses": 10,
        "Parameters": {
            "TargetAttribute": "HP",
            "Modifier": "-1d6",
        },
    })

    engine.add_item_to_inventory(hero, dagger_id)
    engine.add_item_to_inventory(hero, potion_id)
    engine.assign_ability(hero, fireball_id)

    return engine
