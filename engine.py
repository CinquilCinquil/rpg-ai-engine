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
    - Runtime instances use 8-character hexadecimal IDs, as recommended by
      the specification's final section.

Coordinates:
    - (0, 0) is the top-left cell.
    - X = column, Y = row.
    - Matrices are indexed as matrix[Y][X].

This module has no Pygame dependency, so it can also be used headlessly.
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
ATTRIBUTES = {"FOR", "AGI", "INT", "PRE", "CONS"}
STATUSES = {"HP", "HPMAX", "AC", "XP"}

FUNCTIONALITIES = {
    "PASSIVE",
    "EFEITO_EM_ALVO",
    "EFEITO_EM_POSICAO",
    "USO_SIMPLES",
    "NARRATIVO",
}

CONDITION_DURATION_UNITS = {"ACOES", "TURNOS", "RODADAS", "INDEFINIDO"}


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
            "Sucesso": False,
            "Erro": {
                "Codigo": self.code,
                "Mensagem": self.message,
            },
        }
        if self.details:
            result["Erro"]["Detalhes"] = copy.deepcopy(self.details)
        return result


def success(value: Any = None) -> dict:
    """Create a standardized successful command response."""
    result = {"Sucesso": True}
    if value is not None:
        result["Resultado"] = value
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
                    "VALOR_INVALIDO",
                    "Dice expressions require positive count and sides.",
                )
            if count > 10000:
                raise EngineError(
                    "VALOR_INVALIDO",
                    "A dice expression may contain at most 10000 dice.",
                )
            total = sum(self.dice_roller(sides) for _ in range(count))
            return str(total)

        return _DICE_RE.sub(replacement, expression)

    def evaluate(self, expression: str) -> int | float:
        if not isinstance(expression, str) or not expression.strip():
            raise EngineError(
                "ARGUMENTO_INVALIDO",
                "Expression must be a non-empty string.",
            )

        expression = self.replace_dice(expression.strip())

        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise EngineError(
                "ARGUMENTO_INVALIDO",
                f"Invalid expression: {expression}",
            ) from exc

        value = self._eval_node(tree.body)

        if isinstance(value, float) and value.is_integer():
            return int(value)
        return value

    def _eval_node(self, node: ast.AST) -> int | float:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise EngineError("ARGUMENTO_INVALIDO", "Only numeric constants are allowed.")
            return node.value

        if isinstance(node, ast.BinOp) and type(node.op) in self.BINARY_OPS:
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)

            if isinstance(node.op, (ast.Pow,)):
                if abs(right) > 100:
                    raise EngineError("VALOR_INVALIDO", "Exponent is too large.")

            try:
                return self.BINARY_OPS[type(node.op)](left, right)
            except ZeroDivisionError as exc:
                raise EngineError("VALOR_INVALIDO", "Division by zero.") from exc

        if isinstance(node, ast.UnaryOp) and type(node.op) in self.UNARY_OPS:
            return self.UNARY_OPS[type(node.op)](self._eval_node(node.operand))

        raise EngineError(
            "ARGUMENTO_INVALIDO",
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

    The class exposes the public API described by the specification while
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
        self._expression_evaluator = SafeExpressionEvaluator(self.rolar_dado)

        self.maps: dict[int, MapState] = {}
        self.entity_models: dict[int, EntityModel] = {}
        self.item_models: dict[int, ItemModel] = {}

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
            raise EngineError("ARGUMENTO_INVALIDO", f"{name} must be an integer.")
        return value

    @staticmethod
    def _validate_matrix(matrix: Any, width: int, height: int, valid_values: set[str], name: str):
        if not isinstance(matrix, list) or len(matrix) != height:
            raise EngineError(
                "VALOR_INVALIDO",
                f"{name} must contain exactly {height} rows.",
            )
        for y, row in enumerate(matrix):
            if not isinstance(row, list) or len(row) != width:
                raise EngineError(
                    "VALOR_INVALIDO",
                    f"{name}[{y}] must contain exactly {width} columns.",
                )
            for x, value in enumerate(row):
                if value not in valid_values:
                    raise EngineError(
                        "VALOR_INVALIDO",
                        f"Invalid {name} value at ({x}, {y}): {value}.",
                    )

    def _require_current_map(self) -> MapState:
        if self.current_map_id is None or self.current_map_id not in self.maps:
            raise EngineError("MAPA_INEXISTENTE", "No map is currently selected.")
        return self.maps[self.current_map_id]

    def _require_map(self, map_id: int) -> MapState:
        if map_id not in self.maps:
            raise EngineError("MAPA_INEXISTENTE", f"Map {map_id} does not exist.")
        return self.maps[map_id]

    def _require_model(self, entity_id: int) -> EntityModel:
        if entity_id not in self.entity_models:
            raise EngineError("ENTIDADE_INEXISTENTE", f"Entity model {entity_id} does not exist.")
        return self.entity_models[entity_id]

    def _require_instance(self, instance_id: str) -> EntityInstance:
        if instance_id not in self._instances:
            raise EngineError("ID_INEXISTENTE", f"Entity instance {instance_id} does not exist.")
        return self._instances[instance_id]

    def _require_item_model(self, item_id: int) -> ItemModel:
        if item_id not in self.item_models:
            raise EngineError("ID_INEXISTENTE", f"Item model {item_id} does not exist.")
        return self.item_models[item_id]

    def _require_position(self, x: int, y: int, map_state: Optional[MapState] = None):
        map_state = map_state or self._require_current_map()
        if not isinstance(x, int) or not isinstance(y, int):
            raise EngineError("POSICAO_INVALIDA", "X and Y must be integers.")
        if not (0 <= x < map_state.width and 0 <= y < map_state.height):
            raise EngineError(
                "POSICAO_INVALIDA",
                f"Position ({x}, {y}) is outside map {map_state.map_id}.",
            )

    # -----------------------------------------------------------------------
    # Dice and expressions
    # -----------------------------------------------------------------------

    def rolar_dado(self, n: int) -> int:
        """Roll an N-sided die, uniformly from 1 through N."""
        n = self._require_int(n, "N")
        if n <= 0:
            raise EngineError("VALOR_INVALIDO", "N must be greater than zero.")
        return self.random.randint(1, n)

    def calcular_expressao(self, expression: str) -> int | float:
        """Evaluate arithmetic and XdY dice expressions."""
        return self._expression_evaluator.evaluate(expression)

    # -----------------------------------------------------------------------
    # Geometry
    # -----------------------------------------------------------------------

    def calcular_distancia(self, x: int, y: int, w: int, z: int) -> int:
        for value, name in ((x, "X"), (y, "Y"), (w, "W"), (z, "Z")):
            self._require_int(value, name)
        return abs(x - w) + abs(y - z)

    # -----------------------------------------------------------------------
    # Maps
    # -----------------------------------------------------------------------

    def criar_mapa(self, width: int, height: int, config: list[list[str]]) -> int:
        width = self._require_int(width, "W")
        height = self._require_int(height, "H")
        if width <= 0 or height <= 0:
            raise EngineError("VALOR_INVALIDO", "Map dimensions must be positive.")

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

    def popular_mapa(self, width: int, height: int, config: list[list[str]]) -> dict:
        current = self._require_current_map()
        if width != current.width or height != current.height:
            raise EngineError(
                "VALOR_INVALIDO",
                "Structure dimensions must match the current map dimensions.",
            )

        self._validate_matrix(config, width, height, STRUCTURES, "Structure")
        current.structures = deep_copy(config)
        return success()

    def listar_mapas(self) -> list[int]:
        return list(self.maps.keys())

    def consultar_mapa_atual(self) -> dict:
        current = self._require_current_map()
        return {
            "MapaID": current.map_id,
            "Largura": current.width,
            "Altura": current.height,
            "Terreno": deep_copy(current.terrain),
            "Estruturas": deep_copy(current.structures),
            "Entidades": [
                {
                    "InstanciaID": instance.instance_id,
                    "EntidadeID": instance.model_id,
                    "X": instance.x,
                    "Y": instance.y,
                }
                for instance in current.entity_instances.values()
            ],
        }

    def consultar_mapa_atual_id(self) -> int:
        return self._require_current_map().map_id

    def definir_mapa_atual(self, map_id: int) -> dict:
        self._require_map(map_id)
        self.current_map_id = map_id
        return success(map_id)

    # -----------------------------------------------------------------------
    # Entity database
    # -----------------------------------------------------------------------

    def criar_entidade(self, config: dict) -> int:
        if not isinstance(config, dict):
            raise EngineError("ARGUMENTO_INVALIDO", "Entity CONFIG must be an object.")

        entity_type = config.get("Tipo")
        if entity_type not in ENTITY_TYPES:
            raise EngineError(
                "VALOR_INVALIDO",
                f"Tipo must be one of {sorted(ENTITY_TYPES)}.",
            )

        if not config.get("Nome"):
            raise EngineError("VALOR_INVALIDO", "Entity must have a Nome.")

        data = deep_copy(config)

        if entity_type in {"PLAYER", "NPC"}:
            data.setdefault("HPMax", data.get("HP", 1))
            data.setdefault("HP", data["HPMax"])
            data.setdefault("AC", 10)
            data.setdefault("XP", 0)
            data.setdefault("Atributos", {})
            data["Atributos"] = {
                attribute: int(data["Atributos"].get(attribute, 0))
                for attribute in ATTRIBUTES
            }
            data.setdefault("Proficiencias", {})

        entity_id = self._next_entity_id
        self._next_entity_id += 1
        self.entity_models[entity_id] = EntityModel(entity_id, data)
        return entity_id

    # -----------------------------------------------------------------------
    # Entity instances
    # -----------------------------------------------------------------------

    def criar_entidade_no_mapa(self, entity_id: int, x: int, y: int) -> str:
        model = self._require_model(entity_id)
        current = self._require_current_map()
        self._require_position(x, y, current)

        if not self._terrain_allows_movement(current.terrain[y][x], model.data["Tipo"]):
            raise EngineError(
                "POSICAO_INVALIDA",
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

    def listar_entidades_no_banco(self) -> list[int]:
        return list(self.entity_models.keys())

    def listar_entidades_no_mapa(self) -> list[str]:
        current = self._require_current_map()
        return list(current.entity_instances.keys())

    def remover_entidade_do_mapa(self, instance_id: str) -> dict:
        instance = self._require_instance(instance_id)
        current = self._require_current_map()
        if instance_id not in current.entity_instances:
            raise EngineError(
                "ID_INEXISTENTE",
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

    def consultar_entidade(self, instance_id: str) -> dict:
        instance = self._require_instance(instance_id)
        return self._entity_snapshot(instance)

    def consultar_dados_basicos(self, instance_id: str) -> dict:
        instance = self._require_instance(instance_id)
        data = instance.data
        if data["Tipo"] == "STATIC":
            return {
                "Tipo": "STATIC",
                "Nome": data["Nome"],
                "X": instance.x,
                "Y": instance.y,
            }

        return {
            "Tipo": data["Tipo"],
            "Nome": data["Nome"],
            "HP": data["HP"],
            "HPMax": data["HPMax"],
            "AC": data["AC"],
            "XP": data["XP"],
            "Atributos": deep_copy(data["Atributos"]),
            "Proficiencias": deep_copy(data.get("Proficiencias", {})),
            "Condicoes": self.listar_condicoes(instance_id),
        }

    def consultar_dado_basico(self, instance_id: str, field_name: str) -> Any:
        instance = self._require_instance(instance_id)
        if field_name == "Condicoes":
            return self.listar_condicoes(instance_id)
        if field_name not in instance.data:
            raise EngineError("ID_INEXISTENTE", f"Field {field_name} does not exist.")
        return deep_copy(instance.data[field_name])

    def _entity_snapshot(self, instance: EntityInstance) -> dict:
        snapshot = deep_copy(instance.data)
        snapshot.update({
            "InstanciaID": instance.instance_id,
            "ModeloID": instance.model_id,
            "X": instance.x,
            "Y": instance.y,
            "Inventario": list(instance.inventory.keys()),
            "Habilidades": list(instance.abilities),
            "Condicoes": self.listar_condicoes(instance.instance_id),
        })
        return snapshot

    # -----------------------------------------------------------------------
    # Movement / status / attributes
    # -----------------------------------------------------------------------

    def definir_posicao(self, instance_id: str, x: int, y: int) -> dict:
        instance = self._require_instance(instance_id)
        current = self._require_current_map()

        self._require_position(x, y, current)

        if self._has_blocking_condition(instance):
            raise EngineError(
                "ACAO_INDISPONIVEL",
                "The entity cannot move because of an active condition.",
            )

        if not self._terrain_allows_movement(current.terrain[y][x], instance.data["Tipo"]):
            raise EngineError(
                "POSICAO_INVALIDA",
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

    def alterar_status(self, instance_id: str, status: str, modifier: int) -> int:
        instance = self._require_instance(instance_id)
        if instance.data["Tipo"] == "STATIC":
            raise EngineError("ACAO_INDISPONIVEL", "Static entities have no character status.")

        if status not in STATUSES:
            raise EngineError("VALOR_INVALIDO", f"Invalid status: {status}.")
        if not isinstance(modifier, int):
            raise EngineError("ARGUMENTO_INVALIDO", "MOD must be an integer.")

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

    def alterar_atributo(self, instance_id: str, attribute: str, modifier: int) -> int:
        instance = self._require_instance(instance_id)
        if instance.data["Tipo"] == "STATIC":
            raise EngineError("ACAO_INDISPONIVEL", "Static entities have no attributes.")
        if attribute not in ATTRIBUTES:
            raise EngineError("VALOR_INVALIDO", f"Invalid attribute: {attribute}.")
        if not isinstance(modifier, int):
            raise EngineError("ARGUMENTO_INVALIDO", "MOD must be an integer.")

        instance.data["Atributos"][attribute] += modifier
        return instance.data["Atributos"][attribute]

    # -----------------------------------------------------------------------
    # Entity actions
    # -----------------------------------------------------------------------

    def listar_acoes_da_entidade(self, instance_id: str) -> dict:
        instance = self._require_instance(instance_id)

        actions = {
            "DefinirPosicao": {
                "Args": {"X": "int", "Y": "int"},
                "Retorno": "void",
            }
        }

        if instance.data["Tipo"] in {"PLAYER", "NPC"}:
            actions.update({
                "AlterarStatus": {
                    "Args": {"Status": "HP|HPMax|AC|XP", "Modificador": "int"},
                    "Retorno": "int",
                },
                "AlterarAtributo": {
                    "Args": {"Atributo": "FOR|AGI|INT|PRE|CONS", "Modificador": "int"},
                    "Retorno": "int",
                },
                "AplicarTeste": {
                    "Args": {"Atributo": "FOR|AGI|INT|PRE|CONS", "Contexto": "string?"},
                    "Retorno": "object",
                },
                "UsarItem": {
                    "Args": {"Item": "string", "Args": "object"},
                    "Retorno": "object",
                },
                "AplicarCondicao": {
                    "Args": {"Condicao": "object", "Config": "object?"},
                    "Retorno": "object",
                },
            })

        return actions

    def manipular_entidade(self, instance_id: str, action: str, args: dict) -> Any:
        self._require_instance(instance_id)
        if action not in self.listar_acoes_da_entidade(instance_id):
            raise EngineError("ACAO_INEXISTENTE", f"Action {action} does not exist for this entity.")

        if not isinstance(args, dict):
            raise EngineError("ARGUMENTO_INVALIDO", "Args must be an object.")

        if action == "DefinirPosicao":
            return self.definir_posicao(instance_id, args.get("X"), args.get("Y"))
        if action == "AlterarStatus":
            return self.alterar_status(instance_id, args.get("Status"), args.get("Modificador"))
        if action == "AlterarAtributo":
            return self.alterar_atributo(instance_id, args.get("Atributo"), args.get("Modificador"))
        if action == "AplicarTeste":
            return self.aplicar_teste(instance_id, args.get("Atributo"), args.get("Contexto"))
        if action == "UsarItem":
            return self.usar_item_ou_habilidade(instance_id, args.get("Item"), args)
        if action == "AplicarCondicao":
            return self.aplicar_condicao(instance_id, args.get("Condicao"), args.get("Config"))

        raise EngineError("ACAO_INEXISTENTE", f"Unsupported action: {action}.")

    # -----------------------------------------------------------------------
    # Tests
    # -----------------------------------------------------------------------

    def aplicar_teste(self, instance_id: str, attribute: str, context: Optional[str] = None) -> dict:
        instance = self._require_instance(instance_id)
        if instance.data["Tipo"] == "STATIC":
            raise EngineError("ACAO_INDISPONIVEL", "Static entities cannot make attribute tests.")
        if attribute not in ATTRIBUTES:
            raise EngineError("VALOR_INVALIDO", f"Invalid attribute: {attribute}.")

        die = self.rolar_dado(20)
        attribute_modifier = int(instance.data["Atributos"].get(attribute, 0))
        proficiency_modifier = 0
        proficiency_name = None

        if context:
            for name, proficiency in instance.data.get("Proficiencias", {}).items():
                if not isinstance(proficiency, dict):
                    continue
                if self._proficiency_matches(proficiency, context):
                    modifier = int(proficiency.get("Modificador", 0))
                    if modifier > proficiency_modifier:
                        proficiency_modifier = modifier
                        proficiency_name = name

        result = die + attribute_modifier + proficiency_modifier
        return {
            "Dado": die,
            "Atributo": attribute_modifier,
            "Proficiencia": proficiency_modifier,
            "ProficienciaNome": proficiency_name,
            "Resultado": result,
        }

    @staticmethod
    def _proficiency_matches(proficiency: dict, context: str) -> bool:
        keywords = set(re.findall(r"[A-Za-zÀ-ÿ0-9]+", str(proficiency.get("Contexto", "")).lower()))
        context_words = set(re.findall(r"[A-Za-zÀ-ÿ0-9]+", context.lower()))
        return bool(keywords & context_words)

    # -----------------------------------------------------------------------
    # Inventory
    # -----------------------------------------------------------------------

    def inserir_item_no_inventario(self, entity_instance_id: str, item_id: int) -> str:
        entity = self._require_instance(entity_instance_id)
        if entity.data["Tipo"] == "STATIC":
            raise EngineError("ACAO_INDISPONIVEL", "Static entities cannot have inventories.")

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

    def remover_item_do_inventario(self, entity_instance_id: str, item_instance_id: str) -> dict:
        entity = self._require_instance(entity_instance_id)
        if item_instance_id not in entity.inventory:
            raise EngineError("ID_INEXISTENTE", f"Item instance {item_instance_id} is not in the inventory.")

        entity.inventory.pop(item_instance_id)
        self._item_instances.pop(item_instance_id, None)
        return success()

    def listar_itens_no_inventario(self, entity_instance_id: str) -> list[str]:
        entity = self._require_instance(entity_instance_id)
        return list(entity.inventory.keys())

    def listar_habilidades(self, entity_instance_id: str) -> list[int]:
        entity = self._require_instance(entity_instance_id)
        return list(entity.abilities)

    # -----------------------------------------------------------------------
    # Items and abilities
    # -----------------------------------------------------------------------

    def listar_itens_no_banco(self) -> list[int]:
        return list(self.item_models.keys())

    def listar_habilidades_no_banco(self) -> list[int]:
        return [
            item_id
            for item_id, model in self.item_models.items()
            if model.data.get("Tipo") == "HABILIDADE"
        ]

    def consultar_item_ou_habilidade(self, object_id: int) -> dict:
        model = self._require_item_model(object_id)
        return {
            "ModeloID": model.item_id,
            **deep_copy(model.data),
        }

    def criar_item_ou_habilidade(self, config: dict) -> int:
        if not isinstance(config, dict):
            raise EngineError("ARGUMENTO_INVALIDO", "CONFIG must be an object.")

        object_type = config.get("Tipo")
        if object_type not in {"ITEM", "HABILIDADE"}:
            raise EngineError("VALOR_INVALIDO", "Tipo must be ITEM or HABILIDADE.")

        if not config.get("Nome"):
            raise EngineError("VALOR_INVALIDO", "The object must have a Nome.")

        functionality = config.get("Funcionalidade")
        if functionality not in FUNCTIONALITIES:
            raise EngineError(
                "VALOR_INVALIDO",
                f"Funcionalidade must be one of {sorted(FUNCTIONALITIES)}.",
            )

        data = deep_copy(config)
        data.setdefault("Raridade", 1)
        data.setdefault("QuantidadeUsos", None)
        data.setdefault("Parametros", {})

        object_id = self._next_item_id
        self._next_item_id += 1
        self.item_models[object_id] = ItemModel(object_id, data)
        return object_id

    def atribuir_habilidade(self, entity_instance_id: str, ability_id: int) -> dict:
        entity = self._require_instance(entity_instance_id)
        model = self._require_item_model(ability_id)

        if model.data.get("Tipo") != "HABILIDADE":
            raise EngineError("VALOR_INVALIDO", "The selected model is not a HABILIDADE.")

        if ability_id not in entity.abilities:
            entity.abilities.append(ability_id)
        return success()

    def usar_item_ou_habilidade(
        self,
        entity_instance_id: str,
        item_instance_id: str,
        args: Optional[dict] = None,
    ) -> dict:
        """Use an inventory item instance.

        For abilities, use `usar_habilidade` with the model ID.
        """
        args = args or {}
        entity = self._require_instance(entity_instance_id)

        if item_instance_id not in entity.inventory:
            raise EngineError("ID_INEXISTENTE", f"Item instance {item_instance_id} is not in the inventory.")

        item = entity.inventory[item_instance_id]
        return self._execute_effect(
            source_entity=entity,
            model_data=item.data,
            args=args,
            consumed_item=item,
        )

    def usar_habilidade(
        self,
        entity_instance_id: str,
        ability_id: int,
        args: Optional[dict] = None,
    ) -> dict:
        entity = self._require_instance(entity_instance_id)
        model = self._require_item_model(ability_id)
        if model.data.get("Tipo") != "HABILIDADE":
            raise EngineError("VALOR_INVALIDO", "The selected model is not a HABILIDADE.")
        if ability_id not in entity.abilities:
            raise EngineError("ACAO_INDISPONIVEL", "The entity does not possess this ability.")

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
                "ACAO_INDISPONIVEL",
                "The entity cannot perform actions because of an active condition.",
            )

        uses = model_data.get("QuantidadeUsos")
        if consumed_item is not None and uses is not None:
            if int(uses) <= 0:
                raise EngineError("ACAO_INDISPONIVEL", "This item has no uses remaining.")

        functionality = model_data["Funcionalidade"]
        params = deep_copy(model_data.get("Parametros", {}))

        if functionality == "NARRATIVO":
            result = {
                "Sucesso": True,
                "Tipo": "NARRATIVO",
                "Efeito": params.get("Efeito", ""),
                "ContextoAtivacao": params.get("ContextoAtivacao", ""),
            }
        elif functionality == "PASSIVO":
            result = self._execute_passive(source_entity, params)
        elif functionality == "USO_SIMPLES":
            result = self._execute_simple_use(source_entity, params)
        elif functionality == "EFEITO_EM_ALVO":
            result = self._execute_target_effect(source_entity, params, args)
        elif functionality == "EFEITO_EM_POSICAO":
            result = self._execute_position_effect(source_entity, params, args)
        else:
            raise EngineError("VALOR_INVALIDO", "Unknown functionality.")

        if consumed_item is not None and uses is not None:
            consumed_item.data["QuantidadeUsos"] = int(uses) - 1

        self._advance_action(source_entity.instance_id)
        return result

    def _execute_passive(self, entity: EntityInstance, params: dict) -> dict:
        attribute = params.get("AtributoAlvo")
        modifier = int(self.calcular_expressao(str(params.get("Modificador", 0))))
        if attribute not in ATTRIBUTES:
            raise EngineError("VALOR_INVALIDO", "Passive AtributoAlvo must be an attribute.")
        new_value = self.alterar_atributo(entity.instance_id, attribute, modifier)
        return {
            "Sucesso": True,
            "Tipo": "PASSIVO",
            "Atributo": attribute,
            "Modificador": modifier,
            "NovoValor": new_value,
        }

    def _execute_simple_use(self, entity: EntityInstance, params: dict) -> dict:
        status = params.get("AtributoAlvo")
        if status not in STATUSES:
            raise EngineError("VALOR_INVALIDO", "USO_SIMPLES requires a valid status target.")
        modifier = int(self.calcular_expressao(str(params.get("Modificador", 0))))
        old = entity.data[status]
        new = self.alterar_status(entity.instance_id, status, modifier)
        return {
            "Sucesso": True,
            "Tipo": "USO_SIMPLES",
            "AtributoAlvo": status,
            "Dado": modifier,
            "HPAnterior": old if status == "HP" else None,
            "NovoValor": new,
        }

    def _execute_target_effect(self, source_entity: EntityInstance, params: dict, args: dict) -> dict:
        target_id = args.get("Alvo", params.get("Alvo"))
        if target_id is None:
            raise EngineError("ALVO_INVALIDO", "EFEITO_EM_ALVO requires Alvo.")

        target = self._require_instance(str(target_id))
        target_status = params.get("AtributoAlvo")
        if target_status not in STATUSES:
            raise EngineError("VALOR_INVALIDO", "Target effect requires a valid status target.")

        modifier = int(self.calcular_expressao(str(params.get("Modificador", 0))))
        previous = int(target.data[target_status])
        new = self.alterar_status(target.instance_id, target_status, modifier)

        return {
            "Sucesso": True,
            "Tipo": "EFEITO_EM_ALVO",
            "Alvo": target.instance_id,
            "AtributoAlvo": target_status,
            "Dado": abs(modifier),
            "Modificador": modifier,
            "HPAnterior": previous if target_status == "HP" else None,
            "HPAtual": new if target_status == "HP" else None,
            "NovoValor": new,
        }

    def _execute_position_effect(self, source_entity: EntityInstance, params: dict, args: dict) -> dict:
        x = args.get("X", params.get("X"))
        y = args.get("Y", params.get("Y"))
        current = self._require_current_map()
        self._require_position(x, y, current)

        target_status = params.get("AtributoAlvo")
        modifier_expression = str(params.get("Modificador", 0))
        affected = []

        for target in current.entity_instances.values():
            if target.x == x and target.y == y and target.data["Tipo"] in {"PLAYER", "NPC"}:
                if target_status not in STATUSES:
                    raise EngineError("VALOR_INVALIDO", "Position effect requires a valid status target.")
                modifier = int(self.calcular_expressao(modifier_expression))
                previous = int(target.data[target_status])
                new = self.alterar_status(target.instance_id, target_status, modifier)
                affected.append({
                    "InstanciaID": target.instance_id,
                    "Anterior": previous,
                    "Atual": new,
                })

        return {
            "Sucesso": True,
            "Tipo": "EFEITO_EM_POSICAO",
            "X": x,
            "Y": y,
            "Afetados": affected,
        }

    # -----------------------------------------------------------------------
    # Conditions
    # -----------------------------------------------------------------------

    def aplicar_condicao(
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
                "Nome": condition,
                **deep_copy(config or {}),
            }

        name = definition.get("Nome")
        if not name:
            raise EngineError("VALOR_INVALIDO", "Condition must have a Nome.")

        duration = definition.get("Duracao")
        if duration is not None:
            if not isinstance(duration, int) or duration < 0:
                raise EngineError("VALOR_INVALIDO", "Condition duration must be a non-negative integer.")

        duration_unit = definition.get("UnidadeDuracao", "ACOES")
        if duration_unit not in CONDITION_DURATION_UNITS:
            raise EngineError("VALOR_INVALIDO", "Invalid condition duration unit.")

        effect = deep_copy(definition.get("Efeito", {}))
        rules = deep_copy(definition.get("Regras", {}))

        # Explicit condition definitions from the specification.
        defaults = {
            "PegandoFogo": {"Duracao": 5, "UnidadeDuracao": "ACOES", "Efeito": {"HP": "-1d4"}},
            "Congelado": {"Duracao": None, "UnidadeDuracao": "INDEFINIDO", "Efeito": {},
                          "Regras": {"BloqueiaMovimento": True, "BloqueiaAcoes": True}},
            "Envenenado": {"Duracao": 3, "UnidadeDuracao": "ACOES", "Efeito": {"HP": "-1d8"}},
            "Paralisado": {"Duracao": None, "UnidadeDuracao": "INDEFINIDO", "Efeito": {},
                           "Regras": {"BloqueiaMovimento": True, "BloqueiaAcoes": True}},
            "Desmaiado": {"Duracao": None, "UnidadeDuracao": "INDEFINIDO", "Efeito": {},
                          "Regras": {"BloqueiaMovimento": True, "BloqueiaAcoes": True}},
            "Enlouquecido": {"Duracao": None, "UnidadeDuracao": "INDEFINIDO", "Efeito": {},
                             "Regras": {"PerdeControle": True}},
        }

        if name in defaults:
            default = defaults[name]
            if duration is None:
                duration = default["Duracao"]
            if duration_unit == "ACOES" and default["UnidadeDuracao"] != "ACOES":
                duration_unit = default["UnidadeDuracao"]
            if not effect:
                effect = default["Efeito"]
            merged_rules = deep_copy(default.get("Regras", {}))
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
        if name in {"PegandoFogo", "Congelado"}:
            opposite = "Congelado" if name == "PegandoFogo" else "PegandoFogo"
            for cid, active in list(entity.conditions.items()):
                if active.name == opposite:
                    self._remove_condition_internal(entity, cid)
                    return {
                        "Sucesso": True,
                        "Condicao": name,
                        "Interacao": f"{name}+{opposite} removed both conditions.",
                    }

        entity.conditions[condition_id] = instance
        self._conditions[condition_id] = (instance_id, instance)

        return {
            "Sucesso": True,
            "CondicaoID": condition_id,
            "Nome": name,
            "Duracao": duration,
            "UnidadeDuracao": duration_unit,
        }

    def remover_condicao(self, instance_id: str, condition_id: str) -> dict:
        entity = self._require_instance(instance_id)
        if condition_id not in entity.conditions:
            raise EngineError("ID_INEXISTENTE", f"Condition {condition_id} is not active.")
        self._remove_condition_internal(entity, condition_id)
        return success()

    def _remove_condition_internal(self, entity: EntityInstance, condition_id: str):
        entity.conditions.pop(condition_id, None)
        self._conditions.pop(condition_id, None)

    def listar_condicoes(self, instance_id: str) -> list[dict]:
        entity = self._require_instance(instance_id)
        return [
            {
                "CondicaoID": c.condition_id,
                "Nome": c.name,
                "Duracao": c.duration,
                "UnidadeDuracao": c.duration_unit,
                "Efeito": deep_copy(c.effect),
                "Regras": deep_copy(c.rules),
            }
            for c in entity.conditions.values()
        ]

    def alterar_duracao_condicao(self, condition_id: str, duration: Optional[int]) -> dict:
        if condition_id not in self._conditions:
            raise EngineError("ID_INEXISTENTE", f"Condition {condition_id} does not exist.")

        if duration is not None and (not isinstance(duration, int) or duration < 0):
            raise EngineError("VALOR_INVALIDO", "Duration must be a non-negative integer or None.")

        _, condition = self._conditions[condition_id]
        condition.duration = duration
        return success(duration)

    def _has_blocking_condition(self, entity: EntityInstance) -> bool:
        return any(
            c.rules.get("BloqueiaAcoes") or c.rules.get("BloqueiaMovimento")
            for c in entity.conditions.values()
        )

    def _advance_action(self, instance_id: str):
        """Process action-based condition effects and durations."""
        entity = self._require_instance(instance_id)

        for condition_id, condition in list(entity.conditions.items()):
            # Conditions with HP effects apply once per action.
            if condition.duration_unit == "ACOES":
                hp_expression = condition.effect.get("HP")
                if hp_expression:
                    damage_or_heal = int(self.calcular_expressao(str(hp_expression)))
                    self.alterar_status(instance_id, "HP", damage_or_heal)

                if condition.duration is not None:
                    condition.duration -= 1
                    if condition.duration <= 0:
                        self._remove_condition_internal(entity, condition_id)

    # -----------------------------------------------------------------------
    # Convenience combat rule
    # -----------------------------------------------------------------------

    def atacar(
        self,
        attacker_id: str,
        target_id: str,
        attack_modifier: int = 0,
        damage_expression: str = "1d4",
    ) -> dict:
        attacker = self._require_instance(attacker_id)
        target = self._require_instance(target_id)

        if attacker.data["Tipo"] == "STATIC" or target.data["Tipo"] == "STATIC":
            raise EngineError("ALVO_INVALIDO", "Static entities cannot participate in this combat attack.")

        if self._has_blocking_condition(attacker):
            raise EngineError("ACAO_INDISPONIVEL", "Attacker cannot act.")

        roll = self.rolar_dado(20)
        total_attack = roll + attack_modifier
        hit = total_attack >= int(target.data["AC"])

        result = {
            "Acertou": hit,
            "Rolagem": roll,
            "ModificadorAtaque": attack_modifier,
            "ResultadoAtaque": total_attack,
            "ACAlvo": target.data["AC"],
        }

        if hit:
            damage = int(self.calcular_expressao(damage_expression))
            hp_before = int(target.data["HP"])
            hp_after = self.alterar_status(target_id, "HP", -damage)
            result.update({
                "Dano": damage,
                "HPAnterior": hp_before,
                "HPAtual": hp_after,
            })
        else:
            result.update({
                "Dano": 0,
                "HPAnterior": target.data["HP"],
                "HPAtual": target.data["HP"],
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
            return EngineError("ARGUMENTO_INVALIDO", "Command name must be a string.").to_dict()

        commands = {
            "rolar_dado": self.rolar_dado,
            "calcular_expressao": self.calcular_expressao,
            "calcular_distancia": self.calcular_distancia,
            "criar_mapa": self.criar_mapa,
            "popular_mapa": self.popular_mapa,
            "listar_mapas": self.listar_mapas,
            "consultar_mapa_atual": self.consultar_mapa_atual,
            "consultar_mapa_atual_id": self.consultar_mapa_atual_id,
            "definir_mapa_atual": self.definir_mapa_atual,
            "criar_entidade": self.criar_entidade,
            "criar_entidade_no_mapa": self.criar_entidade_no_mapa,
            "listar_entidades_no_banco": self.listar_entidades_no_banco,
            "listar_entidades_no_mapa": self.listar_entidades_no_mapa,
            "remover_entidade_do_mapa": self.remover_entidade_do_mapa,
            "listar_acoes_da_entidade": self.listar_acoes_da_entidade,
            "manipular_entidade": self.manipular_entidade,
            "aplicar_teste": self.aplicar_teste,
            "consultar_dados_basicos": self.consultar_dados_basicos,
            "consultar_dado_basico": self.consultar_dado_basico,
            "definir_posicao": self.definir_posicao,
            "alterar_status": self.alterar_status,
            "alterar_atributo": self.alterar_atributo,
            "inserir_item_no_inventario": self.inserir_item_no_inventario,
            "remover_item_do_inventario": self.remover_item_do_inventario,
            "listar_itens_no_inventario": self.listar_itens_no_inventario,
            "listar_habilidades": self.listar_habilidades,
            "listar_itens_no_banco": self.listar_itens_no_banco,
            "listar_habilidades_no_banco": self.listar_habilidades_no_banco,
            "consultar_item_ou_habilidade": self.consultar_item_ou_habilidade,
            "criar_item_ou_habilidade": self.criar_item_ou_habilidade,
            "atribuir_habilidade": self.atribuir_habilidade,
            "usar_item_ou_habilidade": self.usar_item_ou_habilidade,
            "usar_habilidade": self.usar_habilidade,
            "aplicar_condicao": self.aplicar_condicao,
            "remover_condicao": self.remover_condicao,
            "listar_condicoes": self.listar_condicoes,
            "alterar_duracao_condicao": self.alterar_duracao_condicao,
            "atacar": self.atacar,
        }

        function = commands.get(command_name)
        if function is None:
            return EngineError("ACAO_INEXISTENTE", f"Unknown engine command: {command_name}.").to_dict()

        try:
            value = function(**kwargs)
            if isinstance(value, dict) and "Sucesso" in value:
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

    map_id = engine.criar_mapa(10, 8, terrain)
    engine.popular_mapa(10, 8, [
        ["NONE", "NONE", "NONE", "NONE", "NONE", "NONE", "NONE", "NONE", "NONE", "NONE"],
        ["NONE", "HOUSE", "HOUSE", "NONE", "NONE", "NONE", "NONE", "NONE", "NONE", "NONE"],
        ["NONE", "HOUSE", "HOUSE", "NONE", "NONE", "NONE", "NONE", "NONE", "STORE", "NONE"],
        ["NONE", "NONE", "NONE", "NONE", "NONE", "NONE", "NONE", "NONE", "NONE", "NONE"],
        ["NONE", "NONE", "NONE", "STATUE", "NONE", "NONE", "NONE", "NONE", "NONE", "NONE"],
        ["NONE", "NONE", "NONE", "NONE", "NONE", "NONE", "NONE", "NONE", "NONE", "NONE"],
        ["NONE", "NONE", "NONE", "NONE", "NONE", "NONE", "NONE", "NONE", "NONE", "NONE"],
        ["NONE", "NONE", "NONE", "NONE", "NONE", "NONE", "NONE", "NONE", "NONE", "NONE"],
    ])

    goblin_id = engine.criar_entidade({
        "Tipo": "NPC",
        "Nome": "Goblin",
        "HPMax": 10,
        "HP": 10,
        "AC": 12,
        "XP": 20,
        "Atributos": {"FOR": 2, "AGI": 3, "INT": 0, "PRE": 0, "CONS": 1},
        "Proficiencias": {
            "Sobrevivencia": {
                "Modificador": 2,
                "Contexto": "situacoes relacionadas a sobrevivencia",
            }
        },
    })

    hero_id = engine.criar_entidade({
        "Tipo": "PLAYER",
        "Nome": "Adventurer",
        "HPMax": 30,
        "HP": 30,
        "AC": 14,
        "XP": 20,
        "Atributos": {"FOR": 4, "AGI": 3, "INT": 2, "PRE": 1, "CONS": 4},
        "Proficiencias": {
            "Prestidigitação": {
                "Modificador": 2,
                "Contexto": "manipulacao manual furtos truques",
            }
        },
    })

    statue_id = engine.criar_entidade({
        "Tipo": "STATIC",
        "Nome": "Ancient Statue",
    })

    goblin = engine.criar_entidade_no_mapa(goblin_id, 7, 5)
    hero = engine.criar_entidade_no_mapa(hero_id, 2, 6)
    engine.criar_entidade_no_mapa(statue_id, 3, 4)

    dagger_id = engine.criar_item_ou_habilidade({
        "Tipo": "ITEM",
        "Nome": "Dagger",
        "Raridade": 1,
        "Funcionalidade": "EFEITO_EM_ALVO",
        "QuantidadeUsos": None,
        "Parametros": {
            "AtributoAlvo": "HP",
            "Modificador": "-1d4",
        },
    })

    potion_id = engine.criar_item_ou_habilidade({
        "Tipo": "ITEM",
        "Nome": "Healing Potion",
        "Raridade": 1,
        "Funcionalidade": "USO_SIMPLES",
        "QuantidadeUsos": 1,
        "Parametros": {
            "AtributoAlvo": "HP",
            "Modificador": "2d4",
        },
    })

    fireball_id = engine.criar_item_ou_habilidade({
        "Tipo": "HABILIDADE",
        "Nome": "Fire Bolt",
        "Raridade": 1,
        "Funcionalidade": "EFEITO_EM_ALVO",
        "QuantidadeUsos": 10,
        "Parametros": {
            "AtributoAlvo": "HP",
            "Modificador": "-1d6",
        },
    })

    engine.inserir_item_no_inventario(hero, dagger_id)
    engine.inserir_item_no_inventario(hero, potion_id)
    engine.atribuir_habilidade(hero, fireball_id)

    return engine
