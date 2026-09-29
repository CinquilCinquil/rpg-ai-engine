"""Small headless example of the Engine API."""

from engine import Engine

engine = Engine(seed=42)

map_id = engine.criar_mapa(
    4,
    4,
    [
        ["EARTH", "WATER", "EARTH", "EARTH"],
        ["EARTH", "WATER", "EARTH", "EARTH"],
        ["EARTH", "WATER", "WATER", "EARTH"],
        ["EARTH", "SAND", "EARTH", "EARTH"],
    ],
)

goblin_model = engine.criar_entidade({
    "Tipo": "NPC",
    "Nome": "Goblin",
    "HPMax": 10,
    "HP": 10,
    "AC": 12,
    "XP": 20,
    "Atributos": {
        "FOR": 2,
        "AGI": 3,
        "INT": 0,
        "PRE": 0,
        "CONS": 1,
    },
})

goblin = engine.criar_entidade_no_mapa(goblin_model, 0, 0)

print("Map:", engine.consultar_mapa_atual())
print("Goblin:", engine.consultar_dados_basicos(goblin))
print("Test:", engine.aplicar_teste(goblin, "AGI"))

# The same model can create another independent instance.
goblin_2 = engine.criar_entidade_no_mapa(goblin_model, 3, 3)
engine.alterar_status(goblin, "HP", -5)

print("First HP:", engine.consultar_dado_basico(goblin, "HP"))
print("Second HP:", engine.consultar_dado_basico(goblin_2, "HP"))
