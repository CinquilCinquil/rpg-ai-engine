import unittest

from engine import Engine, EngineError


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = Engine(seed=1)
        self.map_id = self.engine.criar_mapa(
            3,
            3,
            [
                ["EARTH", "EARTH", "EARTH"],
                ["EARTH", "WATER", "EARTH"],
                ["EARTH", "EARTH", "SAND"],
            ],
        )

        self.goblin_model = self.engine.criar_entidade({
            "Tipo": "NPC",
            "Nome": "Goblin",
            "HPMax": 10,
            "HP": 10,
            "AC": 12,
            "XP": 20,
            "Atributos": {"FOR": 2, "AGI": 3, "INT": 0, "PRE": 0, "CONS": 1},
        })

    def test_dice(self):
        self.assertTrue(1 <= self.engine.rolar_dado(20) <= 20)
        self.assertEqual(self.engine.calcular_expressao("1 + 1"), 2)
        self.assertEqual(self.engine.calcular_expressao("2 * (5 + 2)"), 14)

    def test_manhattan_distance(self):
        self.assertEqual(self.engine.calcular_distancia(1, 2, 4, 6), 7)

    def test_instance_state_is_independent(self):
        a = self.engine.criar_entidade_no_mapa(self.goblin_model, 0, 0)
        b = self.engine.criar_entidade_no_mapa(self.goblin_model, 2, 0)

        self.engine.alterar_status(a, "HP", -5)

        self.assertEqual(self.engine.consultar_dado_basico(a, "HP"), 5)
        self.assertEqual(self.engine.consultar_dado_basico(b, "HP"), 10)

    def test_water_is_blocked_by_default(self):
        with self.assertRaises(EngineError):
            self.engine.criar_entidade_no_mapa(self.goblin_model, 1, 1)

    def test_inventory_uses_instances(self):
        entity = self.engine.criar_entidade_no_mapa(self.goblin_model, 0, 0)
        item_model = self.engine.criar_item_ou_habilidade({
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

        item_a = self.engine.inserir_item_no_inventario(entity, item_model)
        item_b = self.engine.inserir_item_no_inventario(entity, item_model)

        self.assertNotEqual(item_a, item_b)
        self.assertEqual(len(self.engine.listar_itens_no_inventario(entity)), 2)

    def test_condition_interaction(self):
        entity = self.engine.criar_entidade_no_mapa(self.goblin_model, 0, 0)
        self.engine.aplicar_condicao(entity, "PegandoFogo")
        result = self.engine.aplicar_condicao(entity, "Congelado")

        self.assertEqual(result["Interacao"], "Congelado+PegandoFogo removed both conditions.")
        self.assertEqual(self.engine.listar_condicoes(entity), [])

    def test_structured_command_errors(self):
        response = self.engine.command("definir_mapa_atual", map_id=999)
        self.assertFalse(response["Sucesso"])
        self.assertEqual(response["Erro"]["Codigo"], "MAPA_INEXISTENTE")


if __name__ == "__main__":
    unittest.main()
