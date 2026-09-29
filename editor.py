from __future__ import annotations

import pygame

from engine import EngineError, create_demo_engine


# ---------------------------------------------------------------------------
# UI constants
# ---------------------------------------------------------------------------

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 800
PANEL_WIDTH = 360
TOP_BAR = 58
BOTTOM_BAR = 34

BG = (18, 20, 25)
PANEL = (27, 30, 37)
PANEL_ALT = (34, 38, 46)
GRID = (68, 72, 82)
TEXT = (235, 237, 242)
MUTED = (155, 160, 171)
ACCENT = (96, 165, 250)
ERROR = (239, 96, 96)
SUCCESS = (105, 210, 145)

TERRAIN_COLORS = {
    "EARTH": (96, 83, 65),
    "SAND": (190, 166, 105),
    "WATER": (61, 117, 160),
}

STRUCTURE_MARKERS = {
    "HOUSE": "H",
    "STORE": "$",
    "STATUE": "S",
}

ENTITY_COLORS = {
    "PLAYER": (80, 210, 145),
    "NPC": (226, 92, 92),
    "STATIC": (185, 185, 195),
}


class RPGEditor:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("AI RPG Engine - Pygame Editor")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont("consolas", 16)
        self.small_font = pygame.font.SysFont("consolas", 13)
        self.title_font = pygame.font.SysFont("consolas", 20, bold=True)
        self.big_font = pygame.font.SysFont("consolas", 26, bold=True)

        self.engine = create_demo_engine()

        self.selected_entity: str | None = None
        self.selected_cell: tuple[int, int] | None = None
        self.terrain_brush = "EARTH"
        self.show_debug = True
        self.message = "Ready."
        self.message_kind = "normal"
        self.message_timer = 0.0

        self.camera_x = 0
        self.camera_y = 0

        self.goblins_by_model = [
            model_id
            for model_id in self.engine.list_entity_models()
            if self.engine.entity_models[model_id].data.get("Name") == "Goblin"
        ]

    # -----------------------------------------------------------------------
    # Geometry
    # -----------------------------------------------------------------------

    def map_rect(self):
        return pygame.Rect(
            0,
            TOP_BAR,
            self.screen.get_width() - PANEL_WIDTH,
            self.screen.get_height() - TOP_BAR - BOTTOM_BAR,
        )

    def cell_size(self):
        current = self.engine.maps[self.engine.current_map_id]
        rect = self.map_rect()
        return min(rect.width // current.width, rect.height // current.height)

    def grid_origin(self):
        rect = self.map_rect()
        size = self.cell_size()
        width = self.engine.maps[self.engine.current_map_id].width * size
        height = self.engine.maps[self.engine.current_map_id].height * size
        return (
            rect.left + (rect.width - width) // 2,
            rect.top + (rect.height - height) // 2,
        )

    def cell_from_mouse(self, position):
        mouse_x, mouse_y = position
        current = self.engine.maps[self.engine.current_map_id]
        size = self.cell_size()
        origin_x, origin_y = self.grid_origin()
        x = (mouse_x - origin_x) // size
        y = (mouse_y - origin_y) // size
        if 0 <= x < current.width and 0 <= y < current.height:
            return int(x), int(y)
        return None

    def cell_rect(self, x, y):
        size = self.cell_size()
        ox, oy = self.grid_origin()
        return pygame.Rect(ox + x * size, oy + y * size, size, size)

    # -----------------------------------------------------------------------
    # Messaging
    # -----------------------------------------------------------------------

    def notify(self, text, kind="normal"):
        self.message = text
        self.message_kind = kind
        self.message_timer = 4.0

    def handle_error(self, exc):
        if isinstance(exc, EngineError):
            self.notify(exc.message, "error")
        else:
            self.notify(str(exc), "error")

    # -----------------------------------------------------------------------
    # Rendering
    # -----------------------------------------------------------------------

    def draw(self):
        self.screen.fill(BG)
        self.draw_top_bar()
        self.draw_map()
        self.draw_side_panel()
        self.draw_bottom_bar()
        pygame.display.flip()

    def draw_top_bar(self):
        pygame.draw.rect(self.screen, PANEL, (0, 0, self.screen.get_width(), TOP_BAR))
        title = self.title_font.render("AI RPG ENGINE", True, TEXT)
        self.screen.blit(title, (18, 15))

        current_id = self.engine.get_current_map_id()
        map_state = self.engine.maps[current_id]
        info = f"Map #{current_id}  {map_state.width}x{map_state.height}"
        text = self.font.render(info, True, MUTED)
        self.screen.blit(text, (235, 19))

        brush = f"Terrain brush: {self.terrain_brush}"
        text = self.font.render(brush, True, ACCENT)
        self.screen.blit(text, (530, 19))

        debug = "DEBUG ON" if self.show_debug else "DEBUG OFF"
        text = self.font.render(debug, True, SUCCESS if self.show_debug else MUTED)
        self.screen.blit(text, (780, 19))

    def draw_map(self):
        current = self.engine.maps[self.engine.current_map_id]

        for y in range(current.height):
            for x in range(current.width):
                rect = self.cell_rect(x, y)
                terrain = current.terrain[y][x]
                color = TERRAIN_COLORS.get(terrain, (80, 80, 80))
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, GRID, rect, 1)

                structure = current.structures[y][x]
                if structure != "NONE":
                    marker = STRUCTURE_MARKERS.get(structure, "?")
                    label = self.font.render(marker, True, (30, 30, 30))
                    self.screen.blit(label, label.get_rect(center=rect.center))

        # Selection.
        if self.selected_cell:
            x, y = self.selected_cell
            rect = self.cell_rect(x, y)
            pygame.draw.rect(self.screen, (255, 255, 255), rect, 3)

        # Entities.
        for instance in current.entity_instances.values():
            rect = self.cell_rect(instance.x, instance.y)
            center = rect.center
            radius = max(9, self.cell_size() // 4)
            color = ENTITY_COLORS.get(instance.data["Type"], (255, 255, 255))
            pygame.draw.circle(self.screen, color, center, radius)

            label = self.small_font.render(instance.data["Name"][:8], True, (15, 15, 18))
            self.screen.blit(label, label.get_rect(center=center))

            if instance.instance_id == self.selected_entity:
                pygame.draw.circle(self.screen, (255, 255, 255), center, radius + 5, 2)

    def draw_side_panel(self):
        x = self.screen.get_width() - PANEL_WIDTH
        pygame.draw.rect(self.screen, PANEL, (x, TOP_BAR, PANEL_WIDTH, self.screen.get_height() - TOP_BAR - BOTTOM_BAR))

        cursor_y = TOP_BAR + 18
        heading = self.title_font.render("INSPECTOR", True, TEXT)
        self.screen.blit(heading, (x + 18, cursor_y))
        cursor_y += 38

        current = self.engine.maps[self.engine.current_map_id]
        selected = self.engine._instances.get(self.selected_entity) if self.selected_entity else None

        if selected:
            cursor_y = self.draw_entity_inspector(x + 18, cursor_y, selected)
        elif self.selected_cell:
            cx, cy = self.selected_cell
            cursor_y = self.draw_cell_inspector(x + 18, cursor_y, current, cx, cy)
        else:
            cursor_y = self.draw_help(x + 18, cursor_y)

        if self.show_debug:
            cursor_y += 12
            pygame.draw.line(
                self.screen,
                GRID,
                (x + 18, cursor_y),
                (self.screen.get_width() - 18, cursor_y),
            )
            cursor_y += 12

            debug_lines = [
                f"Entities: {len(current.entity_instances)}",
                f"Entity models: {len(self.engine.entity_models)}",
                f"Item models: {len(self.engine.item_models)}",
                f"Selected: {self.selected_entity or '-'}",
            ]
            for line in debug_lines:
                text = self.small_font.render(line, True, MUTED)
                self.screen.blit(text, (x + 18, cursor_y))
                cursor_y += 19

    def draw_entity_inspector(self, x, y, entity):
        lines = [
            (entity.data["Name"], TEXT, self.big_font),
            (f"Instance: {entity.instance_id}", MUTED, self.small_font),
            (f"Model ID: {entity.model_id}", MUTED, self.small_font),
            (f"Type: {entity.data['Type']}", TEXT, self.font),
            (f"Position: ({entity.x}, {entity.y})", TEXT, self.font),
        ]

        if entity.data["Type"] in {"PLAYER", "NPC"}:
            lines.extend([
                (f"HP: {entity.data['HP']} / {entity.data['HPMax']}", TEXT, self.font),
                (f"AC: {entity.data['AC']}   XP: {entity.data['XP']}", TEXT, self.font),
            ])

            attributes = "  ".join(
                f"{key}:{value}" for key, value in entity.data["Attributes"].items()
            )
            lines.append((attributes, TEXT, self.small_font))

            conditions = entity.conditions
            lines.append((f"Conditions: {len(conditions)}", TEXT, self.font))

            if conditions:
                for condition in conditions.values():
                    duration = (
                        "∞"
                        if condition.duration is None
                        else f"{condition.duration} {condition.duration_unit.lower()}"
                    )
                    lines.append((f"  {condition.name} ({duration})", (235, 180, 100), self.small_font))

            lines.append((f"Inventory: {len(entity.inventory)}", TEXT, self.font))
            lines.append((f"Abilities: {len(entity.abilities)}", TEXT, self.font))

        for content, color, font in lines:
            surface = font.render(content, True, color)
            self.screen.blit(surface, (x, y))
            y += font.get_height() + 7

        return y

    def draw_cell_inspector(self, x, y, current, cx, cy):
        lines = [
            (f"CELL ({cx}, {cy})", self.big_font, TEXT),
            (f"Terrain: {current.terrain[cy][cx]}", self.font, TEXT),
            (f"Structure: {current.structures[cy][cx]}", self.font, TEXT),
        ]
        entities = [
            entity
            for entity in current.entity_instances.values()
            if entity.x == cx and entity.y == cy
        ]
        lines.append((f"Entities: {len(entities)}", self.font, TEXT))
        for entity in entities:
            lines.append((f"  {entity.data['Name']} [{entity.instance_id}]", self.small_font, MUTED))

        for content, font, color in lines:
            surface = font.render(content, True, color)
            self.screen.blit(surface, (x, y))
            y += font.get_height() + 8

        return y

    def draw_help(self, x, y):
        title = self.font.render("Editor controls", True, TEXT)
        self.screen.blit(title, (x, y))
        y += 30

        controls = [
            "Left click  Select cell/entity",
            "Right click Move selected entity",
            "1 / 2 / 3    Terrain brush",
            "H / S / T    Structures",
            "N            Spawn Goblin",
            "D            Damage selected",
            "F            Apply fire",
            "R            Remove conditions",
            "C            Toggle debug",
            "X            Clear selections"
        ]
        for line in controls:
            surface = self.small_font.render(line, True, MUTED)
            self.screen.blit(surface, (x, y))
            y += 20
        return y

    def draw_bottom_bar(self):
        y = self.screen.get_height() - BOTTOM_BAR
        pygame.draw.rect(self.screen, PANEL, (0, y, self.screen.get_width(), BOTTOM_BAR))

        color = ERROR if self.message_kind == "error" else SUCCESS if self.message_kind == "success" else MUTED
        text = self.small_font.render(self.message, True, color)
        self.screen.blit(text, (14, y + 9))

    # -----------------------------------------------------------------------
    # Input
    # -----------------------------------------------------------------------

    def handle_event(self, event):
        if event.type == pygame.QUIT:
            return False

        if event.type == pygame.VIDEORESIZE:
            self.screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)
            return True

        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                cell = self.cell_from_mouse(event.pos)
                if cell:
                    self.select_cell(cell)
            elif event.button == 3:
                cell = self.cell_from_mouse(event.pos)
                if cell and self.selected_entity:
                    try:
                        self.engine.set_position(self.selected_entity, *cell)
                        self.selected_cell = cell
                        self.notify(f"Moved {self.selected_entity} to {cell}.", "success")
                    except Exception as exc:
                        self.handle_error(exc)
            return True

        if event.type == pygame.KEYDOWN:
            return self.handle_key(event.key)

        return True

    def select_cell(self, cell):
        self.selected_cell = cell
        current = self.engine.maps[self.engine.current_map_id]
        entities = [
            e for e in current.entity_instances.values()
            if (e.x, e.y) == cell
        ]

        if entities:
            self.selected_entity = entities[0].instance_id
            self.notify(f"Selected {entities[0].data['Name']} ({self.selected_entity}).")
        else:
            self.selected_entity = None
            self.notify(f"Selected cell {cell}.")

    def handle_key(self, key):
        if key == pygame.K_ESCAPE:
            if self.selected_entity:
                self.selected_entity = None
                self.notify("Entity selection cleared.")
            else:
                return False

        elif key == pygame.K_c:
            self.show_debug = not self.show_debug

        elif key == pygame.K_x:
            self.selected_entity = None
            self.selected_cell = None

        elif key == pygame.K_1:
            self.terrain_brush = "EARTH"
            self.paint_selected_cell()

        elif key == pygame.K_2:
            self.terrain_brush = "SAND"
            self.paint_selected_cell()

        elif key == pygame.K_3:
            self.terrain_brush = "WATER"
            self.paint_selected_cell()

        elif key in (pygame.K_h, pygame.K_s, pygame.K_t):
            structure = {
                pygame.K_h: "HOUSE",
                pygame.K_s: "STORE",
                pygame.K_t: "STATUE",
            }[key]
            self.toggle_structure(structure)

        elif key == pygame.K_n:
            self.spawn_goblin()

        elif key == pygame.K_d:
            self.damage_selected()

        elif key == pygame.K_f:
            self.apply_fire()

        elif key == pygame.K_r:
            self.remove_conditions()

        return True

    # -----------------------------------------------------------------------
    # Editor commands
    # -----------------------------------------------------------------------

    def paint_selected_cell(self):
        if not self.selected_cell:
            self.notify(f"Brush selected: {self.terrain_brush}.")
            return

        x, y = self.selected_cell
        current = self.engine.maps[self.engine.current_map_id]

        # Direct world editing is an engine command in this editor because
        # terrain painting is part of the Master's world-authoring role.
        current.terrain[y][x] = self.terrain_brush
        self.notify(f"Terrain at ({x}, {y}) set to {self.terrain_brush}.", "success")

    def toggle_structure(self, structure):
        if not self.selected_cell:
            self.notify("Select a cell first.", "error")
            return

        x, y = self.selected_cell
        current = self.engine.maps[self.engine.current_map_id]
        current.structures[y][x] = "NONE" if current.structures[y][x] == structure else structure
        self.notify(f"Structure at ({x}, {y}): {current.structures[y][x]}.", "success")

    def spawn_goblin(self):
        if not self.selected_cell:
            self.notify("Select a cell first.", "error")
            return
        if not self.goblins_by_model:
            self.notify("Goblin model not found.", "error")
            return

        try:
            instance_id = self.engine.spawn_entity(
                self.goblins_by_model[0],
                *self.selected_cell,
            )
            self.selected_entity = instance_id
            self.notify(f"Spawned Goblin {instance_id}.", "success")
        except Exception as exc:
            self.handle_error(exc)

    def damage_selected(self):
        if not self.selected_entity:
            self.notify("Select an entity first.", "error")
            return

        try:
            entity = self.engine._instances[self.selected_entity]
            if entity.data["Type"] == "STATIC":
                self.notify("Static entities do not have HP.", "error")
                return
            damage = self.engine.evaluate_expression("1d6")
            hp_before = entity.data["HP"]
            hp_after = self.engine.modify_status(self.selected_entity, "HP", -int(damage))
            self.notify(f"{entity.data['Name']} took {damage} damage: {hp_before} -> {hp_after}.", "success")
        except Exception as exc:
            self.handle_error(exc)

    def apply_fire(self):
        if not self.selected_entity:
            self.notify("Select an entity first.", "error")
            return
        try:
            result = self.engine.apply_condition(self.selected_entity, "OnFire")
            self.notify(f"Applied OnFire: {result.get('ConditionID', 'interaction')}.", "success")
        except Exception as exc:
            self.handle_error(exc)

    def remove_conditions(self):
        if not self.selected_entity:
            self.notify("Select an entity first.", "error")
            return

        try:
            entity = self.engine._instances[self.selected_entity]
            for condition_id in list(entity.conditions):
                self.engine.remove_condition(self.selected_entity, condition_id)
            self.notify("All conditions removed.", "success")
        except Exception as exc:
            self.handle_error(exc)

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------

    def run(self):
        running = True
        while running:
            dt = self.clock.tick(60) / 1000.0
            self.message_timer = max(0.0, self.message_timer - dt)

            for event in pygame.event.get():
                running = self.handle_event(event)
                if not running:
                    break

            self.draw()

        pygame.quit()