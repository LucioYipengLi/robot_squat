"""Newton GL input and ImGui panel. All upstream viewer internals live here."""

from .wrist_controls import POSE_KEYS


class NewtonInput:
    def __init__(self, sim, controller):
        visualizers = [v for v in sim.visualizers if v.cfg.visualizer_type == "newton_gl"]
        if len(visualizers) != 1:
            raise ValueError("Upper teleop requires exactly one Newton GL visualizer (--viz newton_gl)")
        self.viewer = getattr(visualizers[0], "_viewer", None)
        self.gui = getattr(self.viewer, "gui", None)
        if self.gui is None or not callable(getattr(self.viewer, "is_key_down", None)):
            raise RuntimeError("Newton GL viewer has no interactive keyboard/ImGui interface")
        self.window = self.viewer.renderer.window
        if not isinstance(getattr(self.gui, "_ui_callbacks", None), dict):
            raise RuntimeError("Unsupported Newton UI callback registry; review the project adapter")
        self.controller = controller
        self._closed = False
        self._focused = True
        self._held, self._blocked = set(), set()
        self._repeat = {}
        self._clock = 0.0
        self._keys = set(POSE_KEYS) | {"1", "2", "3", "N", "R", "T", "M", "G"}
        self._handlers = {"on_deactivate": self._deactivate, "on_activate": self._activate}
        self._panel = self.draw
        self.window.push_handlers(**self._handlers)
        try:
            self.viewer.register_ui_callback(self._panel, position="side")
        except BaseException:
            self.close()
            raise

    def _deactivate(self):
        self._focused = False
        self._blocked |= self._held
        self._held.clear()

    def _activate(self):
        self._focused = True
        self._blocked |= {key for key in self._keys if self.viewer.is_key_down(key)}

    def poll(self, dt):
        if self._closed:
            return
        held = {key for key in self._keys if self.viewer.is_key_down(key)}
        self._blocked &= held
        if not self._focused or self.gui.is_keyboard_capturing() or self.viewer.is_paused():
            self._blocked |= held
            self._held.clear()
            self._repeat.clear()
            return
        held -= self._blocked
        self._clock += dt
        for key in sorted(held):
            initial = key not in self._held
            repeat = key in POSE_KEYS and self._clock >= self._repeat.get(key, 0)
            if initial or repeat:
                self.controller.key(key, rotate=self.viewer.is_key_down("shift"))
                self._repeat[key] = self._clock + self.controller.cfg.key_repeat_s
        self._held = held

    def draw(self, ui):
        if self._closed:
            return
        c = self.controller
        ui.set_next_item_open(True, ui.Cond_.once)
        if not ui.collapsing_header("Upper wrist teleop"):
            return
        changed, index = ui.combo("Control", int(c.source == "auto"), ["Manual", "Automatic"])
        if changed:
            c.source = ["manual", "auto"][index]
            if c.source == "manual" and c.arm_mode is None:
                c.arm_mode = 2
            c.request("configure")
        arms = ["Left", "Right", "Both"] + (["Sample"] if c.source == "auto" else [])
        changed, index = ui.combo("Active arms", c.arm_mode if c.arm_mode is not None else 3, arms)
        if changed:
            c.arm_mode = index if index < 3 else None
            c.request("configure")
        if c.source == "auto":
            kinds = c.available_categories
            changed, index = ui.combo("Task", kinds.index(c.category), kinds)
            if changed:
                c.category = kinds[index]
                c.request("configure")
        else:
            _, c.editor = ui.combo("Edit", c.editor, ["Left target", "Right target", "Pair"])
        _, c.fine = ui.checkbox("Fine increments (G)", c.fine)
        changed, c.stochastic = ui.checkbox("Training stochastic actions", c.stochastic)
        if changed:
            c.request("policy")
        for label, action in [
            ("Next / reset (N)", "next"),
            ("Retry target (R)", "retry"),
            ("Current wrists (T)", "restore"),
        ]:
            if ui.button(label):
                c.request(action)
        ui.text_wrapped("I/K: Y, J/L: X, U/O: Z. Hold Shift to rotate. 1/2/3: arms. M: manual/auto.")
        ui.text_wrapped("World frame; position in m, orientation in deg. Space pauses the viewer.")
        for side, label in enumerate(("Left", "Right")):
            ui.begin_disabled(c.source != "manual" or c.arm_mode not in (side, 2))
            p_changed, p = ui.input_float3(label + " XYZ", c.poses[side, :3].tolist())
            r_changed, r = ui.input_float3(label + " RPY", c.angles_deg(side))
            if p_changed or r_changed:
                c.set_pose(side, p, r)
            ui.end_disabled()
        for name, value in c.status.items():
            ui.text_wrapped(f"{name}: {value}")

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._held.clear()
        self._repeat.clear()
        try:
            self.window.remove_handlers(**self._handlers)
        finally:
            for callbacks in self.gui._ui_callbacks.values():
                while self._panel in callbacks:
                    callbacks.remove(self._panel)
