"""Kit keyboard/gamepad subscriptions, created only after simulation launch."""

import logging
import weakref
from contextlib import ExitStack

from .controls import HELP_TEXT

logger = logging.getLogger(__name__)


class KitInput:
    def __init__(self, controller, *, devices="both"):
        import carb.input
        import omni.appwindow

        self.controller = controller
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._gamepad = None
        self._keyboard_sub = self._gamepad_sub = None
        self._carb = carb
        window = omni.appwindow.get_default_app_window()
        if window is None:
            raise RuntimeError("Teleop requires a visible Kit window (--viz kit)")
        self._window = window
        proxy = weakref.proxy(self)
        try:
            if devices in ("keyboard", "both"):
                self._keyboard = window.get_keyboard()
                if self._keyboard is not None:
                    self._keyboard_sub = self._input.subscribe_to_keyboard_events(
                        self._keyboard, lambda event, *args, obj=proxy: obj._on_keyboard(event)
                    )
            if devices in ("gamepad", "both"):
                self._gamepad = window.get_gamepad(0)
                if self._gamepad is not None:
                    from isaaclab.app.settings_manager import get_settings_manager

                    self._settings = get_settings_manager()
                    self._camera_control = self._settings.get("/persistent/app/omniverse/gamepadCameraControl", True)
                    self._settings.set("/persistent/app/omniverse/gamepadCameraControl", False)
                    self._gamepad_sub = self._input.subscribe_to_gamepad_events(
                        self._gamepad, lambda event, *args, obj=proxy: obj._on_gamepad(event)
                    )
                elif devices == "both":
                    logger.info("No gamepad connected; keyboard control remains available")
            if self._keyboard_sub is None and self._gamepad_sub is None:
                raise RuntimeError(f"No usable {devices} input device; connect it before launching teleop")
        except BaseException:
            self.close()
            raise

    def _on_keyboard(self, event):
        types = self._carb.input.KeyboardEventType
        if event.type in (types.KEY_PRESS, types.KEY_REPEAT):
            action = self.controller.key(event.input.name, repeat=event.type == types.KEY_REPEAT)
            if action == "help":
                print(HELP_TEXT)
        return True

    def _on_gamepad(self, event):
        self.controller.gamepad_event(event.input.name, float(event.value))
        return True

    def poll(self):
        if self._gamepad_sub is not None and self._window.get_gamepad(0) is None:
            self.controller.clear_gamepad()

    def close(self):
        """Release subscriptions before the environment and Kit are destroyed."""
        # Attempt every release even if one native unsubscribe raises during shutdown.
        with ExitStack() as cleanup:
            settings = self.__dict__.pop("_settings", None)
            if settings is not None:
                cleanup.callback(settings.set, "/persistent/app/omniverse/gamepadCameraControl", self._camera_control)
            if self._keyboard_sub is not None:
                cleanup.callback(self._input.unsubscribe_to_keyboard_events, self._keyboard, self._keyboard_sub)
                self._keyboard_sub = None
            if self._gamepad_sub is not None:
                cleanup.callback(self._input.unsubscribe_to_gamepad_events, self._gamepad, self._gamepad_sub)
                self._gamepad_sub = None
