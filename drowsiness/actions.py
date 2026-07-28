from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Protocol

from .risk_controller import BuzzerMode, RiskDecision


class RiskActionSink(Protocol):
    def publish(self, decision: RiskDecision) -> None: ...

    def close(self) -> None: ...


class RiskEventPublisher:
    """Fan out risk decisions to Qt, hardware, logging, or network sinks."""

    def __init__(self) -> None:
        self._subscribers: list[Callable[[RiskDecision], None]] = []

    def subscribe(self, callback: Callable[[RiskDecision], None]) -> None:
        if callback not in self._subscribers:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[RiskDecision], None]) -> None:
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    def publish(self, decision: RiskDecision) -> None:
        for callback in tuple(self._subscribers):
            callback(decision)


class DigitalOutput(Protocol):
    def set_active(self, active: bool) -> None: ...

    def close(self) -> None: ...


class JetsonGPIOOutput:
    """Lazy Jetson.GPIO digital output; importing this module is hardware-safe."""

    def __init__(
        self,
        pin: int,
        *,
        numbering: str = "BOARD",
        active_high: bool = True,
    ) -> None:
        if pin <= 0:
            raise ValueError("GPIO pin must be positive.")
        try:
            import Jetson.GPIO as gpio
        except ImportError as error:
            raise RuntimeError(
                "Jetson.GPIO is required for a physical buzzer."
            ) from error
        mode = getattr(gpio, numbering, None)
        if mode is None or numbering not in ("BOARD", "BCM"):
            raise ValueError("numbering must be BOARD or BCM.")
        self._gpio = gpio
        self._pin = int(pin)
        self._active_high = bool(active_high)
        gpio.setwarnings(False)
        gpio.setmode(mode)
        gpio.setup(self._pin, gpio.OUT, initial=self._value(False))

    def _value(self, active: bool) -> int:
        return self._gpio.HIGH if active == self._active_high else self._gpio.LOW

    def set_active(self, active: bool) -> None:
        self._gpio.output(self._pin, self._value(bool(active)))

    def close(self) -> None:
        self.set_active(False)
        self._gpio.cleanup(self._pin)


class BuzzerPatternController:
    """Non-blocking buzzer patterns driven by RiskDecision events."""

    def __init__(self, output: DigitalOutput) -> None:
        self._output = output
        self._mode = BuzzerMode.OFF
        self._wake = threading.Event()
        self._closed = False
        self._thread = threading.Thread(
            target=self._run, name="buzzer-pattern", daemon=True
        )
        self._thread.start()

    def publish(self, decision: RiskDecision) -> None:
        self.set_mode(decision.buzzer_mode)

    def set_mode(self, mode: BuzzerMode) -> None:
        self._mode = BuzzerMode(mode)
        self._wake.set()

    def close(self) -> None:
        self._closed = True
        self._wake.set()
        self._thread.join(timeout=2.0)
        self._output.close()

    def _wait(self, seconds: float) -> bool:
        interrupted = self._wake.wait(seconds)
        self._wake.clear()
        return interrupted or self._closed

    def _run(self) -> None:
        while not self._closed:
            mode = self._mode
            if mode is BuzzerMode.OFF:
                self._output.set_active(False)
                self._wait(1.0)
            elif mode is BuzzerMode.ALERT:
                self._pulse(0.2, 0.3)
                self._pulse(0.2, 5.3)
            else:
                self._pulse(0.3, 0.2)
        self._output.set_active(False)

    def _pulse(self, on_seconds: float, off_seconds: float) -> None:
        self._output.set_active(True)
        if self._wait(on_seconds):
            return
        self._output.set_active(False)
        self._wait(off_seconds)
