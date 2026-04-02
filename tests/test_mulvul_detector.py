"""Tests for adaptive MulVul detector routing."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evoprompt.agents.base import DetectionResult, RoutingResult
from evoprompt.agents.mulvul import MulVulDetector
from evoprompt.agents.router_agent import RouterAgent


class StubRouter:
    """Deterministic router stub for MulVul tests."""

    def __init__(self, categories):
        self._routing_result = RoutingResult(
            categories=categories,
            evidence_used=[],
            raw_response="router-response",
        )

    def route(self, code: str) -> RoutingResult:
        return self._routing_result


class StubDetector:
    """Detector stub that records calls and returns a fixed result."""

    def __init__(self, category: str, prediction: str, confidence: float):
        self.category = category
        self.prediction = prediction
        self.confidence = confidence
        self.calls = 0

    def detect(self, code: str) -> DetectionResult:
        self.calls += 1
        return DetectionResult(
            prediction=self.prediction,
            confidence=self.confidence,
            evidence=f"{self.category} evidence",
            category=self.category,
            cwe=self.prediction if self.prediction.startswith("CWE-") else "",
            agent_id=f"detector_{self.category.lower()}",
        )


def _build_detectors():
    return {
        "Memory": StubDetector("Memory", "CWE-120", 0.91),
        "Injection": StubDetector("Injection", "CWE-89", 0.72),
        "Input": StubDetector("Input", "CWE-22", 0.61),
        "Crypto": StubDetector("Crypto", "CWE-327", 0.44),
    }


def test_router_parse_response_sorts_and_filters_predictions():
    router = RouterAgent(llm_client=object())
    response = """
    {
      "predictions": [
        {"category": "Injection", "confidence": 0.52},
        {"category": "Memory", "confidence": 0.81},
        {"category": "Unknown", "confidence": 0.99},
        {"category": "Memory", "confidence": 0.33}
      ]
    }
    """

    parsed = router._parse_response(response)

    assert parsed == [("Memory", 0.81), ("Injection", 0.52)]


def test_adaptive_routing_selects_single_detector_for_confident_router():
    detectors = _build_detectors()
    detector = MulVulDetector(
        router=StubRouter([
            ("Memory", 0.92),
            ("Injection", 0.40),
            ("Input", 0.32),
            ("Benign", 0.05),
        ]),
        detectors=detectors,
        parallel=False,
        max_agents=3,
        adaptive=True,
        routing_confidence_threshold=0.2,
        routing_relative_threshold=0.6,
    )

    details = detector.detect_with_details("code sample")

    assert details["routing"]["selected"] == [("Memory", 0.92)]
    assert detectors["Memory"].calls == 1
    assert detectors["Injection"].calls == 0
    assert detectors["Input"].calls == 0


def test_adaptive_routing_selects_multiple_detectors_when_router_is_uncertain():
    detectors = _build_detectors()
    detector = MulVulDetector(
        router=StubRouter([
            ("Memory", 0.51),
            ("Injection", 0.46),
            ("Input", 0.31),
            ("Crypto", 0.18),
            ("Benign", 0.07),
        ]),
        detectors=detectors,
        parallel=False,
        max_agents=3,
        adaptive=True,
        routing_confidence_threshold=0.2,
        routing_relative_threshold=0.6,
    )

    details = detector.detect_with_details("code sample")

    assert details["routing"]["selected"] == [
        ("Memory", 0.51),
        ("Injection", 0.46),
        ("Input", 0.31),
    ]
    assert detectors["Memory"].calls == 1
    assert detectors["Injection"].calls == 1
    assert detectors["Input"].calls == 1
    assert detectors["Crypto"].calls == 0


def test_benign_short_circuit_skips_specialist_detectors():
    detectors = _build_detectors()
    detector = MulVulDetector(
        router=StubRouter([
            ("Benign", 0.88),
            ("Memory", 0.42),
            ("Injection", 0.19),
        ]),
        detectors=detectors,
        parallel=False,
        max_agents=3,
        adaptive=True,
        routing_relative_threshold=0.6,
        benign_short_circuit_threshold=0.8,
    )

    result = detector.detect("code sample")

    assert result.prediction == "Benign"
    assert result.category == "Benign"
    assert all(stub.calls == 0 for stub in detectors.values())


def test_fixed_routing_preserves_max_agent_cap():
    detectors = _build_detectors()
    detector = MulVulDetector(
        router=StubRouter([
            ("Memory", 0.80),
            ("Injection", 0.79),
            ("Input", 0.78),
            ("Crypto", 0.77),
        ]),
        detectors=detectors,
        parallel=False,
        max_agents=2,
        adaptive=False,
    )

    details = detector.detect_with_details("code sample")

    assert details["routing"]["selected"] == [
        ("Memory", 0.80),
        ("Injection", 0.79),
    ]
    assert detectors["Memory"].calls == 1
    assert detectors["Injection"].calls == 1
    assert detectors["Input"].calls == 0
    assert detectors["Crypto"].calls == 0
