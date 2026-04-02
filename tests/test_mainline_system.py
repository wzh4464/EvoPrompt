import pytest

from evoprompt.agents.hierarchical_detector import MAJOR_TO_MIDDLE
from evoprompt.mainline.artifacts import PromptArtifact
from evoprompt.mainline.system import MainlineDetectorSystem


class StubLLMClient:
    def __init__(self, responses=None, default_response=None):
        self.responses = responses or {}
        self.default_response = default_response or (
            '{"predictions":[{"category":"Benign","confidence":0.70}]}'
        )

    def generate(self, prompt: str, **kwargs) -> str:
        for marker, response in self.responses.items():
            if marker in prompt:
                return response
        return self.default_response

    def batch_generate(self, prompts, **kwargs):
        return [self.generate(prompt, **kwargs) for prompt in prompts]


def _build_artifact(include_cwe: bool = True) -> PromptArtifact:
    prompts = {
        f"major_{major}": (
            f"You are a security expert specializing in {major} vulnerabilities.\n"
            "{evidence}\n{code}\n{candidates}"
        )
        for major in MAJOR_TO_MIDDLE.keys()
    }
    prompts["middle_Buffer Errors"] = (
        "You are a Buffer Errors vulnerability expert.\n"
        "{evidence}\n{code}\n{candidates}"
    )
    if include_cwe:
        prompts["cwe_CWE-120"] = (
            "You are a vulnerability expert. Identify if this code has CWE-120.\n"
            "## Possible CWEs: {candidates}\n{evidence}\n{code}"
        )
    return PromptArtifact.from_mapping({"prompts": prompts})


def test_mainline_detector_returns_best_router_detector_path():
    artifact = _build_artifact()
    client = StubLLMClient(
        responses={
            "specializing in Memory vulnerabilities": (
                '{"predictions":[{"category":"Memory","confidence":0.91}]}'
            ),
            "specializing in Injection vulnerabilities": (
                '{"predictions":[{"category":"Benign","confidence":0.80}]}'
            ),
            "Buffer Errors vulnerability expert": (
                '{"predictions":[{"category":"Buffer Errors","confidence":0.84}]}'
            ),
            "Possible CWEs: CWE-119, CWE-120, CWE-125, CWE-787, CWE-805, Benign": (
                '{"predictions":[{"cwe":"CWE-120","confidence":0.88}]}'
            ),
        }
    )
    system = MainlineDetectorSystem(client, artifact)

    result = system.detect("strcpy(buf, input);")

    assert result.major == "Memory"
    assert result.middle == "Buffer Errors"
    assert result.cwe == "CWE-120"
    assert result.prediction == "CWE-120"


def test_score_detector_requires_top1_match():
    artifact = _build_artifact(include_cwe=False)
    client = StubLLMClient(
        responses={
            "specializing in Memory vulnerabilities": (
                '{"predictions":['
                '{"category":"Benign","confidence":0.90},'
                '{"category":"Memory","confidence":0.60}'
                "]} "
            )
        }
    )
    system = MainlineDetectorSystem(client, artifact)

    score = system._score_detector(system.major_detectors["Memory"], "char *p = argv[1];")

    assert score.predicted_label == "Benign"
    assert score.confidence == 0.0


def test_detect_backoffs_to_benign_when_middle_rejects():
    artifact = _build_artifact(include_cwe=False)
    client = StubLLMClient(
        responses={
            "specializing in Memory vulnerabilities": (
                '{"predictions":[{"category":"Memory","confidence":0.91}]}'
            ),
            "Buffer Errors vulnerability expert": (
                '{"predictions":['
                '{"category":"Benign","confidence":0.84},'
                '{"category":"Buffer Errors","confidence":0.40}'
                "]} "
            ),
        }
    )
    system = MainlineDetectorSystem(client, artifact)

    result = system.detect("strcpy(buf, input);")

    assert result.prediction == "Benign"
    assert result.major == "Memory"
    assert result.middle == "Benign"
    assert result.cwe == "Unknown"


def test_detect_keeps_middle_when_cwe_rejects():
    artifact = _build_artifact()
    client = StubLLMClient(
        responses={
            "specializing in Memory vulnerabilities": (
                '{"predictions":[{"category":"Memory","confidence":0.91}]}'
            ),
            "Buffer Errors vulnerability expert": (
                '{"predictions":[{"category":"Buffer Errors","confidence":0.84}]}'
            ),
            "Possible CWEs: CWE-119, CWE-120, CWE-125, CWE-787, CWE-805, Benign": (
                '{"predictions":['
                '{"cwe":"Benign","confidence":0.88},'
                '{"cwe":"CWE-120","confidence":0.42}'
                "]} "
            ),
        }
    )
    system = MainlineDetectorSystem(client, artifact)

    result = system.detect("strcpy(buf, input);")

    assert result.prediction == "Buffer Errors"
    assert result.middle == "Buffer Errors"
    assert result.cwe == "Unknown"


def test_init_raises_on_missing_major_prompt():
    artifact = PromptArtifact.from_mapping(
        {
            "prompts": {
                "major_Memory": "memory prompt\n{evidence}\n{code}\n{candidates}",
            }
        }
    )

    with pytest.raises(ValueError, match="missing router prompts for major classes"):
        MainlineDetectorSystem(StubLLMClient(), artifact)
