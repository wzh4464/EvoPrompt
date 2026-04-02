"""MulVul Multi-Agent Vulnerability Detector.

Implements the complete detection pipeline from method.md Algorithm 3:
1. Stage I: Coarse-grained Routing (Router Agent)
2. Stage II: Fine-grained Detection (Detector Agents)
3. Stage III: Decision Aggregation
"""

from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from .base import DetectionResult, RoutingResult
from .router_agent import RouterAgent
from .detector_agent import DetectorAgent, DetectorAgentFactory
from .aggregator import DecisionAggregator


class MulVulDetector:
    """Complete MulVul detection pipeline.

    Implements Router-Detector architecture:
    - Router predicts ranked categories
    - Detector fan-out adapts to router confidence
    - Results are aggregated by confidence
    """

    def __init__(
        self,
        router: RouterAgent,
        detectors: Dict[str, DetectorAgent],
        aggregator: DecisionAggregator = None,
        parallel: bool = True,
        max_agents: int = 3,
        adaptive: bool = True,
        min_agents: int = 1,
        routing_confidence_threshold: float = 0.2,
        routing_relative_threshold: float = 0.6,
        benign_short_circuit_threshold: float = 0.8,
    ):
        """Initialize MulVul detector.

        Args:
            router: Router agent for category prediction
            detectors: Dict of category -> DetectorAgent
            aggregator: Decision aggregator (default: confidence-based)
            parallel: Whether to run detectors in parallel
            max_agents: Maximum number of detector agents to invoke
            adaptive: Whether to adapt detector fan-out using router confidence
            min_agents: Minimum number of detector agents to invoke when not
                short-circuiting to Benign
            routing_confidence_threshold: Absolute minimum confidence for
                adaptive detector fan-out
            routing_relative_threshold: Relative threshold against the highest
                vulnerable category confidence for adaptive fan-out
            benign_short_circuit_threshold: If Benign is the top route and is
                this confident, skip specialist detectors unless a vulnerable
                category remains competitive
        """
        self.router = router
        self.detectors = detectors
        self.aggregator = aggregator or DecisionAggregator()
        self.parallel = parallel
        self.max_agents = max(1, max_agents)
        self.k = self.max_agents
        self.adaptive = adaptive
        self.min_agents = max(1, min_agents)
        self.routing_confidence_threshold = routing_confidence_threshold
        self.routing_relative_threshold = routing_relative_threshold
        self.benign_short_circuit_threshold = benign_short_circuit_threshold

    def detect(self, code: str) -> DetectionResult:
        """Detect vulnerability in code.

        Args:
            code: Source code to analyze

        Returns:
            DetectionResult with prediction, confidence, and evidence
        """
        # Stage I: Routing
        routing_result = self.router.route(code)
        selected_categories = self._select_detector_categories(routing_result)

        if not selected_categories:
            return self._handle_empty_selection(routing_result)

        # Stage II: Detection
        if self.parallel:
            results = self._detect_parallel(code, selected_categories)
        else:
            results = self._detect_sequential(code, selected_categories)

        # Stage III: Aggregation
        final_result = self.aggregator.aggregate(results)

        # Add routing info to result
        final_result.raw_response = (
            f"Selected detectors: {[c[0] for c in selected_categories]}\n"
            f"Router ranking: {[c[0] for c in routing_result.categories]}\n"
            f"{final_result.raw_response}"
        )

        return final_result

    def detect_with_details(self, code: str) -> Dict:
        """Detect with full details including intermediate results.

        Returns:
            Dict with routing, detection, and aggregation details
        """
        # Stage I: Routing
        routing_result = self.router.route(code)
        selected_categories = self._select_detector_categories(routing_result)

        # Stage II: Detection
        if not selected_categories:
            final_result = self._handle_empty_selection(routing_result)
            detector_results = []
        elif self.parallel:
            detector_results = self._detect_parallel(code, selected_categories)
            final_result = self.aggregator.aggregate(detector_results)
        else:
            detector_results = self._detect_sequential(code, selected_categories)
            final_result = self.aggregator.aggregate(detector_results)

        return {
            "routing": {
                "top_k": selected_categories,
                "ranked": routing_result.categories,
                "selected": selected_categories,
                "selected_agent_count": len(selected_categories),
                "evidence_count": len(routing_result.evidence_used),
            },
            "detectors": [r.to_dict() for r in detector_results],
            "final": final_result.to_dict(),
        }

    def _select_detector_categories(
        self,
        routing_result: RoutingResult,
    ) -> List[tuple]:
        """Select detector categories from router output.

        The router returns a ranked list across all categories. When adaptive
        mode is enabled, specialist detector fan-out expands only when the
        router remains uncertain about the vulnerable category.
        """
        ranked_categories = routing_result.get_top_k()
        vulnerable_categories = [
            (category, confidence)
            for category, confidence in ranked_categories
            if category in self.detectors
        ]

        if not vulnerable_categories:
            return []

        if not self.adaptive:
            return vulnerable_categories[:self.max_agents]

        top_category = routing_result.top_category
        top_confidence = routing_result.top_confidence
        top_vulnerable_confidence = vulnerable_categories[0][1]

        if (
            top_category == "Benign"
            and top_confidence >= self.benign_short_circuit_threshold
            and top_vulnerable_confidence
            < top_confidence * self.routing_relative_threshold
        ):
            return []

        dynamic_threshold = max(
            self.routing_confidence_threshold,
            top_vulnerable_confidence * self.routing_relative_threshold,
        )

        selected = [
            (category, confidence)
            for category, confidence in vulnerable_categories
            if confidence >= dynamic_threshold
        ][:self.max_agents]

        if not selected:
            min_required = min(
                len(vulnerable_categories),
                self.max_agents,
                self.min_agents,
            )
            return vulnerable_categories[:min_required]

        if len(selected) < self.min_agents:
            min_required = min(
                len(vulnerable_categories),
                self.max_agents,
                self.min_agents,
            )
            return vulnerable_categories[:min_required]

        return selected

    def _handle_empty_selection(
        self,
        routing_result: RoutingResult,
    ) -> DetectionResult:
        """Handle cases where no specialist detector should be invoked."""
        if routing_result.top_category == "Benign":
            return DetectionResult(
                prediction="Benign",
                confidence=routing_result.top_confidence,
                evidence=(
                    "Router predicted Benign with high confidence, so no "
                    "specialist detector was invoked."
                ),
                category="Benign",
                agent_id="router_benign",
                raw_response=routing_result.raw_response,
            )

        return DetectionResult(
            prediction="Unknown",
            confidence=0.0,
            evidence="Router did not return any supported detector category.",
            agent_id="router_empty",
            raw_response=routing_result.raw_response,
        )

    def _detect_parallel(
        self, code: str, categories: List[tuple]
    ) -> List[DetectionResult]:
        """Run detectors in parallel."""
        results = []

        if not categories:
            return results

        with ThreadPoolExecutor(max_workers=len(categories)) as executor:
            futures = {}
            for category, confidence in categories:
                if category in self.detectors:
                    future = executor.submit(
                        self.detectors[category].detect, code
                    )
                    futures[future] = category

            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    category = futures[future]
                    results.append(DetectionResult(
                        prediction="Error",
                        confidence=0.0,
                        evidence=str(e),
                        category=category,
                        agent_id=f"detector_{category.lower()}",
                    ))

        return results

    def _detect_sequential(
        self, code: str, categories: List[tuple]
    ) -> List[DetectionResult]:
        """Run detectors sequentially."""
        results = []

        for category, confidence in categories:
            if category in self.detectors:
                try:
                    result = self.detectors[category].detect(code)
                    results.append(result)
                except Exception as e:
                    results.append(DetectionResult(
                        prediction="Error",
                        confidence=0.0,
                        evidence=str(e),
                        category=category,
                        agent_id=f"detector_{category.lower()}",
                    ))

        return results

    @classmethod
    def create_default(
        cls,
        llm_client,
        retriever=None,
        max_agents: int = 3,
        parallel: bool = True,
        adaptive: bool = True,
        k: Optional[int] = None,
        min_agents: int = 1,
        routing_confidence_threshold: float = 0.2,
        routing_relative_threshold: float = 0.6,
        benign_short_circuit_threshold: float = 0.8,
    ) -> "MulVulDetector":
        """Create MulVul detector with default configuration.

        Args:
            llm_client: LLM client for all agents
            retriever: Optional retriever for evidence
            max_agents: Maximum number of detector agents
            parallel: Whether to run detectors in parallel
            adaptive: Whether to adapt detector fan-out using router confidence
            k: Backward-compatible alias for max_agents

        Returns:
            Configured MulVulDetector instance
        """
        if k is not None:
            max_agents = k

        router = RouterAgent(
            llm_client=llm_client,
            retriever=retriever,
        )

        detectors = DetectorAgentFactory.create_all_detectors(
            llm_client=llm_client,
            retriever=retriever,
        )

        aggregator = DecisionAggregator()

        return cls(
            router=router,
            detectors=detectors,
            aggregator=aggregator,
            parallel=parallel,
            max_agents=max_agents,
            adaptive=adaptive,
            min_agents=min_agents,
            routing_confidence_threshold=routing_confidence_threshold,
            routing_relative_threshold=routing_relative_threshold,
            benign_short_circuit_threshold=benign_short_circuit_threshold,
        )
