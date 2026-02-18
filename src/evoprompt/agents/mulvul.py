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
    - Router predicts top-k categories
    - Only relevant Detectors are invoked
    - Results are aggregated by confidence
    """

    def __init__(
        self,
        router: RouterAgent,
        detectors: Dict[str, DetectorAgent],
        aggregator: DecisionAggregator = None,
        parallel: bool = True,
        k: int = 3,
    ):
        """Initialize MulVul detector.

        Args:
            router: Router agent for category prediction
            detectors: Dict of category -> DetectorAgent
            aggregator: Decision aggregator (default: confidence-based)
            parallel: Whether to run detectors in parallel
            k: Number of top categories to route to
        """
        self.router = router
        self.detectors = detectors
        self.aggregator = aggregator or DecisionAggregator()
        self.parallel = parallel
        self.k = k

    def detect(self, code: str) -> DetectionResult:
        """Detect vulnerability in code.

        Args:
            code: Source code to analyze

        Returns:
            DetectionResult with prediction, confidence, and evidence
        """
        # Stage I: Routing
        routing_result = self.router.route(code)
        top_k_categories = routing_result.get_top_k(self.k)

        # Stage II: Detection
        if self.parallel:
            results = self._detect_parallel(code, top_k_categories)
        else:
            results = self._detect_sequential(code, top_k_categories)

        # Stage III: Aggregation
        final_result = self.aggregator.aggregate(results)

        # Add routing info to result
        final_result.raw_response = f"Routed to: {[c[0] for c in top_k_categories]}\n{final_result.raw_response}"

        return final_result

    def detect_with_details(self, code: str) -> Dict:
        """Detect with full details including intermediate results.

        Returns:
            Dict with routing, detection, and aggregation details
        """
        # Stage I: Routing
        routing_result = self.router.route(code)
        top_k_categories = routing_result.get_top_k(self.k)

        # Stage II: Detection
        if self.parallel:
            detector_results = self._detect_parallel(code, top_k_categories)
        else:
            detector_results = self._detect_sequential(code, top_k_categories)

        # Stage III: Aggregation
        final_result = self.aggregator.aggregate(detector_results)

        return {
            "routing": {
                "top_k": top_k_categories,
                "evidence_count": len(routing_result.evidence_used),
            },
            "detectors": [r.to_dict() for r in detector_results],
            "final": final_result.to_dict(),
        }

    def _detect_parallel(
        self, code: str, categories: List[tuple]
    ) -> List[DetectionResult]:
        """Run detectors in parallel."""
        results = []

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
        k: int = 3,
        parallel: bool = True,
    ) -> "MulVulDetector":
        """Create MulVul detector with default configuration.

        Args:
            llm_client: LLM client for all agents
            retriever: Optional retriever for evidence
            k: Number of top categories
            parallel: Whether to run detectors in parallel

        Returns:
            Configured MulVulDetector instance
        """
        router = RouterAgent(
            llm_client=llm_client,
            retriever=retriever,
            k=k,
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
            k=k,
        )
