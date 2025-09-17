"""Dataset handling for EvoPrompt."""

import json
import os
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class Sample:
    """Represents a single data sample."""

    def __init__(self, input_text: str, target: str, metadata: Optional[Dict] = None):
        self.input_text = input_text
        self.target = target
        self.metadata = metadata or {}

    def __repr__(self):
        return f"Sample(input='{self.input_text[:50]}...', target='{self.target}')"


class Dataset(ABC):
    """Abstract base class for datasets."""

    def __init__(self, name: str):
        self.name = name
        self._samples = []

    @abstractmethod
    def load_data(self, data_path: str) -> List[Sample]:
        """Load data from file."""
        pass

    def get_samples(self, n: Optional[int] = None) -> List[Sample]:
        """Get n samples (or all if n is None)."""
        if n is None:
            return self._samples
        return self._samples[:n]

    def __len__(self):
        return len(self._samples)


class PrimevulDataset(Dataset):
    """Dataset class for Primevul vulnerability detection data."""

    def __init__(self, data_path: str, split: str = "dev"):
        super().__init__(f"primevul_{split}")
        self.data_path = data_path
        self.split = split
        self._samples = self.load_data(data_path)

    def load_data(self, data_path: str) -> List[Sample]:
        """Load Primevul data from JSONL or tab-separated format."""
        samples = []

        if not os.path.exists(data_path):
            logger.warning(f"Data file not found: {data_path}")
            return samples

        try:
            with open(data_path, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
                f.seek(0)  # Reset file pointer

                # Detect format based on first line
                if first_line.startswith("{"):
                    # JSONL format
                    for line in f:
                        if line.strip():
                            item = json.loads(line.strip())

                            # Extract function code and vulnerability label
                            func_code = item.get("func", "")
                            target = str(
                                item.get("target", 0)
                            )  # 0=benign, 1=vulnerable

                            # Create metadata
                            metadata = {
                                "idx": item.get("idx"),
                                "project": item.get("project"),
                                "commit_id": item.get("commit_id"),
                                "cwe": item.get("cwe", []),
                                "cve": item.get("cve"),
                                "func_hash": item.get("func_hash"),
                            }

                            sample = Sample(
                                input_text=func_code.strip(),
                                target=target,
                                metadata=metadata,
                            )
                            samples.append(sample)
                else:
                    # Tab-separated format
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue

                        parts = line.split("\t")
                        if len(parts) != 2:
                            logger.warning(f"Invalid format at line {line_num}: {line}")
                            continue

                        func_code, target = parts

                        sample = Sample(
                            input_text=func_code.strip(),
                            target=target.strip(),
                            metadata={"line_num": line_num},
                        )
                        samples.append(sample)

        except Exception as e:
            logger.error(f"Error loading data from {data_path}: {e}")

        logger.info(f"Loaded {len(samples)} samples from {data_path}")
        return samples


class BenchmarkDataset(Dataset):
    """Dataset class for benchmark vulnerability annotations stored as JSON."""

    def __init__(self, data_path: str, split: str = "dev"):
        super().__init__(f"benchmark_{split}")
        self.data_path = data_path
        self.split = split
        self._samples = self.load_data(data_path)

    def _parse_ground_truth(
        self, issues: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
        """Extract category names, CWE codes, and normalized issue metadata."""
        categories: List[str] = []
        cwe_codes: List[str] = []
        normalized_issues: List[Dict[str, Any]] = []

        for issue in issues:
            if not isinstance(issue, dict):
                continue

            category = issue.get("category")
            if category:
                categories.append(str(category))

            cwe_id = issue.get("cwe_id")
            if cwe_id is not None:
                try:
                    cwe_int = int(cwe_id)
                    cwe_codes.append(f"CWE-{cwe_int}")
                except (ValueError, TypeError):
                    # Skip malformed entries without aborting load
                    pass

            normalized_issues.append(
                {
                    "category": category,
                    "cwe_id": cwe_id,
                    "line": issue.get("line"),
                    "lineno": issue.get("lineno"),
                }
            )

        return categories, cwe_codes, normalized_issues

    def load_data(self, data_path: str) -> List[Sample]:
        """Load benchmark data from a JSON array."""
        samples: List[Sample] = []

        if not os.path.exists(data_path):
            logger.warning(f"Data file not found: {data_path}")
            return samples

        try:
            with open(data_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict) and "data" in data:
                # Allow wrapper format {"data": [...]}
                data = data["data"]

            if not isinstance(data, list):
                logger.error(f"Benchmark data should be a list, got {type(data)}")
                return samples

            for idx, item in enumerate(data):
                if not isinstance(item, dict):
                    logger.debug(f"Skipping non-dict entry at index {idx}")
                    continue

                raw_code = item.get("code", "")
                if not isinstance(raw_code, str):
                    raw_code = str(raw_code)
                code = raw_code.replace("\r\n", "\n").strip()

                issues = item.get("gt", []) or []
                categories, cwe_codes, normalized_issues = self._parse_ground_truth(
                    issues
                )
                has_vulnerability = bool(normalized_issues)

                metadata: Dict[str, Any] = {
                    "index": item.get("index", idx),
                    "lang": item.get("lang"),
                    "source": item.get("source"),
                    "categories": categories,
                    "cwe": cwe_codes,
                    "issues": normalized_issues,
                    "false_positives": item.get("fp", []),
                }

                sample = Sample(
                    input_text=code,
                    target="1" if has_vulnerability else "0",
                    metadata=metadata,
                )
                samples.append(sample)

        except json.JSONDecodeError as exc:
            logger.error(f"Failed to parse benchmark JSON: {exc}")
        except Exception as exc:  # pylint: disable=broad-except
            logger.error(f"Error loading data from {data_path}: {exc}")

        logger.info(f"Loaded {len(samples)} samples from {data_path}")
        return samples


class SVENDataset(Dataset):
    """Dataset class for SVEN vulnerability detection data."""

    def __init__(self, data_path: str, split: str = "dev"):
        super().__init__(f"sven_{split}")
        self.data_path = data_path
        self.split = split
        self._samples = self.load_data(data_path)

    def load_data(self, data_path: str) -> List[Sample]:
        """Load SVEN data from tab-separated format."""
        samples = []

        if not os.path.exists(data_path):
            logger.warning(f"Data file not found: {data_path}")
            return samples

        try:
            with open(data_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split("\t")
                    if len(parts) != 2:
                        logger.warning(f"Invalid format at line {line_num}: {line}")
                        continue

                    func_code, target = parts

                    sample = Sample(
                        input_text=func_code.strip(),
                        target=target.strip(),
                        metadata={"line_num": line_num},
                    )
                    samples.append(sample)

        except Exception as e:
            logger.error(f"Error loading data from {data_path}: {e}")

        logger.info(f"Loaded {len(samples)} samples from {data_path}")
        return samples


class TextClassificationDataset(Dataset):
    """General text classification dataset."""

    def __init__(self, data_path: str, split: str = "dev"):
        super().__init__(f"text_cls_{split}")
        self.data_path = data_path
        self.split = split
        self._samples = self.load_data(data_path)

    def load_data(self, data_path: str) -> List[Sample]:
        """Load text classification data."""
        samples = []

        if not os.path.exists(data_path):
            logger.warning(f"Data file not found: {data_path}")
            return samples

        try:
            with open(data_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split("\t")
                    if len(parts) >= 2:
                        text = parts[0].strip()
                        label = parts[1].strip()

                        sample = Sample(
                            input_text=text,
                            target=label,
                            metadata={"line_num": line_num},
                        )
                        samples.append(sample)

        except Exception as e:
            logger.error(f"Error loading data from {data_path}: {e}")

        logger.info(f"Loaded {len(samples)} samples from {data_path}")
        return samples


def create_dataset(dataset_name: str, data_path: str, split: str = "dev") -> Dataset:
    """Factory function to create datasets."""
    dataset_name = dataset_name.lower()

    if dataset_name == "primevul":
        return PrimevulDataset(data_path, split)
    if dataset_name == "benchmark":
        return BenchmarkDataset(data_path, split)
    if dataset_name == "sven":
        return SVENDataset(data_path, split)
    return TextClassificationDataset(data_path, split)


def prepare_primevul_data(primevul_dir: str, output_dir: str) -> Tuple[str, str]:
    """Prepare Primevul data for EvoPrompt format."""
    primevul_dir = Path(primevul_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert JSONL to tab-separated format
    dev_file = primevul_dir / "dev.jsonl"
    test_file = primevul_dir / "primevul_test.jsonl"

    dev_output = output_dir / "dev.txt"
    test_output = output_dir / "test.txt"

    def convert_jsonl_to_txt(input_file: Path, output_file: Path):
        """Convert JSONL format to tab-separated format."""
        if not input_file.exists():
            logger.warning(f"Input file not found: {input_file}")
            return

        with open(input_file, "r", encoding="utf-8") as f_in, open(
            output_file, "w", encoding="utf-8"
        ) as f_out:
            for line in f_in:
                if line.strip():
                    item = json.loads(line.strip())
                    func_code = item.get("func", "").strip()
                    target = str(item.get("target", 0))

                    # Clean up function code for single line format
                    func_code = func_code.replace("\n", " ").replace("\t", " ")
                    while "  " in func_code:
                        func_code = func_code.replace("  ", " ")

                    f_out.write(f"{func_code}\t{target}\n")

    # Convert files
    if dev_file.exists():
        convert_jsonl_to_txt(dev_file, dev_output)
        logger.info(f"Converted dev data: {dev_output}")

    if test_file.exists():
        convert_jsonl_to_txt(test_file, test_output)
        logger.info(f"Converted test data: {test_output}")
    else:
        # Use dev as test if test doesn't exist
        if dev_output.exists():
            import shutil

            shutil.copy2(dev_output, test_output)
            logger.info(f"Copied dev data as test: {test_output}")

    return str(dev_output), str(test_output)
