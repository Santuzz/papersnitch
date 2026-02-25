#!/usr/bin/env python3
"""
Compute reproducibility score from input_analysis.json file.

This is a standalone script that doesn't require Django.
All necessary functions and models are embedded in this file.
"""

import json
import sys
from pathlib import Path
from typing import Optional, Dict, List
from pydantic import BaseModel, Field, ConfigDict


# ============================================================================
# Pydantic Models (copied from pydantic_schemas.py)
# ============================================================================


class ResearchMethodologyAnalysis(BaseModel):
    """Analysis of research methodology type for context-aware scoring."""

    model_config = ConfigDict(extra="forbid")

    methodology_type: str = Field(
        description="Type: 'deep_learning', 'machine_learning', 'algorithm', 'simulation', 'data_analysis', 'theoretical', 'other'"
    )
    requires_training: bool = Field(
        description="Whether this research requires model training"
    )
    requires_datasets: bool = Field(
        description="Whether this research requires datasets"
    )
    requires_splits: bool = Field(
        description="Whether this research requires dataset splits"
    )
    methodology_notes: str = Field(description="Notes about the research methodology")


class RepositoryStructureAnalysis(BaseModel):
    """Analysis of repository structure and dependencies."""

    model_config = ConfigDict(extra="forbid")

    is_standalone: bool = Field(
        description="Whether repository is standalone or built on another repo"
    )
    base_repository: Optional[str] = Field(
        description="Base repository if not standalone"
    )
    has_requirements: bool = Field(description="Requirements file exists")
    requirements_match_imports: Optional[bool] = Field(
        description="Whether requirements match code imports (None if cannot check)"
    )
    requirements_issues: List[str] = Field(
        description="List of issues with requirements"
    )


class CodeAvailabilityAnalysis(BaseModel):
    """Analysis of what code components are available."""

    model_config = ConfigDict(extra="forbid")

    has_training_code: bool = Field(description="Training code available")
    training_code_paths: List[str] = Field(description="Paths to training code files")
    has_evaluation_code: bool = Field(description="Evaluation/inference code available")
    evaluation_code_paths: List[str] = Field(
        description="Paths to evaluation code files"
    )
    has_documented_commands: bool = Field(
        description="Commands to run code are documented"
    )
    command_documentation_location: Optional[str] = Field(
        description="Where commands are documented (README, scripts, etc.)"
    )


class ArtifactsAnalysis(BaseModel):
    """Analysis of checkpoints and dataset availability."""

    model_config = ConfigDict(extra="forbid")

    has_checkpoints: bool = Field(description="Model checkpoints are released")
    checkpoint_locations: List[str] = Field(
        default_factory=list, description="URLs or paths to checkpoints"
    )
    has_dataset_links: bool = Field(description="Dataset download links available")
    dataset_coverage: str = Field(
        description="'full', 'partial', or 'none' - coverage of dataset links"
    )
    dataset_links: Optional[List[Dict[str, str]]] = Field(
        default=None, description="List of dataset names and their download URLs"
    )


class DatasetSplitsAnalysis(BaseModel):
    """Analysis of dataset splits and experiment replicability."""

    model_config = ConfigDict(extra="forbid")

    splits_specified: bool = Field(
        description="Dataset splits (train/val/test) are specified"
    )
    splits_provided: bool = Field(
        description="Split files or exact splits are provided in repo"
    )
    random_seeds_documented: bool = Field(
        description="Random seeds are documented for reproducible splits"
    )
    splits_notes: str = Field(
        description="Notes about dataset splits and replicability"
    )


class ReproducibilityDocumentation(BaseModel):
    """Analysis of reproducibility documentation."""

    model_config = ConfigDict(extra="forbid")

    has_readme: bool = Field(description="README file exists")
    has_results_table: bool = Field(description="README includes results table")
    has_reproduction_commands: bool = Field(
        description="README includes precise commands to reproduce results"
    )
    documentation_notes: str = Field(description="Notes about documentation quality")


# ============================================================================
# Reproducibility Score Calculation (copied from shared_helpers.py)
# ============================================================================


def compute_reproducibility_score(
    methodology: Optional[ResearchMethodologyAnalysis],
    structure: Optional[RepositoryStructureAnalysis],
    components: Optional[CodeAvailabilityAnalysis],
    artifacts: Optional[ArtifactsAnalysis],
    dataset_splits: Optional[DatasetSplitsAnalysis],
    documentation: Optional[ReproducibilityDocumentation],
) -> tuple[float, Dict[str, float], List[str]]:
    """
    Compute reproducibility score programmatically from extracted facts.

    Adapts scoring weights based on research methodology type:
    - deep_learning/machine_learning: Full weights on training, checkpoints, splits
    - algorithm: Focus on code completeness and examples
    - simulation: Focus on parameters and seeds
    - data_analysis: Focus on data and scripts
    - theoretical: Focus on proofs/derivations (if applicable)

    Returns:
        (score, breakdown, recommendations)
        - score: 0-100 overall reproducibility score
        - breakdown: Dict with component scores (0-100 scale)
        - recommendations: List of improvement suggestions
    """
    breakdown = {
        "code_completeness": 0.0,  # 2.5-3.0 points (adaptive)
        "dependencies": 0.0,  # 1.0 points
        "artifacts": 0.0,  # 0-2.5 points (adaptive)
        "dataset_splits": 0.0,  # 0-2.0 points (adaptive)
        "documentation": 0.0,  # 2.0 points
    }

    # Track max points for each component (for normalization to 0-100 scale)
    component_max = {
        "code_completeness": 0.0,
        "dependencies": 1.0,
        "artifacts": 0.0,
        "dataset_splits": 0.0,
        "documentation": 2.0,
    }

    recommendations = []

    # Determine methodology-specific weights
    if methodology:
        requires_training = methodology.requires_training
        requires_datasets = methodology.requires_datasets
        requires_splits = methodology.requires_splits
        method_type = methodology.methodology_type
    else:
        # Default to ML assumptions if no methodology detected
        requires_training = True
        requires_datasets = True
        requires_splits = True
        method_type = "unknown"

    # 1. Code Completeness (2.5-3.0 points, adaptive)
    max_code_points = 3.0 if requires_training else 2.5
    component_max["code_completeness"] = max_code_points

    if components:
        score = 0.0

        if requires_training:
            # ML/DL: Needs both training and evaluation
            if components.has_training_code and components.has_evaluation_code:
                score = 2.5
            elif components.has_evaluation_code or components.has_training_code:
                score = 1.5
                if not components.has_training_code:
                    recommendations.append(
                        "Add training code to enable full reproducibility"
                    )
                if not components.has_evaluation_code:
                    recommendations.append("Add evaluation/inference code")
            else:
                score = 0.5
                recommendations.append("Provide both training and evaluation code")
        else:
            # Non-ML: Evaluation/implementation code is sufficient
            if components.has_evaluation_code:
                score = 2.0  # Full credit for having implementation
            elif components.has_training_code:  # Could be simulation code
                score = 2.0
            else:
                score = 0.5
                recommendations.append(
                    f"Provide implementation code for the {method_type} method"
                )

        # Bonus for documented commands (always valuable)
        if components.has_documented_commands:
            score += 0.5
        else:
            recommendations.append("Document precise commands to run the code")

        breakdown["code_completeness"] = min(score, max_code_points)
    else:
        recommendations.append("Provide complete code implementation")

    # 2. Dependencies (1.0 point - always important)
    if structure:
        if structure.has_requirements:
            if structure.requirements_match_imports is True:
                breakdown["dependencies"] = 1.0
            elif structure.requirements_match_imports is False:
                breakdown["dependencies"] = 0.5
                recommendations.append(
                    "Fix dependencies file - some imports are missing"
                )
            else:
                breakdown["dependencies"] = 0.7
        else:
            recommendations.append(
                "Add requirements/dependencies file with all necessary packages and versions"
            )

    # 3. Artifacts (0-2.5 points, adaptive)
    if requires_datasets or requires_training:
        component_max["artifacts"] = 2.5
        if artifacts:
            # Checkpoints: 0-1.0 point (only for models)
            if requires_training:
                if artifacts.has_checkpoints:
                    breakdown["artifacts"] += 1.0
                else:
                    recommendations.append(
                        "Release model checkpoints to enable result verification without retraining"
                    )

            # Dataset links: 0-1.5 points (weighted by coverage)
            if requires_datasets:
                if artifacts.has_dataset_links:
                    if artifacts.dataset_coverage == "full":
                        breakdown["artifacts"] += 1.5
                    elif artifacts.dataset_coverage == "partial":
                        breakdown["artifacts"] += 0.8
                        recommendations.append(
                            "Provide download links for ALL datasets used"
                        )
                    else:
                        breakdown["artifacts"] += 0.3
                else:
                    recommendations.append("Provide dataset download links")
            else:
                # Non-dataset research: Give partial credit if repo is complete
                breakdown["artifacts"] += 1.0  # Baseline for having working code
        else:
            if requires_training:
                recommendations.append("Release model checkpoints and dataset links")
            elif requires_datasets:
                recommendations.append("Provide dataset download links")
    else:
        # Non-data research: Award full artifacts points if code is complete
        component_max["artifacts"] = 2.0
        if components and (
            components.has_evaluation_code or components.has_training_code
        ):
            breakdown["artifacts"] = 2.0  # Full credit for complete implementation

    # 4. Dataset Splits (0-2.0 points, adaptive) - CRITICAL for ML, less for others
    if requires_splits:
        component_max["dataset_splits"] = 2.0
        if dataset_splits:
            score = 0.0
            if dataset_splits.splits_specified:
                score += 0.7
            else:
                recommendations.append(
                    "Specify which dataset splits (train/val/test) were used"
                )

            if dataset_splits.splits_provided:
                score += 0.7
            else:
                recommendations.append("Provide split files or explicit split logic")

            if dataset_splits.random_seeds_documented:
                score += 0.6
            else:
                recommendations.append(
                    "Document random seeds for reproducible data partitioning"
                )

            breakdown["dataset_splits"] = score
        else:
            recommendations.append(
                "Document dataset splits and random seeds for experiment replicability"
            )
    else:
        # Non-ML: Award points if seeds/parameters are documented
        component_max["dataset_splits"] = 1.5
        if dataset_splits and dataset_splits.random_seeds_documented:
            breakdown["dataset_splits"] = 1.5  # Reward for documenting randomness
            recommendations.append("Continue documenting all sources of randomness")
        else:
            breakdown["dataset_splits"] = (
                0.5  # Partial credit for deterministic methods
            )
            if method_type in ["simulation", "algorithm"]:
                recommendations.append(
                    "Document random seeds and parameters for reproducible results"
                )

    # 5. Documentation (2.0 points - always critical)
    if documentation:
        if documentation.has_readme:
            breakdown["documentation"] += 0.5
        else:
            recommendations.append("Create comprehensive README file")

        if documentation.has_results_table:
            breakdown["documentation"] += 0.75
        else:
            recommendations.append("Include results table in README for comparison")

        if documentation.has_reproduction_commands:
            breakdown["documentation"] += 0.75
        else:
            recommendations.append("Document step-by-step reproduction commands")
    else:
        recommendations.append(
            "Add comprehensive documentation with results and reproduction steps"
        )

    # Calculate maximum achievable score for this paper type (for normalization)
    max_possible_score = 0.0
    max_possible_score += max_code_points  # 2.5 or 3.0
    max_possible_score += 1.0  # dependencies (always)

    if requires_datasets or requires_training:
        max_possible_score += 2.5  # artifacts
    else:
        max_possible_score += 2.0  # baseline for complete code

    if requires_splits:
        max_possible_score += 2.0  # dataset_splits
    else:
        max_possible_score += 1.5  # best case for non-split papers (documented seeds)

    max_possible_score += 2.0  # documentation (always)

    # Compute raw total score
    raw_score = sum(breakdown.values())

    # Normalize to 100-point scale
    total_score = (
        (raw_score / max_possible_score) * 100.0 if max_possible_score > 0 else 0.0
    )

    # Round to 1 decimal place
    total_score = round(total_score, 1)

    # Scale each component breakdown to 0-100 based on its OWN max value
    breakdown_normalized = {}
    for component, raw_value in breakdown.items():
        max_for_component = component_max.get(component, 1.0)
        if max_for_component > 0:
            normalized = round((raw_value / max_for_component) * 100.0, 1)
        else:
            normalized = 0.0
        breakdown_normalized[component] = normalized

    return total_score, breakdown_normalized, recommendations


# ============================================================================
# Main Script Functions
# ============================================================================


def load_analysis_from_json(json_path: str) -> dict:
    """Load analysis data from JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def extract_code_analysis(data: dict) -> Optional[dict]:
    """
    Extract code_analysis section from the JSON data.

    Handles both direct code_analysis format and nested paper format.
    """
    # Check if data has a paper name as top-level key
    for keys, values in data.items():
        print(
            f"Top-level key: {keys} (type: {type(keys)}) - value type: {type(values)}"
        )
    if len(data) == 1 and isinstance(list(data.values())[0], dict):
        paper_data = list(data.values())[0]
        if "code_analysis" in paper_data:
            return paper_data["code_analysis"]

    # Check if code_analysis is at the top level
    if "code_analysis" in data:
        return data["code_analysis"]

    # Assume the entire data is the code_analysis
    return data


def parse_analysis_components(code_analysis: dict) -> tuple:
    """Parse code analysis components into Pydantic models.

    Handles two formats:
    1. Direct format: keys like 'research_methodology', 'repository_structure', etc.
    2. Structured format: data nested under 'structured_data' with keys like 'methodology', 'structure', etc.
    """

    # Check if data is in structured_data format
    if "structured_data" in code_analysis:
        code_analysis = code_analysis["structured_data"]

    # Extract research methodology (handles both 'research_methodology' and 'methodology' keys)
    methodology = None
    if "research_methodology" in code_analysis:
        methodology_data = code_analysis["research_methodology"]
        methodology = ResearchMethodologyAnalysis(**methodology_data)
    elif "methodology" in code_analysis:
        methodology_data = code_analysis["methodology"]
        methodology = ResearchMethodologyAnalysis(**methodology_data)

    # Extract repository structure (handles both 'repository_structure' and 'structure' keys)
    structure = None
    if "repository_structure" in code_analysis:
        structure_data = code_analysis["repository_structure"]
        structure = RepositoryStructureAnalysis(**structure_data)
    elif "structure" in code_analysis:
        structure_data = code_analysis["structure"]
        structure = RepositoryStructureAnalysis(**structure_data)

    # Extract code components (handles both 'code_components' and 'components' keys)
    components = None
    if "code_components" in code_analysis:
        components_data = code_analysis["code_components"]
        components = CodeAvailabilityAnalysis(**components_data)
    elif "components" in code_analysis:
        components_data = code_analysis["components"]
        components = CodeAvailabilityAnalysis(**components_data)

    # Extract artifacts
    artifacts_obj = None
    if "artifacts" in code_analysis:
        artifacts_data = code_analysis["artifacts"]
        artifacts_obj = ArtifactsAnalysis(**artifacts_data)

    # Extract dataset splits
    dataset_splits = None
    if "dataset_splits" in code_analysis:
        splits_data = code_analysis["dataset_splits"]
        dataset_splits = DatasetSplitsAnalysis(**splits_data)

    # Extract documentation
    documentation = None
    if "documentation" in code_analysis:
        doc_data = code_analysis["documentation"]
        documentation = ReproducibilityDocumentation(**doc_data)

    return (
        methodology,
        structure,
        components,
        artifacts_obj,
        dataset_splits,
        documentation,
    )


def main():
    """Main function to compute and print reproducibility score."""

    # Default input file
    input_file = "input_analysis.json"

    # Allow command-line argument for custom file path
    if len(sys.argv) > 1:
        input_file = sys.argv[1]

    # Check if file exists
    if not Path(input_file).exists():
        print(f"Error: File '{input_file}' not found.", file=sys.stderr)
        print(f"Usage: python {sys.argv[0]} [input_file.json]", file=sys.stderr)
        sys.exit(1)

    try:
        # Load JSON data
        print(f"Loading analysis from: {input_file}")
        data = load_analysis_from_json(input_file)

        # Extract code_analysis section
        code_analysis = extract_code_analysis(data)
        if code_analysis is None:
            print(
                "Error: Could not find code_analysis section in JSON.", file=sys.stderr
            )
            sys.exit(1)

        # Parse components
        (
            methodology,
            structure,
            components,
            artifacts_obj,
            dataset_splits,
            documentation,
        ) = parse_analysis_components(code_analysis)

        # Compute reproducibility score
        score, breakdown, recommendations = compute_reproducibility_score(
            methodology=methodology,
            structure=structure,
            components=components,
            artifacts=artifacts_obj,
            dataset_splits=dataset_splits,
            documentation=documentation,
        )

        # Print results
        print("\n" + "=" * 60)
        print("REPRODUCIBILITY SCORE COMPUTATION RESULTS")
        print("=" * 60)

        print(f"\nOverall Reproducibility Score: {score}/100")

        print("\nScore Breakdown:")
        print("-" * 60)
        for component, value in breakdown.items():
            component_name = component.replace("_", " ").title()
            print(f"  {component_name:.<40} {value:>6.1f}/100")

        if recommendations:
            print("\nRecommendations for Improvement:")
            print("-" * 60)
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")

        print("\n" + "=" * 60)

        # Also output as JSON for easy parsing
        output_data = {
            "reproducibility_score": score,
            "score_breakdown": breakdown,
            "recommendations": recommendations,
        }

        output_file = input_file.replace(".json", "_output.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)

        print(f"\nResults also saved to: {output_file}")

    except Exception as e:
        print(f"Error processing file: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
