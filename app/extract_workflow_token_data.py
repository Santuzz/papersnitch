#!/usr/bin/env python
"""
Extract token usage data from workflow nodes for the latest run of each paper
using paper_processing_with_reproducibility workflow, grouped by node category.

Usage:
    docker exec -it django-web-dev-bolelli python extract_workflow_token_data.py
"""

import os
import sys
import django

# Setup Django environment
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web.settings")
django.setup()

from django.db.models import Max
from workflow_engine.models import WorkflowDefinition, WorkflowRun, WorkflowNode
from django.db.models import Subquery, OuterRef


# Node groupings
NODE_GROUPS = {
    "paper": [
        "paper_type_classification",
        "section_embeddings",
        "reproducibility_checklist",
        "final_aggregation",
    ],
    "code": ["code_embedding", "code_repository_analysis"],
    "dataset": ["dataset_documentation_check"],
}


def extract_token_data():
    """
    Extract token usage data from the latest workflow run for each paper
    using paper_processing_with_reproducibility workflow.
    Groups nodes by category (paper, code, dataset) and sums tokens.
    """

    # Step 1: Find the workflow definition
    print("=" * 80)
    print("STEP 1: Finding WorkflowDefinition")
    print("=" * 80)

    # Get all versions of paper_processing_with_reproducibility
    version = 8
    workflow_defs = WorkflowDefinition.objects.filter(
        name="paper_processing_with_reproducibility", version=version
    )

    if not workflow_defs.exists():
        print(
            "✗ ERROR: WorkflowDefinition 'paper_processing_with_reproducibility' not found!"
        )
        print("\nAvailable workflow definitions:")
        for wd in WorkflowDefinition.objects.all():
            print(f"  - {wd.name} (v{wd.version})")
        return

    print(
        f"✓ Found {workflow_defs.count()} versions of paper_processing_with_reproducibility:"
    )
    for wd in workflow_defs:
        print(f"  - v{wd.version} (ID: {wd.id}, Active: {wd.is_active})")

    # # Use the latest version
    # workflow_def = workflow_defs.first()
    # print(f"\n  Using latest version: v{workflow_def.version}")
    # print(f"  ID: {workflow_def.id}")
    # print(f"  Created: {workflow_def.created_at}")
    # print(f"  Active: {workflow_def.is_active}")

    # Step 2: Find all papers with workflow runs
    print("\n" + "=" * 80)
    print("STEP 2: Finding Papers with WorkflowRuns")
    print("=" * 80)
    workflow_def = workflow_defs.first()  # Use the first version for filtering runs

    # Group by paper and get latest run for each
    paper_ids = (
        WorkflowRun.objects.filter(workflow_definition=workflow_def)
        .order_by("paper_id")
        .values_list("paper_id", flat=True)
        .distinct()
    )
    print(f"✓ Found {len(paper_ids)} papers with workflow runs\n")

    # Step 3: Process each paper's latest run
    print("=" * 80)
    print("STEP 3: Processing Latest Run for Each Paper")
    print("=" * 80)

    all_papers_data = []

    for paper_id in [
        "2660",
        "430",
        "405",
        "402",
        "322",
        "157",
        "129",
        "47",
        "35",
        "29",
    ]:

        # Get ALL runs for this paper to check count
        all_paper_runs = WorkflowRun.objects.filter(
            workflow_definition=workflow_def, paper_id=paper_id
        ).order_by("-created_at")

        run_count = all_paper_runs.count()

        # Get ONLY the latest (most recent) run for this paper

        if paper_id == 402 and version == 9:
            latest_run = all_paper_runs[1]
        else:
            latest_run = all_paper_runs.first()

        if not latest_run:
            continue

        print(f"\n--- Paper ID: {paper_id} ---")
        if run_count > 1:
            print(f"⚠ Note: {run_count} runs found for this paper, using latest")
        print(f"Paper Title: {latest_run.paper.title[:80]}...")
        print(f"Run ID: {latest_run.id}")
        print(f"Status: {latest_run.status}")
        print(f"Run Number: {latest_run.run_number}")
        print(f"Created: {latest_run.created_at}")

        # Get all nodes for this run
        nodes = WorkflowNode.objects.filter(workflow_run=latest_run)

        if not nodes.exists():
            print("  ⚠ No nodes found for this run")
            continue

        # Group and sum tokens by category
        grouped_tokens = {
            "paper": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "nodes": [],
            },
            "code": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "nodes": [],
            },
            "dataset": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "nodes": [],
            },
        }

        for node in nodes:
            # Find which group this node belongs to
            node_group = None
            for group_name, node_ids in NODE_GROUPS.items():
                if node.node_id in node_ids:
                    node_group = group_name
                    break

            if node_group:
                grouped_tokens[node_group]["input_tokens"] += node.input_tokens
                grouped_tokens[node_group]["output_tokens"] += node.output_tokens
                grouped_tokens[node_group]["total_tokens"] += node.total_tokens
                grouped_tokens[node_group]["nodes"].append(
                    {
                        "node_id": node.node_id,
                        "status": node.status,
                        "input_tokens": node.input_tokens,
                        "output_tokens": node.output_tokens,
                        "total_tokens": node.total_tokens,
                        "was_cached": node.was_cached,
                    }
                )

        # Display grouped results
        print("\n  Token Usage by Group:")
        for group_name in ["paper", "code", "dataset"]:
            group_data = grouped_tokens[group_name]
            print(f"\n  [{group_name.upper()}]")
            print(f"    Input Tokens:  {group_data['input_tokens']:,}")
            print(f"    Output Tokens: {group_data['output_tokens']:,}")
            print(f"    Total Tokens:  {group_data['total_tokens']:,}")
            print(f"    Nodes: {len(group_data['nodes'])}")

        # Store for JSON export
        paper_data = {
            "paper_id": paper_id,
            "paper_title": latest_run.paper.title,
            "workflow_run_id": str(latest_run.id),
            "run_number": latest_run.run_number,
            "status": latest_run.status,
            "created_at": latest_run.created_at,
            "started_at": latest_run.started_at,
            "completed_at": latest_run.completed_at,
            "grouped_tokens": grouped_tokens,
        }
        all_papers_data.append(paper_data)

    # Step 4: Calculate aggregate statistics
    print("\n" + "=" * 80)
    print("STEP 4: Aggregate Statistics Across All Papers")
    print("=" * 80)

    aggregate_by_group = {
        "paper": {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "paper_count": 0,
        },
        "code": {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "paper_count": 0,
        },
        "dataset": {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "paper_count": 0,
        },
    }

    for paper_data in all_papers_data:
        for group_name in ["paper", "code", "dataset"]:
            group_tokens = paper_data["grouped_tokens"][group_name]

            # Only count papers that have tokens in this group
            if group_tokens["total_tokens"] > 0:
                aggregate_by_group[group_name]["input_tokens"] += group_tokens[
                    "input_tokens"
                ]
                aggregate_by_group[group_name]["output_tokens"] += group_tokens[
                    "output_tokens"
                ]
                aggregate_by_group[group_name]["total_tokens"] += group_tokens[
                    "total_tokens"
                ]
                aggregate_by_group[group_name]["paper_count"] += 1

    # Calculate averages first
    for group_name in ["paper", "code", "dataset"]:
        group_data = aggregate_by_group[group_name]
        paper_count = group_data["paper_count"]

        if paper_count > 0:
            group_data["avg_input"] = group_data["input_tokens"] / paper_count
            group_data["avg_output"] = group_data["output_tokens"] / paper_count
            group_data["avg_total"] = group_data["total_tokens"] / paper_count
        else:
            group_data["avg_input"] = 0
            group_data["avg_output"] = 0
            group_data["avg_total"] = 0

    # Calculate variance (second pass through data)
    for group_name in ["paper", "code", "dataset"]:
        group_data = aggregate_by_group[group_name]
        paper_count = group_data["paper_count"]

        if paper_count > 0:
            var_input_sum = 0
            var_output_sum = 0
            var_total_sum = 0

            for paper_data in all_papers_data:
                group_tokens = paper_data["grouped_tokens"][group_name]

                if group_tokens["total_tokens"] > 0:
                    var_input_sum += (
                        group_tokens["input_tokens"] - group_data["avg_input"]
                    ) ** 2
                    var_output_sum += (
                        group_tokens["output_tokens"] - group_data["avg_output"]
                    ) ** 2
                    var_total_sum += (
                        group_tokens["total_tokens"] - group_data["avg_total"]
                    ) ** 2

            group_data["var_input"] = var_input_sum / paper_count
            group_data["var_output"] = var_output_sum / paper_count
            group_data["var_total"] = var_total_sum / paper_count

            # Calculate standard deviation as well
            group_data["std_input"] = group_data["var_input"] ** 0.5
            group_data["std_output"] = group_data["var_output"] ** 0.5
            group_data["std_total"] = group_data["var_total"] ** 0.5
        else:
            group_data["var_input"] = 0
            group_data["var_output"] = 0
            group_data["var_total"] = 0
            group_data["std_input"] = 0
            group_data["std_output"] = 0
            group_data["std_total"] = 0

    print(f"\nTotal Papers Processed: {len(all_papers_data)}")
    print("\nAverage Token Usage per Paper by Group:")

    grand_total_input = 0
    grand_total_output = 0
    total_papers_with_tokens = 0

    for group_name in ["paper", "code", "dataset"]:
        group_data = aggregate_by_group[group_name]
        paper_count = group_data["paper_count"]

        if paper_count > 0:
            print(f"\n[{group_name.upper()}]")
            print(f"  Papers with tokens: {paper_count}")
            print(f"  Avg Input Tokens:   {group_data['avg_input']:,.2f}")
            print(f"  Avg Output Tokens:  {group_data['avg_output']:,.2f}")
            print(f"  Avg Total Tokens:   {group_data['avg_total']:,.2f}")
            print(f"  Var Input Tokens:   {group_data['var_input']:,.2f}")
            print(f"  Var Output Tokens:  {group_data['var_output']:,.2f}")
            print(f"  Var Total Tokens:   {group_data['var_total']:,.2f}")
            print(f"  Std Input Tokens:   {group_data['std_input']:,.2f}")
            print(f"  Std Output Tokens:  {group_data['std_output']:,.2f}")
            print(f"  Std Total Tokens:   {group_data['std_total']:,.2f}")

            grand_total_input += group_data["input_tokens"]
            grand_total_output += group_data["output_tokens"]
        else:
            print(f"\n[{group_name.upper()}]")
            print(f"  Papers with tokens: 0")
            print(f"  Avg Input Tokens:   0.00")
            print(f"  Avg Output Tokens:  0.00")
            print(f"  Avg Total Tokens:   0.00")
            print(f"  Var Input Tokens:   0.00")
            print(f"  Var Output Tokens:  0.00")
            print(f"  Var Total Tokens:   0.00")

    # Calculate papers that contributed to at least one group
    papers_with_any_tokens = set()
    for paper_data in all_papers_data:
        for group_name in ["paper", "code", "dataset"]:
            if paper_data["grouped_tokens"][group_name]["total_tokens"] > 0:
                papers_with_any_tokens.add(paper_data["paper_id"])

    total_papers_with_tokens = len(papers_with_any_tokens)

    if total_papers_with_tokens > 0:
        grand_avg_input = grand_total_input / total_papers_with_tokens
        grand_avg_output = grand_total_output / total_papers_with_tokens
        grand_avg_total = (
            grand_total_input + grand_total_output
        ) / total_papers_with_tokens

        # Calculate variance for grand average
        grand_var_input_sum = 0
        grand_var_output_sum = 0
        grand_var_total_sum = 0

        for paper_data in all_papers_data:
            if paper_data["paper_id"] in papers_with_any_tokens:
                # Sum tokens across all groups for this paper
                paper_total_input = sum(
                    paper_data["grouped_tokens"][group]["input_tokens"]
                    for group in ["paper", "code", "dataset"]
                )
                paper_total_output = sum(
                    paper_data["grouped_tokens"][group]["output_tokens"]
                    for group in ["paper", "code", "dataset"]
                )
                paper_total_tokens = sum(
                    paper_data["grouped_tokens"][group]["total_tokens"]
                    for group in ["paper", "code", "dataset"]
                )

                grand_var_input_sum += (paper_total_input - grand_avg_input) ** 2
                grand_var_output_sum += (paper_total_output - grand_avg_output) ** 2
                grand_var_total_sum += (paper_total_tokens - grand_avg_total) ** 2

        grand_var_input = grand_var_input_sum / total_papers_with_tokens
        grand_var_output = grand_var_output_sum / total_papers_with_tokens
        grand_var_total = grand_var_total_sum / total_papers_with_tokens

        grand_std_input = grand_var_input**0.5
        grand_std_output = grand_var_output**0.5
        grand_std_total = grand_var_total**0.5

        print(
            f"\n[GRAND AVERAGE (across {total_papers_with_tokens} papers with tokens)]"
        )
        print(f"  Avg Input Tokens:   {grand_avg_input:,.2f}")
        print(f"  Avg Output Tokens:  {grand_avg_output:,.2f}")
        print(f"  Avg Total Tokens:   {grand_avg_total:,.2f}")
        print(f"  Var Input Tokens:   {grand_var_input:,.2f}")
        print(f"  Var Output Tokens:  {grand_var_output:,.2f}")
        print(f"  Var Total Tokens:   {grand_var_total:,.2f}")
        print(f"  Std Input Tokens:   {grand_std_input:,.2f}")
        print(f"  Std Output Tokens:  {grand_std_output:,.2f}")
        print(f"  Std Total Tokens:   {grand_std_total:,.2f}")
    else:
        print(f"\n[GRAND AVERAGE]")
        print(f"  No papers with tokens found")
        grand_avg_input = 0
        grand_avg_output = 0
        grand_avg_total = 0
        grand_var_input = 0
        grand_var_output = 0
        grand_var_total = 0
        grand_std_input = 0
        grand_std_output = 0
        grand_std_total = 0

    print("=" * 80)

    # Step 5: Export to JSON
    print("\n" + "=" * 80)
    print("STEP 5: Exporting to JSON")
    print("=" * 80)

    import json
    from datetime import datetime

    def serialize_datetime(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Type {type(obj)} not serializable")

    export_data = {
        "workflow_definition": {
            "id": str(workflow_def.id),
            "name": workflow_def.name,
            "version": workflow_def.version,
        },
        "node_groups": NODE_GROUPS,
        "papers": all_papers_data,
        "aggregate_statistics": {
            "total_papers": len(all_papers_data),
            "papers_with_tokens": total_papers_with_tokens,
            "by_group": {
                group_name: {
                    "paper_count": group_data["paper_count"],
                    "total_input_tokens": group_data["input_tokens"],
                    "total_output_tokens": group_data["output_tokens"],
                    "total_tokens": group_data["total_tokens"],
                    "avg_input_tokens": group_data["avg_input"],
                    "avg_output_tokens": group_data["avg_output"],
                    "avg_total_tokens": group_data["avg_total"],
                    "var_input_tokens": group_data["var_input"],
                    "var_output_tokens": group_data["var_output"],
                    "var_total_tokens": group_data["var_total"],
                    "std_input_tokens": group_data["std_input"],
                    "std_output_tokens": group_data["std_output"],
                    "std_total_tokens": group_data["std_total"],
                }
                for group_name, group_data in aggregate_by_group.items()
            },
            "grand_total": {
                "total_input_tokens": grand_total_input,
                "total_output_tokens": grand_total_output,
                "total_tokens": grand_total_input + grand_total_output,
            },
            "grand_average": {
                "papers_with_tokens": total_papers_with_tokens,
                "avg_input_tokens": grand_avg_input,
                "avg_output_tokens": grand_avg_output,
                "avg_total_tokens": grand_avg_total,
                "var_input_tokens": grand_var_input,
                "var_output_tokens": grand_var_output,
                "var_total_tokens": grand_var_total,
                "std_input_tokens": grand_std_input,
                "std_output_tokens": grand_std_output,
                "std_total_tokens": grand_std_total,
            },
        },
    }

    output_file = "/app/workflow_token_data.json"
    with open(output_file, "w") as f:
        json.dump(export_data, f, indent=2, default=serialize_datetime)

    print(f"✓ Data exported to: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    extract_token_data()
