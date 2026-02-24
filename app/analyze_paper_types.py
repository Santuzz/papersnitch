#!/usr/bin/env python3
"""
Analyze paper type distribution across all conferences.

This script queries the database for all paper type classifications
and generates a chart showing the distribution.
"""

import os
import sys
import django
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Setup Django
sys.path.insert(0, '/home/administrator/papersnitch/app')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'webApp.settings')
django.setup()

from workflow_engine.models import WorkflowNode, NodeArtifact, WorkflowRun
from webApp.models import Paper, Conference


def get_paper_type_distribution():
    """
    Extract paper type classifications from NodeArtifacts.
    
    Returns:
        dict: Paper type counts
        dict: Paper type to paper IDs mapping
        int: Total papers with classification
    """
    # Query all nodes with node_id='paper_type_classification' that are completed
    classification_nodes = WorkflowNode.objects.filter(
        node_id='paper_type_classification',
        status='completed'
    ).select_related('workflow_run')
    
    print(f"Found {classification_nodes.count()} completed paper type classification nodes")
    
    paper_types = []
    paper_type_details = []
    papers_by_type = {}
    
    for node in classification_nodes:
        # Get the result artifact
        artifacts = NodeArtifact.objects.filter(
            node=node,
            name='result'
        )
        
        for artifact in artifacts:
            if artifact.inline_data and 'paper_type' in artifact.inline_data:
                paper_type = artifact.inline_data['paper_type']
                confidence = artifact.inline_data.get('confidence', 0.0)
                reasoning = artifact.inline_data.get('reasoning', '')
                
                paper_types.append(paper_type)
                
                # Get paper info
                workflow_run = node.workflow_run
                
                try:
                    paper = workflow_run.paper
                    paper_id = paper.id
                    paper_title = paper.title
                    paper_conference = str(paper.conference) if paper.conference else 'Unknown'
                except (Paper.DoesNotExist, AttributeError):
                    paper = None
                    paper_id = None
                    paper_title = 'Unknown (Paper Deleted)'
                    paper_conference = 'Unknown'
                
                detail = {
                    'paper_type': paper_type,
                    'confidence': confidence,
                    'reasoning': reasoning[:100] if reasoning else '',
                    'paper_id': paper_id,
                    'workflow_run_id': str(workflow_run.id),
                    'title': paper_title,
                    'conference': paper_conference
                }
                
                paper_type_details.append(detail)
                
                # Track papers by type
                if paper_type not in papers_by_type:
                    papers_by_type[paper_type] = []
                papers_by_type[paper_type].append(detail)
    
    type_counts = Counter(paper_types)
    return type_counts, papers_by_type, paper_type_details


def print_statistics(type_counts, papers_by_type, paper_type_details):
    """Print detailed statistics about paper types."""
    total = sum(type_counts.values())
    
    print("\n" + "="*80)
    print("PAPER TYPE DISTRIBUTION ANALYSIS")
    print("="*80)
    print(f"\nTotal papers analyzed: {total}")
    print(f"\nPaper types found: {list(type_counts.keys())}")
    
    print("\n" + "-"*80)
    print("DISTRIBUTION BY TYPE:")
    print("-"*80)
    
    for paper_type, count in type_counts.most_common():
        percentage = (count / total * 100) if total > 0 else 0
        print(f"  {paper_type:15s}: {count:4d} papers ({percentage:5.1f}%)")
    
    # Check for theoretical papers
    theoretical_count = type_counts.get('theoretical', 0)
    print("\n" + "-"*80)
    if theoretical_count > 0:
        print(f"✓ FOUND {theoretical_count} THEORETICAL PAPERS")
        print("-"*80)
        print("\nTheoretical papers:")
        for detail in papers_by_type.get('theoretical', []):
            print(f"\n  Paper ID: {detail['paper_id']}")
            print(f"  Title: {detail.get('title', 'Unknown')[:80]}")
            print(f"  Conference: {detail.get('conference', 'Unknown')}")
            print(f"  Confidence: {detail['confidence']:.2f}")
            print(f"  Reasoning: {detail['reasoning']}")
    else:
        print("✗ NO THEORETICAL PAPERS FOUND")
    print("-"*80)
    
    # Conference breakdown
    print("\n" + "-"*80)
    print("DISTRIBUTION BY CONFERENCE:")
    print("-"*80)
    
    conferences = {}
    for detail in paper_type_details:
        conf_name = detail.get('conference', 'Unknown')
        if conf_name not in conferences:
            conferences[conf_name] = Counter()
        conferences[conf_name][detail['paper_type']] += 1
    
    for conf_name in sorted(conferences.keys()):
        conf_types = conferences[conf_name]
        conf_total = sum(conf_types.values())
        print(f"\n  {conf_name}:")
        print(f"    Total: {conf_total} papers")
        for paper_type, count in conf_types.most_common():
            percentage = (count / conf_total * 100) if conf_total > 0 else 0
            print(f"      {paper_type:15s}: {count:4d} ({percentage:5.1f}%)")


def create_chart(type_counts, output_path='paper_type_distribution.png'):
    """Create a bar chart showing paper type distribution."""
    if not type_counts:
        print("No data to plot")
        return
    
    # Sort by count descending
    sorted_types = type_counts.most_common()
    types = [t[0] for t in sorted_types]
    counts = [t[1] for t in sorted_types]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar chart
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    bars = ax1.bar(types, counts, color=colors[:len(types)], edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax1.set_xlabel('Paper Type', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Papers', fontsize=14, fontweight='bold')
    ax1.set_title('Paper Type Distribution Across All Conferences', 
                  fontsize=16, fontweight='bold', pad=20)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.tick_params(axis='both', labelsize=11)
    
    # Rotate x-axis labels if needed
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Pie chart
    total = sum(counts)
    percentages = [(c/total)*100 for c in counts]
    
    wedges, texts, autotexts = ax2.pie(
        counts, 
        labels=types,
        autopct='%1.1f%%',
        colors=colors[:len(types)],
        startangle=90,
        textprops={'fontsize': 11, 'weight': 'bold'}
    )
    
    # Make percentage text more visible
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(11)
        autotext.set_weight('bold')
    
    ax2.set_title('Paper Type Proportions', 
                  fontsize=16, fontweight='bold', pad=20)
    
    # Add total count as subtitle
    fig.suptitle(f'Total Papers Analyzed: {total}', 
                fontsize=12, y=0.02, style='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Chart saved to: {output_path}")
    
    # Also save in the media directory if it exists
    media_path = '/home/administrator/papersnitch/media/paper_type_distribution.png'
    try:
        import os
        os.makedirs(os.path.dirname(media_path), exist_ok=True)
        plt.savefig(media_path, dpi=300, bbox_inches='tight')
        print(f"✓ Chart also saved to: {media_path}")
    except Exception as e:
        print(f"⚠ Could not save to media directory: {e}")
    
    plt.close()


def main():
    """Main execution function."""
    print("Starting paper type analysis...")
    print("Querying database...\n")
    
    # Get distribution data
    type_counts, papers_by_type, paper_type_details = get_paper_type_distribution()
    
    if not type_counts:
        print("\n⚠ No paper type classifications found in the database.")
        print("   Papers need to be processed through the workflow first.")
        return
    
    # Print statistics
    print_statistics(type_counts, papers_by_type, paper_type_details)
    
    # Create chart
    print("\n" + "="*80)
    print("GENERATING CHART...")
    print("="*80)
    create_chart(type_counts)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
