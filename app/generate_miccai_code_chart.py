"""
Generate bar chart showing code availability for MICCAI conferences by year.

For each MICCAI conference edition, displays:
- Total number of papers
- Number of papers with GitHub link in database (code_url field)
- Number of papers with verified, accessible code (from workflow verification)
  * Verified means: URL is accessible (not 404), repository contains actual code files
  * Excludes: empty repos, 404 links, repos with only docs/data
"""

import os
import sys
import django
from collections import defaultdict

# Setup Django
sys.path.insert(0, '/home/administrator/papersnitch/app')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'webApp.settings')
django.setup()

from django.db.models import Q, Count, Case, When, IntegerField
from webApp.models import Paper, Conference, CodeFileEmbedding
from workflow_engine.models import WorkflowRun, WorkflowNode, NodeArtifact
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Use non-interactive backend for server environments
matplotlib.use('Agg')


def get_miccai_statistics():
    """
    Query database and compute statistics for MICCAI conferences.
    
    Returns:
        List of tuples: (year, total_papers, with_github_link, with_verified_code)
        
    Note:
        - with_github_link: Papers with code_url field populated
        - with_verified_code: Papers where code_availability_check node verified:
            * URL is accessible (not 404)
            * Repository contains actual code files (not empty, not just docs)
    """
    # Get all MICCAI conferences
    miccai_conferences = Conference.objects.filter(
        Q(name__icontains='MICCAI') | Q(acronym__icontains='MICCAI')
    ).order_by('year')
    
    if not miccai_conferences.exists():
        print("⚠️  No MICCAI conferences found in database!")
        return []
    
    print(f"Found {miccai_conferences.count()} MICCAI conference(s):\n")
    
    stats = []
    
    for conference in miccai_conferences:
        year = conference.year or 'N/A'
        
        # Get all papers for this conference
        papers = Paper.objects.filter(conference=conference)
        total_papers = papers.count()
        
        # Papers with GitHub link (code_url is not empty)
        with_github_link = papers.filter(
            code_url__isnull=False
        ).exclude(code_url='').count()
        
        # Papers with actual code verified (from workflow node results)
        # This checks if code_availability_check node returned code_available=True
        # which means: URL exists, is accessible (not 404), and contains actual code files
        papers_with_verified_code = 0
        papers_checked = 0
        
        for paper in papers:
            # Get latest completed workflow run for this paper
            latest_run = WorkflowRun.objects.filter(
                paper=paper,
                status='completed'
            ).order_by('-created_at').first()
            
            if latest_run:
                # Get code_availability_check node
                code_check_node = WorkflowNode.objects.filter(
                    workflow_run=latest_run,
                    node_id='code_availability_check',
                    status='completed'
                ).first()
                
                if code_check_node:
                    papers_checked += 1
                    # Get result artifact
                    result_artifact = NodeArtifact.objects.filter(
                        node=code_check_node,
                        name='result'
                    ).first()
                    
                    if result_artifact and result_artifact.inline_data:
                        result = result_artifact.inline_data
                        if result.get('code_available') == True:
                            papers_with_verified_code += 1
        
        with_actual_code = papers_with_verified_code
        
        print(f"  {conference.name}")
        print(f"    Year: {year}")
        print(f"    Total papers: {total_papers}")
        print(f"    With GitHub link: {with_github_link} ({with_github_link/total_papers*100:.1f}%)")
        print(f"    With verified code: {with_actual_code} ({with_actual_code/total_papers*100:.1f}%)")
        print(f"      - Papers checked by workflow: {papers_checked}")
        print(f"      - Broken/empty links: {with_github_link - with_actual_code if papers_checked > 0 else 'N/A'}")
        print()
        
        stats.append((year, total_papers, with_github_link, with_actual_code))
    
    return stats


def generate_bar_chart(stats, output_file='miccai_code_availability.pdf'):
    """
    Generate grouped bar chart from statistics.
    
    Args:
        stats: List of tuples (year, total_papers, with_github_link, with_verified_code)
        output_file: Path to save the chart image
    """
    if not stats:
        print("No data to plot!")
        return
    
    # Extract data
    years = [str(year) for year, _, _, _ in stats]
    total_papers = [total for _, total, _, _ in stats]
    with_github = [github for _, _, github, _ in stats]
    with_code = [code for _, _, _, code in stats]
    
    # Set up the bar chart
    x = np.arange(len(years))
    width = 0.25  # Width of bars
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create bars
    bars1 = ax.bar(x - width, total_papers, width, label='Total Papers', 
                   color='#3498db', alpha=0.8)
    bars2 = ax.bar(x, with_github, width, label='With GitHub Link', 
                   color='#2ecc71', alpha=0.8)
    bars3 = ax.bar(x + width, with_code, width, label='With Verified Code', 
                   color='#e74c3c', alpha=0.8)
    
    # Customize chart - enlarged fonts
    ax.set_xlabel('Year', fontsize=18, fontweight='bold')
    ax.set_ylabel('Number of Papers', fontsize=18, fontweight='bold')
    ax.set_title('MICCAI Code Availability by Year', fontsize=20, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(years, fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.legend(fontsize=16, loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars - enlarged with percentages
    def add_labels(bars, totals=None):
        for i, bar in enumerate(bars):
            height = bar.get_height()
            if height > 0:
                # If totals provided, calculate and show percentage
                if totals:
                    pct = (height / totals[i] * 100) if totals[i] > 0 else 0
                    label = f'{int(height)}\n({pct:.1f}%)'
                else:
                    label = f'{int(height)}'
                
                ax.annotate(label,
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=11, fontweight='bold')
    
    add_labels(bars1)  # Total papers - no percentage
    add_labels(bars2, total_papers)  # GitHub links with percentage
    add_labels(bars3, total_papers)  # Verified code with percentage
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Chart saved to: {output_file}")
    
    # Also try to display if in interactive mode
    try:
        plt.show()
    except:
        pass


def generate_summary_table(stats):
    """
    Print a summary table to the console.
    
    Args:
        stats: List of tuples (year, total_papers, with_github_link, with_verified_code)
    """
    print("\n" + "="*80)
    print("MICCAI CODE AVAILABILITY SUMMARY")
    print("="*80)
    print(f"{'Year':<10} {'Total':>10} {'GitHub Link':>15} {'Verified Code':>15} {'Link %':>10} {'Code %':>10}")
    print("-"*80)
    
    total_all = 0
    total_github = 0
    total_code = 0
    
    for year, total, github, code in stats:
        github_pct = (github / total * 100) if total > 0 else 0
        code_pct = (code / total * 100) if total > 0 else 0
        
        print(f"{str(year):<10} {total:>10} {github:>15} {code:>15} {github_pct:>9.1f}% {code_pct:>9.1f}%")
        
        total_all += total
        total_github += github
        total_code += code
    
    print("-"*80)
    
    if total_all > 0:
        avg_github_pct = (total_github / total_all * 100)
        avg_code_pct = (total_code / total_all * 100)
        print(f"{'TOTAL':<10} {total_all:>10} {total_github:>15} {total_code:>15} {avg_github_pct:>9.1f}% {avg_code_pct:>9.1f}%")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate bar chart for MICCAI code availability statistics'
    )
    parser.add_argument(
        '--output', '-o',
        default='miccai_code_availability.png',
        help='Output filename for the chart (default: miccai_code_availability.png)'
    )
    parser.add_argument(
        '--table-only',
        action='store_true',
        help='Only print table, do not generate chart'
    )
    
    args = parser.parse_args()
    
    # Get statistics from database
    stats = get_miccai_statistics()
    
    if not stats:
        print("No MICCAI conferences found or no papers available.")
        sys.exit(1)
    
    # Generate summary table
    generate_summary_table(stats)
    
    # Generate bar chart (unless table-only mode)
    if not args.table_only:
        generate_bar_chart(stats, args.output)
    else:
        print("Skipping chart generation (--table-only mode)")
