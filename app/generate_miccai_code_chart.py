"""
Generate bar chart showing code availability for MICCAI conferences by year.

For each MICCAI conference edition, displays:
- Total number of papers
- Number of papers with GitHub link available (code_url)
- Number of papers with actual code accessible (code_text or code embeddings)
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
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Use non-interactive backend for server environments
matplotlib.use('Agg')


def get_miccai_statistics():
    """
    Query database and compute statistics for MICCAI conferences.
    
    Returns:
        List of tuples: (year, total_papers, with_github_link, with_actual_code)
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
        
        # Papers with actual code available
        # Method 1: Check if code_text field has content
        with_code_text = papers.filter(
            code_text__isnull=False
        ).exclude(code_text='').count()
        
        # Method 2: Check if code embeddings exist
        paper_ids_with_embeddings = CodeFileEmbedding.objects.filter(
            paper__conference=conference
        ).values_list('paper_id', flat=True).distinct()
        
        with_code_embeddings = papers.filter(id__in=paper_ids_with_embeddings).count()
        
        # Use the maximum to avoid undercounting (some might have text, some embeddings)
        with_actual_code = max(with_code_text, with_code_embeddings)
        
        print(f"  {conference.name}")
        print(f"    Year: {year}")
        print(f"    Total papers: {total_papers}")
        print(f"    With GitHub link: {with_github_link} ({with_github_link/total_papers*100:.1f}%)")
        print(f"    With actual code: {with_actual_code} ({with_actual_code/total_papers*100:.1f}%)")
        print(f"      - With code_text: {with_code_text}")
        print(f"      - With code embeddings: {with_code_embeddings}")
        print()
        
        stats.append((year, total_papers, with_github_link, with_actual_code))
    
    return stats


def generate_bar_chart(stats, output_file='miccai_code_availability.png'):
    """
    Generate grouped bar chart from statistics.
    
    Args:
        stats: List of tuples (year, total_papers, with_github_link, with_actual_code)
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
    bars3 = ax.bar(x + width, with_code, width, label='With Actual Code', 
                   color='#e74c3c', alpha=0.8)
    
    # Customize chart
    ax.set_xlabel('Year', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Papers', fontsize=14, fontweight='bold')
    ax.set_title('MICCAI Code Availability by Year', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(years, fontsize=12)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{int(height)}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=9, fontweight='bold')
    
    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)
    
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
        stats: List of tuples (year, total_papers, with_github_link, with_actual_code)
    """
    print("\n" + "="*80)
    print("MICCAI CODE AVAILABILITY SUMMARY")
    print("="*80)
    print(f"{'Year':<10} {'Total':>10} {'GitHub Link':>15} {'Actual Code':>15} {'Link %':>10} {'Code %':>10}")
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
