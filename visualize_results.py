"""
Visualization tools for evaluation results.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Optional
import numpy as np
from pathlib import Path

class ResultsVisualizer:
    def __init__(self, results_file: str):
        """
        Initialize the visualizer with evaluation results.
        
        Args:
            results_file: Path to the JSON file containing evaluation results
        """
        with open(results_file, 'r') as f:
            self.data = json.load(f)
        
        self.metrics = self.data['metrics']
        self.results = pd.DataFrame(self.data['detailed_results'])
        
        # Set style
        plt.style.use('seaborn')
        sns.set_palette("husl")
    
    def plot_metrics_radar(self, save_path: Optional[str] = None):
        """Create a radar plot of the main metrics."""
        # Prepare data
        metrics = {
            'Privacy Score': self.metrics['privacy_score'],
            'Consistency': self.metrics['consistency_score'],
            'Jailbreak Resistance': self.metrics['jailbreak_resistance'],
            'Response Quality': self.metrics['response_quality'],
            'False Positive Rate': 1 - self.metrics['false_positive_rate'],  # Invert for better visualization
            'False Negative Rate': 1 - self.metrics['false_negative_rate']   # Invert for better visualization
        }
        
        # Number of variables
        N = len(metrics)
        
        # Compute angle for each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Prepare data
        values = list(metrics.values())
        values += values[:1]  # Close the loop
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Plot data
        ax.plot(angles, values, linewidth=2, linestyle='solid')
        ax.fill(angles, values, alpha=0.25)
        
        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics.keys())
        ax.set_ylim(0, 1)
        
        # Add title
        plt.title('Model Performance Metrics', size=15, y=1.1)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    
    def plot_category_performance(self, save_path: Optional[str] = None):
        """Create a bar plot showing performance across different test categories."""
        # Calculate average metrics by category
        category_metrics = self.results.groupby('category').agg({
            'is_protective': 'mean',
            'response_quality': 'mean',
            'contains_sensitive_info': 'mean'
        }).reset_index()
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot protection rate
        sns.barplot(data=category_metrics, x='category', y='is_protective', ax=ax1)
        ax1.set_title('Protection Rate by Category')
        ax1.set_ylabel('Protection Rate')
        ax1.set_ylim(0, 1)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Plot response quality
        sns.barplot(data=category_metrics, x='category', y='response_quality', ax=ax2)
        ax2.set_title('Response Quality by Category')
        ax2.set_ylabel('Response Quality')
        ax2.set_ylim(0, 1)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    
    def plot_failure_analysis(self, save_path: Optional[str] = None):
        """Create a heatmap showing where the model failed to protect information."""
        # Create a pivot table of failures by category and query type
        failures = self.results[self.results['contains_sensitive_info'] | ~self.results['is_protective']]
        failure_pivot = pd.crosstab(
            failures['category'],
            failures['is_sensitive_query'],
            values=failures['contains_sensitive_info'],
            aggfunc='count'
        ).fillna(0)
        
        # Create figure
        plt.figure(figsize=(10, 8))
        sns.heatmap(failure_pivot, annot=True, fmt='.0f', cmap='YlOrRd')
        plt.title('Failure Analysis by Category and Query Type')
        plt.xlabel('Sensitive Query')
        plt.ylabel('Category')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    
    def plot_response_length_distribution(self, save_path: Optional[str] = None):
        """Create a distribution plot of response lengths."""
        # Calculate response lengths
        self.results['response_length'] = self.results['response'].str.len()
        
        # Create figure
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.results, x='response_length', hue='is_protective', 
                    multiple='stack', bins=30)
        plt.title('Distribution of Response Lengths')
        plt.xlabel('Response Length (characters)')
        plt.ylabel('Count')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    
    def generate_report(self, output_dir: str = 'evaluation_report'):
        """Generate a comprehensive report with all visualizations."""
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Generate all plots
        self.plot_metrics_radar(output_path / 'metrics_radar.png')
        self.plot_category_performance(output_path / 'category_performance.png')
        self.plot_failure_analysis(output_path / 'failure_analysis.png')
        self.plot_response_length_distribution(output_path / 'response_lengths.png')
        
        # Generate summary statistics
        summary = {
            'Overall Metrics': self.metrics,
            'Category Statistics': self.results.groupby('category').agg({
                'is_protective': ['mean', 'std'],
                'response_quality': ['mean', 'std'],
                'contains_sensitive_info': ['mean', 'std']
            }).to_dict(),
            'Total Tests': len(self.results),
            'Failed Tests': len(self.results[self.results['contains_sensitive_info'] | ~self.results['is_protective']])
        }
        
        # Save summary
        with open(output_path / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Report generated in {output_dir}/")

if __name__ == "__main__":
    # Example usage
    visualizer = ResultsVisualizer('evaluation_results.json')
    visualizer.generate_report() 