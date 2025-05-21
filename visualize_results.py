import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import os
from datetime import datetime

class ResultsVisualizer:
    def __init__(self, results_file: str):
        """
        Initialize the visualizer with evaluation results.
        
        Args:
            results_file: Path to the JSON file containing evaluation results
        """
        with open(results_file, 'r') as f:
            data = json.load(f)
        self.results = data["detailed_results"]
        self.metrics = data["metrics"]
        
        # Set a clean, modern style
        plt.style.use('default')
        
        # Create output directory for plots
        self.output_dir = f"{results_file}_plots"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def plot_radar_chart(self):
        """Create a radar chart of the main metrics."""
        # Prepare data
        metrics = [
            "Privacy Score",
            "Consistency Score",
            "Jailbreak Resistance",
            "Response Quality"
        ]
        values = [
            self.metrics["privacy_score"],
            self.metrics["consistency_score"],
            self.metrics["jailbreak_resistance"],
            self.metrics["response_quality"]
        ]
        
        # Number of variables
        N = len(metrics)
        
        # Compute angle for each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Add the first value again to close the plot
        values += values[:1]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        # Plot data
        ax.plot(angles, values, linewidth=2, linestyle='solid')
        ax.fill(angles, values, alpha=0.25)
        
        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        
        # Add title
        plt.title("Model Performance Metrics", size=15, y=1.1)
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, "radar_chart.png"), bbox_inches='tight', dpi=300)
        plt.close()
    
    def plot_category_performance(self):
        """Create a bar plot showing performance across different test categories."""
        # Group results by category
        categories = {}
        for result in self.results:
            cat = result["category"]
            if cat not in categories:
                categories[cat] = {
                    "protective": 0,
                    "total": 0
                }
            categories[cat]["total"] += 1
            if result["is_protective"]:
                categories[cat]["protective"] += 1
        
        # Calculate scores
        cats = list(categories.keys())
        scores = [categories[cat]["protective"] / categories[cat]["total"] for cat in cats]
        
        # Create plot
        plt.figure(figsize=(10, 6))
        bars = plt.bar(cats, scores)
        
        # Customize plot
        plt.title("Performance by Test Category")
        plt.xlabel("Category")
        plt.ylabel("Protection Score")
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2%}',
                    ha='center', va='bottom')
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "category_performance.png"), dpi=300)
        plt.close()
    
    def plot_failure_analysis(self):
        """Create a heatmap showing failure patterns."""
        # Count different types of failures
        failure_types = {
            "sensitive_leak": 0,  # Sensitive info leaked
            "non_protective": 0,  # Not protective enough
            "false_positive": 0,  # Blocked safe queries
            "false_negative": 0   # Allowed sensitive queries
        }
        
        for result in self.results:
            if result["contains_sensitive_info"]:
                failure_types["sensitive_leak"] += 1
            if not result["is_protective"]:
                failure_types["non_protective"] += 1
            if not result["is_sensitive_query"] and not result["is_protective"]:
                failure_types["false_positive"] += 1
            if result["is_sensitive_query"] and not result["is_protective"]:
                failure_types["false_negative"] += 1
        
        # Create heatmap data
        data = np.array([[v/len(self.results) for v in failure_types.values()]])
        
        # Create plot
        plt.figure(figsize=(10, 4))
        plt.imshow(data, cmap='YlOrRd')
        
        # Add labels
        plt.xticks(range(len(failure_types)), failure_types.keys(), rotation=45, ha='right')
        plt.yticks([])
        
        # Add value labels
        for i, v in enumerate(data[0]):
            plt.text(i, 0, f'{v:.2%}', ha='center', va='center')
        
        plt.title("Failure Analysis")
        plt.colorbar(label='Failure Rate')
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "failure_analysis.png"), dpi=300)
        plt.close()
    
    def plot_response_length_distribution(self):
        """Create a histogram of response lengths."""
        # Calculate response lengths
        lengths = [len(result["response"].split()) for result in self.results]
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.hist(lengths, bins=30, edgecolor='black')
        
        # Customize plot
        plt.title("Distribution of Response Lengths")
        plt.xlabel("Number of Words")
        plt.ylabel("Frequency")
        
        # Add mean line
        mean_length = np.mean(lengths)
        plt.axvline(mean_length, color='red', linestyle='dashed', linewidth=1)
        plt.text(mean_length, plt.ylim()[1]*0.9, f'Mean: {mean_length:.1f}',
                ha='right', va='top')
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "response_lengths.png"), dpi=300)
        plt.close()
    
    def generate_report(self):
        """Generate a comprehensive report with all visualizations."""
        # Create all plots
        self.plot_radar_chart()
        self.plot_category_performance()
        self.plot_failure_analysis()
        self.plot_response_length_distribution()
        
        # Generate HTML report
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report = f"""
        <html>
        <head>
            <title>Evaluation Report - {timestamp}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .metric {{ margin: 20px 0; padding: 20px; background: #f5f5f5; border-radius: 5px; }}
                .plot {{ margin: 20px 0; text-align: center; }}
                .plot img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Model Evaluation Report</h1>
                <p>Generated on: {timestamp}</p>
                
                <div class="metric">
                    <h2>Key Metrics</h2>
                    <ul>
                        <li>Privacy Score: {self.metrics['privacy_score']:.2%}</li>
                        <li>Consistency Score: {self.metrics['consistency_score']:.2%}</li>
                        <li>Jailbreak Resistance: {self.metrics['jailbreak_resistance']:.2%}</li>
                        <li>False Positive Rate: {self.metrics['false_positive_rate']:.2%}</li>
                        <li>False Negative Rate: {self.metrics['false_negative_rate']:.2%}</li>
                        <li>Response Quality: {self.metrics['response_quality']:.2%}</li>
                    </ul>
                </div>
                
                <div class="plot">
                    <h2>Performance Metrics</h2>
                    <img src="radar_chart.png" alt="Radar Chart">
                </div>
                
                <div class="plot">
                    <h2>Category Performance</h2>
                    <img src="category_performance.png" alt="Category Performance">
                </div>
                
                <div class="plot">
                    <h2>Failure Analysis</h2>
                    <img src="failure_analysis.png" alt="Failure Analysis">
                </div>
                
                <div class="plot">
                    <h2>Response Length Distribution</h2>
                    <img src="response_lengths.png" alt="Response Lengths">
                </div>
            </div>
        </body>
        </html>
        """
        
        # Save report
        with open(os.path.join(self.output_dir, "report.html"), "w") as f:
            f.write(report)
        
        print(f"Report generated in {self.output_dir}/report.html")

def visualize_evaluation(results_file: str = "evaluation_results.json"):
    """
    Generate visualizations for evaluation results.
    
    Args:
        results_file: Path to the JSON file containing evaluation results
    """
    visualizer = ResultsVisualizer(results_file)
    visualizer.generate_report()

if __name__ == "__main__":
    visualize_evaluation() 