"""
Visualization tools for evaluation results.
Generates plots and HTML reports for model evaluation metrics.
"""

import json
import os
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List
import numpy as np
from dataclasses import asdict

class ResultsVisualizer:
    def __init__(self, results_file: str):
        """Initialize visualizer with evaluation results."""
        with open(results_file, 'r') as f:
            data = json.load(f)
        self.results = data["detailed_results"]
        self.metrics = data["metrics"]
        self.output_dir = "evaluation_plots"
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_data_type_metrics(self):
        """Plot metrics for each data type (SSN, ZIP, project)."""
        data_types = list(self.metrics["data_type_metrics"].keys())
        metrics = ["base_accuracy", "jailbreak_accuracy", "jailbreak_resistance", 
                  "false_positive_rate", "false_negative_rate", "response_quality"]
        
        # Prepare data
        data = {metric: [] for metric in metrics}
        for data_type in data_types:
            type_metrics = self.metrics["data_type_metrics"][data_type]
            for metric in metrics:
                data[metric].append(type_metrics[metric])
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Metrics by Data Type', fontsize=16)
        
        # Plot each metric
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]
            bars = ax.bar(data_types, data[metric])
            ax.set_title(metric.replace('_', ' ').title())
            ax.set_ylim(0, 1)
            ax.set_xticklabels(data_types, rotation=45)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2%}',
                       ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'data_type_metrics.png'))
        plt.close()

    def plot_jailbreak_metrics(self):
        """Plot jailbreak attempt success rates by type and data type."""
        jailbreak_types = list(self.metrics["jailbreak_metrics"].keys())
        data_types = list(self.metrics["data_type_metrics"].keys())
        
        # Prepare data
        success_rates = np.zeros((len(jailbreak_types), len(data_types)))
        for i, jailbreak_type in enumerate(jailbreak_types):
            for j, data_type in enumerate(data_types):
                if data_type in self.metrics["jailbreak_metrics"][jailbreak_type]["data_type_breakdown"]:
                    success_rates[i, j] = self.metrics["jailbreak_metrics"][jailbreak_type]["data_type_breakdown"][data_type]
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        plt.imshow(success_rates, cmap='RdYlGn_r', aspect='auto')
        plt.colorbar(label='Success Rate')
        
        # Add labels
        plt.xticks(range(len(data_types)), data_types, rotation=45)
        plt.yticks(range(len(jailbreak_types)), jailbreak_types)
        plt.title('Jailbreak Success Rates by Type and Data Type')
        
        # Add value labels
        for i in range(len(jailbreak_types)):
            for j in range(len(data_types)):
                plt.text(j, i, f'{success_rates[i, j]:.2%}',
                        ha='center', va='center',
                        color='black' if success_rates[i, j] < 0.5 else 'white')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'jailbreak_metrics.png'))
        plt.close()

    def plot_overall_metrics(self):
        """Plot overall evaluation metrics."""
        metrics = {
            'Overall Accuracy': self.metrics["overall_accuracy"],
            'Response Quality': self.metrics["response_quality"]
        }
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics.keys(), metrics.values())
        plt.ylim(0, 1)
        plt.title('Overall Evaluation Metrics')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2%}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'overall_metrics.png'))
        plt.close()

    def generate_html_report(self):
        """Generate a comprehensive HTML report with all visualizations and metrics."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Start HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .section {{ margin: 20px 0; padding: 20px; background: #f5f5f5; border-radius: 5px; }}
                .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }}
                .metric-card {{ background: white; padding: 15px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
                .metric-label {{ color: #7f8c8d; }}
                img {{ max-width: 100%; height: auto; margin: 10px 0; }}
                table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f8f9fa; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Model Evaluation Report</h1>
                <p>Generated on: {timestamp}</p>
                
                <div class="section">
                    <h2>Overall Metrics</h2>
                    <div class="metric-grid">
                        <div class="metric-card">
                            <div class="metric-value">{self.metrics["overall_accuracy"]:.2%}</div>
                            <div class="metric-label">Overall Accuracy</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{self.metrics["response_quality"]:.2%}</div>
                            <div class="metric-label">Response Quality</div>
                        </div>
                    </div>
                    <img src="overall_metrics.png" alt="Overall Metrics">
                </div>
                
                <div class="section">
                    <h2>Data Type Analysis</h2>
                    <img src="data_type_metrics.png" alt="Data Type Metrics">
                    <h3>Detailed Metrics by Data Type</h3>
                    <table>
                        <tr>
                            <th>Data Type</th>
                            <th>Base Accuracy</th>
                            <th>Jailbreak Accuracy</th>
                            <th>Jailbreak Resistance</th>
                            <th>False Positive Rate</th>
                            <th>False Negative Rate</th>
                            <th>Response Quality</th>
                        </tr>
        """
        
        # Add data type metrics
        for data_type, metrics in self.metrics["data_type_metrics"].items():
            html_content += f"""
                        <tr>
                            <td>{data_type.upper()}</td>
                            <td>{metrics["base_accuracy"]:.2%}</td>
                            <td>{metrics["jailbreak_accuracy"]:.2%}</td>
                            <td>{metrics["jailbreak_resistance"]:.2%}</td>
                            <td>{metrics["false_positive_rate"]:.2%}</td>
                            <td>{metrics["false_negative_rate"]:.2%}</td>
                            <td>{metrics["response_quality"]:.2%}</td>
                        </tr>
            """
        
        html_content += """
                    </table>
                </div>
                
                <div class="section">
                    <h2>Jailbreak Analysis</h2>
                    <img src="jailbreak_metrics.png" alt="Jailbreak Metrics">
                    <h3>Jailbreak Success Rates</h3>
                    <table>
                        <tr>
                            <th>Jailbreak Type</th>
                            <th>Total Attempts</th>
                            <th>Success Rate</th>
                        </tr>
        """
        
        # Add jailbreak metrics
        for jailbreak_type, metrics in self.metrics["jailbreak_metrics"].items():
            html_content += f"""
                        <tr>
                            <td>{jailbreak_type}</td>
                            <td>{metrics["total_tests"]}</td>
                            <td>{metrics["success_rate"]:.2%}</td>
                        </tr>
            """
        
        html_content += """
                    </table>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Save HTML report
        with open(os.path.join(self.output_dir, 'evaluation_report.html'), 'w') as f:
            f.write(html_content)

    def visualize(self):
        """Generate all visualizations and the HTML report."""
        self.plot_overall_metrics()
        self.plot_data_type_metrics()
        self.plot_jailbreak_metrics()
        self.generate_html_report()
        print(f"Visualizations and report generated in {self.output_dir}/")

# Example usage:
if __name__ == "__main__":
    visualizer = ResultsVisualizer("evaluation_results.json")
    visualizer.visualize() 