# LLM Privacy Protection Evaluation Framework

A comprehensive framework for evaluating and testing the ability of open-source language models to protect sensitive information and resist jailbreaking attempts.

## Motivation

As language models become more capable and widely deployed, their ability to protect sensitive information while remaining helpful is crucial. This framework was developed to:

1. Test and evaluate how well open-source LLMs can protect sensitive data.
2. Evaluate the degree of hidden knowledge within LLMs about sensitive data.
3. Measure their robustness against various jailbreaking attempts
4. Provide quantitative metrics for privacy protection capabilities
5. Help identify areas where models need improvement
6. Enable comparison between different models and training approaches

## Project Structure

### Core Files

- `generate_data.py`: Generates synthetic sensitive data for testing
  - Creates a JSONL file with synthetic personal information and project secrets
  - Uses Faker library to generate realistic but fake data
  - Configurable number of data points

- `prompt_framework.py`: Defines the testing framework and prompts
  - Contains base instructions for models
  - Defines test categories (ethical framework, jailbreak attempts, etc.)
  - Includes various types of prompts to test different aspects of privacy protection
  - Structured to support easy addition of new test cases

- `model_integration.py`: Handles integration with Hugging Face models
  - Implements `HuggingFaceModel` class for easy model integration
  - Supports both CPU and GPU inference
  - Includes 8-bit quantization for memory efficiency
  - Handles prompt formatting and response generation
  - Provides few-shot examples to guide model behavior

- `evaluate_model.py`: Core evaluation logic
  - Implements metrics calculation
  - Evaluates model responses
  - Detects sensitive information leakage
  - Calculates various protection scores
  - Handles test execution and results collection

- `visualize_results.py`: Creates visualizations and reports
  - Generates radar plots of metrics
  - Creates category performance comparisons
  - Produces failure analysis heatmaps
  - Shows response length distributions
  - Generates comprehensive evaluation reports

### Supporting Files

- `requirements.txt`: Lists all project dependencies
  - Includes specific versions for compatibility
  - Covers ML, data processing, and visualization libraries

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Generate test data:
```bash
python generate_data.py
```

3. Evaluate a model:
```python
from model_integration import evaluate_model

# Evaluate a small model
evaluate_model(
    "facebook/opt-125m",
    temperature=0.7,
    max_length=512
)
```

4. Generate visualizations:
```python
from visualize_results import ResultsVisualizer

visualizer = ResultsVisualizer('evaluation_results.json')
visualizer.generate_report()
```

## Evaluation Metrics

The framework calculates several key metrics:

- **Privacy Score**: How well the model protects sensitive information
- **Consistency Score**: How consistently the model maintains privacy protection
- **Jailbreak Resistance**: How well the model resists jailbreaking attempts
- **False Positive Rate**: Rate of incorrectly blocking safe queries
- **False Negative Rate**: Rate of incorrectly allowing sensitive queries
- **Response Quality**: Balance between helpfulness and privacy protection

## Test Categories

1. **Ethical Framework Tests**
   - Basic privacy protection
   - Authority challenges

2. **Jailbreak Attempts**
   - Role play attacks
   - Technical manipulation
   - Social engineering
   - Indirect extraction
   - Hypothetical scenarios

3. **Secret Protection Tests**
   - Project information protection
   - Business strategy extraction
   - Technical detail protection
   - Pattern analysis attempts
   - Contextual project queries

4. **Advanced Challenges**
   - Multi-step attacks
   - Context manipulation

## Recommended Models for Testing

### Small Models (Quick Testing)
- `facebook/opt-125m`
- `EleutherAI/pythia-70m`
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

### Medium Models
- `meta-llama/Llama-2-7b-chat-hf`
- `tiiuae/falcon-7b-instruct`
- `microsoft/phi-2`

### Large Models
- `meta-llama/Llama-2-13b-chat-hf`
- `tiiuae/falcon-40b-instruct`
- `mistralai/Mistral-7B-Instruct-v0.2`

## Contributing

Contributions are welcome! Areas for potential improvement include:

1. Adding new test categories and prompts
2. Implementing additional evaluation metrics
3. Supporting more model architectures
4. Adding new visualization types
5. Improving the detection of sensitive information
6. Adding support for different prompt templates

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Hugging Face for providing access to open-source models
- The Faker library for generating synthetic data
- The open-source community for various tools and libraries used in this project 