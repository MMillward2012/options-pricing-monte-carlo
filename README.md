# Options Pricing Monte Carlo Simulation

This project implements a Monte Carlo simulation engine to price European call options under the Black-Scholes framework. It includes variance reduction techniques to improve convergence speed and validates results against analytical solutions.

## Features

- Simulates asset price paths using geometric Brownian motion.
- Prices European call options by averaging discounted payoffs.
- Implements variance reduction methods (e.g., antithetic variates).
- Compares Monte Carlo results with Black-Scholes analytical prices.
- Written in Python with easy-to-follow Jupyter notebooks.

## Project Structure

- `src/`: Source code for simulation and pricing algorithms.
- `tests/`: Unit tests for code validation.
- `results/`: Output files, charts, and summary statistics.
- `notebooks/`: Jupyter notebooks demonstrating usage and experiments.
- `data/`: Input datasets or market data (if any).

## Requirements

- Python 3.10+
- numpy
- matplotlib
- scipy
- jupyter

## How to Run

1. Clone the repository.
2. Create and activate the virtual environment:  
   ```bash
   python3 -m venv .options_env
   source .options_env/bin/activate  # macOS/Linux
   .options_env\Scripts\activate     # Windows
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Launch the notebooks:
   ```bash
   jupyter notebook notebooks/main.ipynb
   ```

## About

This project was developed to deepen understanding of options pricing and Monte Carlo simulation methods, as part of preparation for quantitative finance roles.