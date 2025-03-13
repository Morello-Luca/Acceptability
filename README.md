# Optimization of Weights for the Bradley-Terry Model

This project implements an optimization framework to estimate weights (or "satisfaction scores") for various features based on pairwise ranking data using the Bradley-Terry model. The goal is to determine how well a set of weights explains the observed ranking data by maximizing the log-likelihood of the model. This README provides an in-depth explanation of the methodology, the optimization techniques used, and the key parameters involved.

---

## Table of Contents

1. [Overview](#overview)
2. [Bradley-Terry Model](#bradley-terry-model)
3. [Why Use Log-Likelihood?](#why-use-log-likelihood)
4. [Optimization Techniques](#optimization-techniques)
    - [Global Optimization](#global-optimization)
    - [Local Optimization](#local-optimization)
5. [Advanced Strategies](#advanced-strategies)
    - [Multi-start Strategy](#multi-start-strategy)
    - [Comparison of Results](#comparison-of-results)
    - [Hybrid Strategy](#hybrid-strategy)
6. [Parameters and Their Roles](#parameters-and-their-roles)
7. [Project Workflow](#project-workflow)
8. [Installation and Execution](#installation-and-execution)
9. [Conclusion](#conclusion)

---

## 1. Overview

The project aims to optimize the weights assigned to different features (e.g., types of audio) based on respondents’ rankings. The optimization process is driven by maximizing the likelihood (or equivalently minimizing the negative log-likelihood) of observing the given ranking data under the Bradley-Terry model. A combination of global and local optimization techniques is employed to ensure that the solution is robust and not stuck in local minima.

---

## 2. Bradley-Terry Model

The Bradley-Terry model is used to estimate the probability that one item is preferred over another in a pairwise comparison. The probability that item `i` is preferred over item `j` is given by:

$\[
P(i \text{ preferred over } j) = \frac{w_i}{w_i + w_j}
\]$

where \(w_i\) and \(w_j\) are the weights corresponding to items \(i\) and \(j\), respectively. Each respondent's ranking data provides multiple pairwise comparisons, and the overall likelihood of the observed data is computed as the product of these probabilities.

---

## 3. Why Use Log-Likelihood?

When dealing with probabilities, multiplying many small numbers can lead to numerical underflow (i.e., the computed product becomes too small for the computer to represent accurately). By taking the logarithm of probabilities:

- **Numerical Stability:**  
  The product of probabilities is transformed into a sum of logarithms, which avoids underflow issues.

- **Ease of Optimization:**  
  Sums are generally easier to differentiate and optimize compared to products. In our implementation, we minimize the negative log-likelihood, which is equivalent to maximizing the log-likelihood.

---

## 4. Optimization Techniques

To solve for the optimal weights, we use both global and local optimization methods.

### Global Optimization

- **Differential Evolution (DE):**  
  - **Description:** A population-based global optimization algorithm that explores the entire parameter space.  
  - **Benefits:**  
    - Useful for non-convex functions that may contain multiple local minima.  
    - Provides a robust search across a wide range of values.  
  - **Implementation Note:**  
    - DE requires finite bounds. In our implementation, we set:  
      ```python
      bounds = [(0, 10)] * num_weights
      ```
      where `num_weights` is the number of features. This defines the search space for each weight.

### Local Optimization

Local optimization methods refine a solution starting from a given initial guess. We primarily use two methods:

- **L-BFGS-B:**  
  - **Overview:** A limited-memory quasi-Newton method that approximates the Hessian matrix efficiently.  
  - **Advantages:**  
    - Very efficient for smooth, differentiable functions.  
    - Naturally handles bound constraints (e.g., weights must be non-negative).  
  - **Usage:** Ideal for moderately sized problems where the objective function is well-behaved.

- **SLSQP (Sequential Least Squares Programming):**  
  - **Overview:** A gradient-based optimization method that can manage both bound constraints and additional non-linear constraints.  
  - **Advantages:**  
    - Effective when there are complex constraints beyond simple bounds.  
    - Useful for non-linear optimization problems.
  - **Usage:** Particularly suitable when the problem includes constraints such as equality or inequality restrictions beyond non-negativity.

Other local optimization methods exist (like Trust-Region or Newton-CG), but L-BFGS-B and SLSQP are commonly used due to their balance of performance and ease of implementation.

---

## 5. Advanced Strategies

### Multi-start Strategy

- **Purpose:**  
  To mitigate the risk of converging to a local minimum, especially in non-convex problems.
- **Implementation:**  
  The optimization is run multiple times (with `n_starts = 5` by default) using different random starting points. Each run explores a different region of the parameter space.
- **Benefit:**  
  Enhances the probability of finding the global optimum by sampling diverse initial conditions.

### Comparison of Results

- **Approach:**  
  Each optimization run (whether from a global or local method) returns a result with an objective function value (negative log-likelihood).  
- **Outcome:**  
  By comparing these values, we can determine which run produced the best (lowest) negative log-likelihood. This comparative approach builds confidence in the robustness of the solution.

### Hybrid Strategy

- **Concept:**  
  Combine global and local optimization methods to leverage their respective strengths.
- **Implementation:**  
  1. **Global Search:** Use Differential Evolution to perform a broad search across the entire parameter space.
  2. **Local Refinement:** Take the best result from the global search as a starting point for a local optimizer (typically L-BFGS-B) to further refine the solution.
- **Benefit:**  
  The hybrid approach ensures that the solution is not only globally competitive but also finely tuned, leading to improved accuracy and stability.

---

## 6. Parameters and Their Roles

- **n_starts = 5:**  
  Specifies the number of random initializations for local optimization.  
  - **Impact:**  
    - A higher number increases the likelihood of escaping local minima but also increases computation time.

- **bounds = [(0, 10)] * num_weights:**  
  Defines the lower and upper limits for each weight.  
  - **Impact:**  
    - Ensures that weights remain within a practical range (0 to 10).  
    - Required by global methods like Differential Evolution to establish a finite search space.

- **General Behavior:**  
  - The process begins by reading the ranking data from a CSV file.
  - The negative log-likelihood is computed based on the Bradley-Terry model.
  - Global optimization provides a robust initial solution.
  - Multiple local optimizations (using different methods and starting points) further explore the parameter space.
  - Results from all runs are compared, and the best solution is selected and refined.
  - Finally, the optimized weights are normalized (maximum weight scaled to 1) and used to rank the features.

---

## 7. Project Workflow

1. **Data Loading:**  
   The CSV file (e.g., `list.csv`) is read, with the first column containing respondent identifiers and the following columns containing ranking data for each feature.

2. **Global Optimization:**  
   Differential Evolution is used to perform a coarse search of the entire parameter space, with weights constrained by `bounds`.

3. **Local Optimization (Multi-start):**  
   Multiple local optimizations are performed from different random starting points using methods such as L-BFGS-B and SLSQP.

4. **Result Comparison:**  
   The outcomes from each optimization run are recorded and compared based on their negative log-likelihood values.

5. **Hybrid Refinement:**  
   The best solution from the previous steps is further refined using a local optimization method (typically L-BFGS-B).

6. **Normalization and Ranking:**  
   The final weights are normalized (dividing by the maximum weight) and used to produce a ranking of the features.

---

## 8. Installation and Execution

### Installation

1. **Clone or Download the Repository:**  
   Download the source code to your local machine.

2. **Install Dependencies:**  
   Ensure that you have Python 3.7+ installed, then run:
   ```bash
   pip install -r requirements.txt
