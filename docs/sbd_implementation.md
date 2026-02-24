# Semi-Blind Deconvolution (SBD) Implementation

## Overview

This implementation provides tools for Bayesian semi-blind deconvolution using Elliptical Slice Sampling (ESS) and Manifold Elliptical Slice Sampling (MESS).

## Problem Formulation

The forward model is:
```
d = W @ c + e
```

where:
- **d**: Observed data (blurred and noisy image), vectorized, shape `(n,)` where `n = n_v × n_h`
- **c**: True image (vectorized), shape `(n,)`
- **W**: 1D convolution matrix, shape `(n, n)`, formed by the blur kernel `w`
- **w**: Blur kernel, shape `(k,)` where `k ≤ n_v`
- **e**: Observation noise ~ N(0, noise_variance × I)
- **n_v**: Number of rows in the lattice (image height)
- **n_h**: Number of columns in the lattice (image width)

The prior on the image is Gaussian: `c ~ N(0, prior_var × I)`

The convolution is applied along the vertical direction (rows) independently for each column of the image.

## Files

### Core Implementation

1. **`src/multiproposal/problems/sbd.py`**: 
   - Contains the `SemiBlindDeconvolution` class
   - Implements the forward model and log-likelihood computation
   - Builds the 1D convolution matrix W from the blur kernel

2. **`src/multiproposal/data/sbd.py`**:
   - Contains `generate_sbd_data()` function
   - Generates synthetic data for testing and benchmarking
   - Creates random blur kernels and synthetic images

3. **`examples/sbd_demo.py`**:
   - Demonstration script comparing ESS and MESS
   - Includes visualization of results
   - Computes reconstruction errors

4. **`tests/test_sbd.py`**:
   - Unit tests for the SBD implementation
   - Tests data generation, problem initialization, likelihood computation, and matrix structure

## Usage

### Basic Example

```python
import numpy as np
from multiproposal.data.sbd import generate_sbd_data
from multiproposal.problems.sbd import SemiBlindDeconvolution
from multiproposal.algorithms.ess import elliptical_slice_sampling

# Generate synthetic data
data = generate_sbd_data(
    n_v=50,          # Image height
    n_h=50,          # Image width
    kernel_length=5, # Blur kernel length
    prior_var=1.0,
    noise_variance=0.1,
    seed=42,
)

# Create problem instance
problem = SemiBlindDeconvolution(
    d=data["d"],                # Observations
    w=data["w"],                # Blur kernel
    n_v=50,
    n_h=50,
    prior_var=1.0,
    noise_variance=0.1,
)

# Run ESS
rng = np.random.default_rng(42)
samples = elliptical_slice_sampling(
    problem=problem,
    initial_state=data["c_init"],
    n_samples=1000,
    rng=rng,
)

# Compute posterior mean
c_posterior_mean = np.mean(samples, axis=0)

# Reshape to image
img_reconstruction = c_posterior_mean.reshape(50, 50)
```

### Running the Demo

```bash
cd /Users/guillers/Documents/GitHub/multiproposal
python examples/sbd_demo.py
```

This will:
1. Generate synthetic SBD data
2. Run both ESS and MESS algorithms
3. Compare reconstruction quality
4. Save visualization as `sbd_demo_results.png`

### Running Tests

```bash
pytest tests/test_sbd.py -v
```

## Key Features

1. **1D Convolution Matrix**: Efficiently constructs a block-diagonal convolution matrix where each block corresponds to one column of the image

2. **Gaussian Prior**: Uses a simple isotropic Gaussian prior on image pixels

3. **Efficient Likelihood**: Computes log-likelihood using residuals without explicit matrix inverse

4. **Visualization**: Includes comprehensive plotting functions for comparing true images, observations, and reconstructions

## Mathematical Details

### Convolution Matrix Structure

The convolution matrix W is block-diagonal:
```
W = diag(W_col, W_col, ..., W_col)  (n_h blocks)
```

where each `W_col` is a Toeplitz matrix of size `(n_v, n_v)` formed by the kernel `w`:
```
W_col = [w[0]   0      0    ...   0   ]
        [w[1]  w[0]    0    ...   0   ]
        [w[2]  w[1]  w[0]   ...   0   ]
        [ ...   ...   ...   ...  ...  ]
        [ 0    ...  w[k-1] ...  w[0] ]
        [ 0    ...    0    ... w[1]  ]
```

### Log-Likelihood

```
log p(d | c, w) = -1/(2×noise_variance) × ||d - W@c||² + const
```

## Future Extensions

Potential improvements:
1. Support for 2D convolution (blur in both directions)
2. Non-Gaussian priors (e.g., sparsity-promoting priors)
3. Unknown blur kernel (full blind deconvolution)
4. Edge-aware priors for sharper reconstructions
5. Different boundary conditions for convolution
