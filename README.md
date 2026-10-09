# Image-Guided Optimization and Hardware Control

A Python research prototype for image-based feedback, iterative
parameter optimization, and software control of an experimental imaging
setup.

The project focuses on the engineering workflow around **image
processing, objective-function design, numerical optimization, and
hardware--software integration**. The experimental setting provides the
use case; the repository is intended to make the software and
algorithmic work understandable and reusable.

> **Project status:** Research code under organization and validation.
> The supplied source includes hardware-dependent control paths and
> experiment-specific assumptions. A fully reproducible,
> hardware-independent installation has not yet been established.

## What the code does

The current source files implement or explore the following components:

-   **Image analysis:** Read image data from a camera/commander object,
    visualize 2D intensity arrays, estimate an intensity-weighted image
    center, and calculate threshold-based spot-size measures.
-   **Image-quality objectives:** Evaluate image properties such as
    thresholded area, standard deviation, kurtosis, and Laplacian-based
    sharpness.
-   **Image comparison metrics:** Implement image comparison using MSE,
    MAE, PSNR, SSIM, Dice-style overlap loss, KL divergence, and a
    cumulative-distribution-based distance calculation.
-   **Iterative optimization:** Adjust selected parameters in positive
    and negative directions, compare objective values, and reduce the
    step size across optimization stages.
-   **Parameter control:** Provide helper routines to inspect, reset,
    set, increment, or scale Zernike coefficients on a connected
    controller.
-   **Reference-image generation:** Generate idealized Airy-disk or
    Gaussian intensity profiles for comparison and analysis.
-   **Visualization and measurement output:** Plot image arrays and
    overlays, and save selected measurements or arrays to
    experiment-specific locations.

These capabilities are based on the supplied Python source. They should
be treated as implemented research routines, not as a claim that every
mode is tested, robust, or suitable for general-purpose use.

## Engineering workflow

The code follows an image-feedback loop:

1.  **Acquire an image** from the connected imaging system.
2.  **Measure image properties** and/or compare the image with a
    reference profile.
3.  **Evaluate an objective function** to quantify the current result.
4.  **Perturb selected control parameters** and acquire/evaluate
    subsequent images.
5.  **Keep or reverse parameter updates** based on the objective value,
    with step-size reduction across stages.
6.  **Inspect and save results** for further analysis.

The source contains separate routines for focus-related optimization and
image-position optimization. Their behavior depends on the selected
metric, threshold settings, controller interface, and experimental
configuration.

## Technology

The supplied source uses:

-   Python
-   NumPy
-   SciPy
-   Matplotlib
-   scikit-image for selected image metrics
-   A hardware-facing `commander` interface for experimental image
    acquisition and Zernike control

The exact environment and full dependency list still need to be
verified. The existing notebook also references experiment-specific
libraries and local paths, so the notebook should not be assumed to run
on a clean machine as-is.

## Repository map

The original files currently cover these responsibilities:

  -----------------------------------------------------------------------
  Source file                         Observed responsibility
  ----------------------------------- -----------------------------------
  `main.py`                           Optimization routines, control
                                      flow, experimental helper functions

  `Optimizer.py`                      Overlapping/alternate optimization
                                      routines and measurement helpers

  `Img.py`                            Image reading, plotting, saving,
                                      center and radius estimates

  `Airy.py`                           Airy-disk and Gaussian
                                      reference-profile calculations

  `Zernike.py`                        Helper commands for manipulating
                                      Zernike coefficients

  `advanced_slm.ipynb`                Experimental notebook, analyses,
                                      outputs, and hardware-oriented
                                      workflows

  `note.md`                           Notes on selected image-comparison
                                      metrics
  -----------------------------------------------------------------------

The source currently has overlapping logic between `main.py` and
`Optimizer.py`. The eventual repository structure should separate
reusable algorithms from hardware adapters, experiments, tests, and
generated results. This README describes the code's current scope; it
does not imply that the refactoring or test suite is already complete.

## Running the project

### Current state

The supplied code is not yet packaged as a portable, one-command
installation. Some functions expect:

-   A live `commander` object exposing camera image data and a
    hologram/Zernike interface
-   Experiment-specific hardware or vendor libraries
-   Specific folder paths for figures and measurement output
-   Image inputs and parameter settings that are currently embedded in
    the research code

For these reasons, no universal run command is provided yet. The setup
instructions will be added after the hardware interface, dependencies,
file paths, and an offline example have been separated and verified.

### Intended reproducible usage

The next engineering milestone is to provide an offline example that
loads a saved image, runs selected image-analysis and objective-function
routines, and produces a result without requiring the physical setup.
Hardware-backed optimization will be documented separately and will
require the appropriate experimental environment.

## Evaluation and results

The project includes routines that calculate image properties and
objective values, but a reliable headline performance result should only
be reported once the evaluation procedure and supporting data are
identified and checked.

The results section should ultimately include:

-   Before/after images from a documented run
-   Objective-value or convergence plots
-   Quantitative changes in the selected image-quality measurements
-   Runtime and iteration counts where measured
-   A clear distinction between physical experiments, offline analysis,
    and simulated/reference images

No universal performance improvement is claimed in this README.

## Tests and reliability

A test suite and clean-environment validation have not yet been
established for the supplied source. Before treating the repository as
application-ready, the core numerical routines should be tested
independently of hardware, and the controller-dependent code should be
isolated behind a documented interface.

Areas for validation include input shapes and ranges, zero-intensity
images, threshold behavior, metric direction (whether larger or smaller
is better), invalid controller state, and parameter-update behavior.

## Why this project is relevant beyond its experimental setting

The project offers a concrete setting for demonstrating transferable
engineering skills:

-   Turning a practical task into measurable objectives
-   Implementing and comparing numerical objective functions
-   Building iterative search and parameter-update logic
-   Processing and visualizing image data
-   Connecting software logic to an external hardware interface
-   Separating offline analysis from experiment-dependent execution
-   Improving maintainability, testing, and reproducibility in research
    code

This is **not presented as a trained AI/ML model**. Its strongest
current story is algorithmic software, image processing, optimization,
and hardware--software integration---skills relevant to computer vision,
automation, applied algorithms, and AI-enabled systems.

## Scientific and experimental context

The deeper optical background, experimental setup, hardware details, and
physical interpretation belong in separate documentation (for example,
`docs/experimental_setup.md`). The main README keeps that context brief
so readers can first understand the software problem, workflow, and
engineering contributions.

## Limitations

-   The code is research-oriented and contains experiment-specific
    assumptions.
-   Some execution paths require physical hardware or unavailable local
    dependencies.
-   `main.py` and `Optimizer.py` contain overlapping functionality that
    should be reconciled during refactoring.
-   The available code alone does not establish that every objective
    function or optimization mode has been validated.
-   Reproducibility and performance claims must be tied to a documented
    run and its supporting data.

## Development roadmap

1.  Inventory the hardware interface, dependencies, data, and existing
    figures.
2.  Separate hardware-independent image analysis from hardware-specific
    control.
3.  Consolidate duplicate optimization code and define clear interfaces.
4.  Add tests for core metrics and parameter-update logic.
5.  Create an offline demonstration with saved input images and
    generated plots.
6.  Document verified experimental results and setup instructions.
