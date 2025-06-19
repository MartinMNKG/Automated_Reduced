
# Automated_Reduced

**Automated_Reduced** is a tool designed to generate large databases of canonical combustion cases using [Cantera](https://cantera.org/). It supports the following configurations:

- **0D Homogeneous Reactor**
- **1D Premixed Diffusion Flame**
- **1D Counterflow Diffusion Flame**

The tool automatically compares two chemical kinetic mechanisms by simulating each case, processing the results, and computing differences using various fitness functions.
It can also optimize a reduced scheme with a genetic algorithm ln a selected canonical case

---

## 📁 Project Structure

```
project/
│
├── data/
│   ├── detailed.yaml         # Detailed mechanism
│   └── reduced.yaml          # Reduced mechanism
│
├── src/
│   ├── Database/             # Tools to simulate and process the data
│   ├── Fitness/              # Tools to compute fitness metrics from processed data
│   └── GA/                   # Tools to optimize a reduced scheme 
│
├── script/
│   ├── main_launch_database_0D.py       # Launch 0D simulations
│   ├── main_launch_database_1DPMX.py    # Launch 1D premixed flame simulations
│   ├── main_launch_database_1DCF.py     # Launch 1D counterflow flame simulations
│   └── main_calculate_fitness_0D.py     # Calculate fitness for 0D simulations
│
├── script_GA/
│   ├── main_launch_GA_*.py       # Launch Genetic Algorithm
│   ├── 00_VISU_AED.py            # Visualisation of Absolute Error distribution for different optimization
│   ├── 00_VISU_FIT.py            # Visualisation of fitness function
│   └── 00_VISU_0D.py             # Visualisation of 0D reactor at selected generation
│
└── README.md
```

---

## 💻 Installation

Install required dependencies:

```bash
pip install -r rq.txt
```

---

## 🚀 Running Simulations and Processing Data

You have two options:

1. **Launch and process a new database**
2. **Load an existing database and process the data**

You can also enable CSV export for both raw Cantera simulation results and processed data.

---

## 🔬 0D Homogeneous Reactor

To launch 0D homogeneous reactor simulations:

```bash
python script/main_launch_database_0D.py
```

### Input parameters:

- Range of initial pressures
- Range of initial temperatures
- Range of equivalence ratios (ϕ)
- Range of fuel mixture ratios (e.g., NH₃/H₂)

---

## 🔥 1D Premixed Diffusion Flame

To launch premixed flame simulations:

```bash
python script/main_launch_database_1DPMX.py
```

### Input parameters:

- Range of initial pressures
- Range of initial temperatures
- Range of equivalence ratios (ϕ)
- Range of fuel mixture ratios

---

## 🌪️ 1D Counterflow Diffusion Flame

To launch counterflow diffusion flame simulations:

```bash
python script/main_launch_database_1DCF.py
```

### Input parameters:

- Range of initial pressures
- Range of initial temperatures
- Range of fuel mixture ratios

The simulation automatically increases the global strain rate to approach flame extinction, storing all intermediate flame solutions.

---

# Genetic Algorithm 
This project performs automatic optimization of a reduced chemical mechanism for bifuel 0D combustion (e.g., NH₃ + H₂) using a Genetic Algorithm (GA). The algorithm compares 0D simulations between a detailed and reduced mechanism by minimizing an error metric present in the fitness folder 

The evaluation is parallelized with MPI (mpi4py) to speed up the population assessment.

```bash
mpirun -n 4 python bazar/main_launch_GA.py
```

## 🧠 Launcher Script (main_launch_GA.py)

This script sets all the simulation and GA parameters, then calls Launch_GA() from main_GA.py.
- Key parameters:

    - Name_Folder: output folder name.

    - Fitness: fitness function to use (e.g., Calculate_AED, Calculate_PMO, etc.).
        - Type Fit : 
            - Use 'Mini' for AED, PMO, ORCh 
            - Use 'Maxi' for Brookesia (Mean or Max)

    - input_fitness: list of species to compare between mechanisms. If empty ([]), all species in the reduced mechanism are used.

    - Cantera files:

        - Detailed_file: path to the detailed mechanism .yaml file.

        - Reduced_file: path to the reduced mechanism .yaml file.

    - Combustion setup:

        - fuel1, fuel2: fuel names (e.g., "NH3", "H2").

        - oxidizer: oxidizer mixture string (e.g., "O2:0.21, N2:0.79, AR:0.01").

    - 0D Simulation:

        - tmax: end time in seconds.

        - dt: time step.

        - length: Number of datapoint of each simulation 

        - cases_0D: test cases generated via generate_test_cases_bifuel.

    - GA Parameters:

        - pop_size: population size.

        - ngen: number of generations.

        - elitism_size: number of elite individuals preserved each generation.

        - cxpb, mutpb: crossover and mutation probabilities.

    - Restart: if True, loads an existing population for continuation.

    - Hardcoded : 
        - Bounds : +- 10% from the Reduced values 
        - Gaussian mutation : mu:0, sigma:1 , indpb:0.1
        - Tournament size : 3
        - Blend : 0.5
         