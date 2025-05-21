
# Automated_Reduced

**Automated_Reduced** is a tool designed to generate large databases of canonical combustion cases using [Cantera](https://cantera.org/). It supports the following configurations:

- **0D Homogeneous Reactor**
- **1D Premixed Diffusion Flame**
- **1D Counterflow Diffusion Flame**

The tool automatically compares two chemical kinetic mechanisms by simulating each case, processing the results, and computing differences using various fitness functions.

---

## \U0001f4c1 Project Structure

```
project/
\u2502
\u251c\u2500\u2500 data/
\u2502   \u251c\u2500\u2500 detailed.yaml         # Detailed mechanism
\u2502   \u2514\u2500\u2500 reduced.yaml          # Reduced mechanism
\u2502
\u251c\u2500\u2500 src/
\u2502   \u251c\u2500\u2500 Database/             # Tools to simulate and process the data
\u2502   \u2514\u2500\u2500 Fitness/              # Tools to compute fitness metrics from processed data
\u2502
\u251c\u2500\u2500 script/
\u2502   \u251c\u2500\u2500 main_launch_database_0D.py       # Launch 0D simulations
\u2502   \u251c\u2500\u2500 main_launch_database_1DPMX.py    # Launch 1D premixed flame simulations
\u2502   \u251c\u2500\u2500 main_launch_database_1DCF.py     # Launch 1D counterflow flame simulations
\u2502   \u2514\u2500\u2500 main_calculate_fitness_0D.py     # Calculate fitness for 0D simulations
\u2502
\u2514\u2500\u2500 README.md
```

---

## \U0001f4bb Installation

Install required dependencies:

```bash
pip install -r rq.txt
```

---

## \U0001f680 Running Simulations and Processing Data

You have two options:

1. **Launch and process a new database**
2. **Load an existing database and process the data**

You can also enable CSV export for both raw Cantera simulation results and processed data.

---

## \U0001f52c 0D Homogeneous Reactor

To launch 0D homogeneous reactor simulations:

```bash
python script/main_launch_database_0D.py
```

### Input parameters:

- Range of initial pressures
- Range of initial temperatures
- Range of equivalence ratios (\u03d5)
- Range of fuel mixture ratios (e.g., NH\u2083/H\u2082)

---

## \U0001f525 1D Premixed Diffusion Flame

To launch premixed flame simulations:

```bash
python script/main_launch_database_1DPMX.py
```

### Input parameters:

- Range of initial pressures
- Range of initial temperatures
- Range of equivalence ratios (\u03d5)
- Range of fuel mixture ratios

---

## \U0001f32a\ufe0f 1D Counterflow Diffusion Flame

To launch counterflow diffusion flame simulations:

```bash
python script/main_launch_database_1DCF.py
```

### Input parameters:

- Range of initial pressures
- Range of initial temperatures
- Range of fuel mixture ratios

The simulation automatically increases the global strain rate to approach flame extinction, storing all intermediate flame solutions.
