
# Automated_Reduced

**Automated_Reduced** is a tool designed to generate large databases of canonical combustion cases using [Cantera](https://cantera.org/). It supports the following configurations:

- **0D Homogeneous Reactor**
- **1D Premixed Diffusion Flame**
- **1D Counterflow Diffusion Flame**

The tool automatically compares two chemical kinetic mechanisms by simulating each case, processing the results, and computing differences using various fitness functions.

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
│   └── Fitness/              # Tools to compute fitness metrics from processed data
│
├── script/
│   ├── main_launch_database_0D.py       # Launch 0D simulations
│   ├── main_launch_database_1DPMX.py    # Launch 1D premixed flame simulations
│   ├── main_launch_database_1DCF.py     # Launch 1D counterflow flame simulations
│   └── main_calculate_fitness_0D.py     # Calculate fitness for 0D simulations
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