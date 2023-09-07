# Intel Vtune

## Installation (Linux)
```
sudo sh ./l_oneapi_vtune_p_2023.0.0.25339.sh
echo 'source /opt/intel/oneapi/setvars.sh' >> ~/.bashrc
```

After every boot perform the following commands,
```
sudo sysctl -w kernel.yama.ptrace_scope=0
```

Execute `vtune-gui` to start the profiler

## Installation (Windows)
- Run w_oneapi_vtune_p_2023.0.0.25541.exe
- Run Program Files (x86)\Intel\oneAPI\setvars.bat
- Start vtune-gui from start menu

## Getting Started
- Pro Tip: Run Performance Snapshot first to get a view of what is the greatest bottleneck
- Tutorial at [vtune-tutorial.mkv](vtune-tutorial.mkv)

## Problem Statement

### Part 1

- Run the performance snapshot analysis over the sample matrix multiplication code provided. And report the IPC, bad speculation, logical core utilization.
- Now, run hotspot detection on the same code and report top hotspot function along with percentage of CPU time.
- Next, run the microarchitecture exploration tool and report the number of instructions retired, average CPU frequency, effective logical core utilization and explain the difference with the logical core utilization found in performance snapshot analysis.
- For each of these parts, you need to show screenshots to validate your numbers and briefly explain what these metrics are and what information do you gain from them. 

### Part 2

- You are given a C++ program that implements Linear Regression
- The code is written to be intentionally awkward and clunky in places
- You are to use VTune to find the hotspots in the code and identify places where time is being spent unnecessarily
- You are to optimize the code as much as you can, without changing the algorithm used to train the model
- Datasets can be generated using the ```generate_data.py``` script
- The C++ program prints the training accuracy and the time taken to train the model: Submissions that train the model in the least time, without compromising the training accuracy will be awarded marks
