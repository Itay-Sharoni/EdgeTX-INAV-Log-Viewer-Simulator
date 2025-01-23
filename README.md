# EdgeTX INAV Log Viewer / Simulator

> **Author:** Itay Sharoni  
> **With heavy use of ChatGPT**  

Welcome to the **EdgeTX INAV Log Viewer & Simulator** project, an all-in-one Python script for visualizing RC plane flight data from your EdgeTX CSV (and INAV telemetry) logs. This tool animates:

- **Attitude Indicator** (roll/pitch artificial horizon)  
- **GPS Track** (2D breadcrumb on a big subplot)  
- **Altitude Gauge**  
- **Throttle, Rudder, Elevator, Aileron** (sticks)  
- **Ground Speed**, **Battery Voltage**, and more!

It also lets you adjust real-time playback speed, pause/play, reset from the beginning, and even load new CSV logs on the fly.

---

## Features

- **Multi-Channel RC Sticks**: Configurable channel mapping & reversing for each axis.  
- **Attitude Indicator**: Rolls & pitches a horizon background—like a mini flight simulator.  
- **Altitude & Battery**: Simple bar showing your altitude in meters, optional battery voltage display.  
- **GPS Track**: Plots your lat/lon as a 2D “breadcrumb trail,” no map API needed.  
- **Live Playback**: Press space to pause/resume, up/down arrow to tweak speed by 0.1 steps, R to reset.  
- **Load File**: A button in the UI to pick a new CSV without restarting the script.  

---

## Requirements

You’ll need a few Python libraries to run the script:

1. **Python 3.7+** (or better; tested on 3.10)
2. **`matplotlib`** (for plotting)
3. **`numpy`** (for numeric arrays)
4. **`tkinter`** (usually built-in on most Python installs, but not always on Linux)

See [`requirements.txt`](./requirements.txt) for exact package names/versions if you want them pinned.

---

## Installation & Setup

You can set this up in **two** main ways: a virtual environment (recommended) or system-wide.

### 1) Virtual Environment (Recommended)

```bash
# On Windows, open Command Prompt (cmd) or PowerShell:
python -m venv venv
venv\Scripts\activate

# On Linux/Mac:
python3 -m venv venv
source venv/bin/activate

# Once activated, install dependencies:
pip install -r requirements.txt
```

After that, just run the script:

```bash
python sim.py
```

### 2) System-Wide (Not for the faint of heart)

```bash
# On Windows:
pip install -r requirements.txt

# On Linux (you might need sudo):
sudo pip3 install -r requirements.txt
```

Then run:

```bash
python sim.py
```

### Usage

Once you run the script, a GUI window appears:
* Load File: Button at the bottom-left. Click it to pick a CSV from EdgeTX logs.
* Auto-Play: The moment it loads, it starts animating from time=0.
* Pause / Resume: Press Space.
* Speed Adjust: Press ↑ / ↓ keys to change speed by ±0.1.
* Reset: Press R. This sets playback to time=0 but keeps the same file loaded.
* Load Another File: You can load a new CSV anytime. The script resets to the new file.


### CSV Format

See `example.csv`

```csv
Date,Time,Ptch(rad),Roll(rad),Yaw(rad),Alt(m),CH1(us),CH2(us),CH3(us),CH4(us),GPS,GSpd(kmh),RxBt(V)
```

(If you have a different name for `RxBt(V)`, change the `RX_BATT_COLNAME` variable in the script.)


### Humor & Warnings
* This script is the result of a mad scientist merging RC plane nerd knowledge with ChatGPT wizardry.
* It might give you illusions of a fancy flight simulator—but it’s just a data visualizer, folks.
* If it crashes, blame the squirrels in your CPU or your questionable CSV logs.
  
 (We disclaim all liability for flying your RC plane in your living room… you’ve been warned!)

 ### Contributing
 * Pull requests are welcome.
 * For major changes, please open an issue to discuss what you would like to change.


### License
See [`LICENSE`](./LICENSE) page
