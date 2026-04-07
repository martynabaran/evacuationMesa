# Simple Visualization Enhancements

## Overview

This project adds two simple visualization improvements to make the evacuation simulation easier to understand.

## Enhancement 1: Health Bars

**What it does:** Shows a small colored bar above each agent to display their current health.

**How it works:**
- Green bar = healthy (>50% health)
- Yellow bar = injured (20-50% health)
- Red bar = critical (<20% health)
- Bar length shows exact health percentage

**Code location:** `run.py` lines 103-127

**Why this is useful:** You can quickly see which agents are in danger without checking numbers.

## Enhancement 2: Agent State Tracking

**What it does:** Displays counts of agents in panic or rescue states on the metrics graph.

**How it works:**
- Orange dotted line = number of panicking agents
- Purple dotted line = number of agents rescuing relatives

**Code location:** `run.py` lines 130-142

**Why this is useful:** Shows when agents lose their paths (panic) or help family members (rescue).

## Running the Simulation

### Basic run (no visualization):
```bash
python3 run.py
```

### With visualization (see the enhancements):
```bash
python3 run.py --visualize
```

## Files Modified

- `run.py` - Added health bars and state tracking to visualization (~30 lines added)
- `room_layouts/supermarket3.txt` - Test building layout (new file)
- `config.yaml` - Configuration file (new file)
- `.gitignore` - Ignore cache files (new file)

## Technical Details

### Health Bar Implementation
Uses matplotlib Rectangle patches to draw two rectangles:
1. Red background (full bar)
2. Colored foreground (scaled by health percentage)

Simple calculation:
```python
health_pct = agent.health / 100.0
bar_width = bar_width * health_pct
```

### State Tracking Implementation
The metrics were already being collected by the model's DataCollector:
- `Panicking` counter
- `Rescuing` counter

We just added them to the visualization plot with dotted lines to distinguish from main metrics.

## Benefits

1. **Visual feedback:** See agent health without reading numbers
2. **Emergency detection:** Quickly spot agents in critical condition
3. **Behavior monitoring:** Track when agents panic or rescue others
4. **Student-friendly:** Simple, easy-to-understand code additions

## Future Ideas (Optional)

Simple extensions you could add:
- Different marker shapes for different states (circle, square, triangle)
- Show agent age with marker size
- Color fade effect as health decreases
- Sound alert when agents reach critical health
