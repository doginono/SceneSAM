#!/bin/bash

# Navigate to the project directory
cd /home/rozenberszki/D_Project/wsnsl/

# Timeout duration in seconds (3 hours = 10800 seconds)
timeout_duration=13500

# Run each command with a timeout

timeout $timeout_duration python -W ignore run.py configs/Scannet++/8b5caf3398.yaml
sleep 100


timeout $timeout_duration python -W ignore run.py configs/Scannet++/b20a261fdf.yaml
sleep 100


timeout $timeout_duration python -W ignore run.py configs/Scannet++/fe1733741f.yaml
sleep 100

timeout $timeout_duration python -W ignore run.py configs/Scannet++/41b00feddb.yaml
sleep 100


timeout $timeout_duration python -W ignore run.py configs/Scannet++/98b4ec142f.yaml
sleep 100

timeout $timeout_duration python -W ignore run.py configs/Scannet++/56a0ec536c.yaml
sleep 100


timeout $timeout_duration python -W ignore run.py configs/Scannet++/5fb5d2dbf2.yaml
sleep 100



# Print a message at the end of the script
echo "Script execution completed."
