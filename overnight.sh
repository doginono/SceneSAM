cd /home/rozenberszki/project/wsnsl

python -W ignore run.py  configs/Own/room0_panoptic.yaml


sleep 10000

cd /home/rozenberszki/D_Project/wsnsl

python -W ignore run.py  configs/Scannet++/56a0ec536c.yaml

