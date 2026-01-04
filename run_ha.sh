#!/bin/bash
export TRADING_MODE=PAPER
truncate -s 0 /var/log/ha.log
truncate -s 0 cd /root/FyersHA/ha.log
cd /root/FyersHA
source venv/bin/activate
python3.11 -u main.py run 2>&1 | tee -a ha.log