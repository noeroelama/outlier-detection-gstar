while pgrep -f exp11_power_baselines > /dev/null; do sleep 20; done
python3 exp12_adjacent.py > ../results_v2/exp12.log 2>&1
python3 exp13_classification.py > ../results_v2/exp13.log 2>&1
python3 exp14_weights.py > ../results_v2/exp14.log 2>&1
