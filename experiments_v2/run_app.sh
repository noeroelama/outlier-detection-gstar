while pgrep -f "run_rest.sh" > /dev/null; do sleep 20; done
python3 exp20_rainfall.py > ../results_v2/exp20.out 2>&1
