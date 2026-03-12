#!/bin/bash

# Stop script if any command fails
set -e

echo "Starting Flower server..."
python server.py &
SERVER_PID=$!

# Cleanup function (runs on Ctrl+C or exit)
cleanup() {
  echo ""
  echo "Stopping all processes..."
  kill $SERVER_PID 2>/dev/null || true
  kill $(jobs -p) 2>/dev/null || true
  wait
  echo "Cleanup complete."
}

trap cleanup EXIT INT TERM

# Give server time to start
sleep 3

echo "Starting clients..."
for i in 0 1 2 3 4
#for i in 0 1 2 
do
  python cl.py $i &
done

# Wait for all background jobs
wait

echo "Federated Learning run complete."