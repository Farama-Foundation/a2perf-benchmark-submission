#!/bin/bash
## Internal variables.
#TIME_WAITING=0
#SLEEP_TIME=60
#HANDLER_EXIT_CODE=8
#
##NUM_COLLECT_JOBS=4
##ROOT_DIR=/rl-perf/logs/circuit_training/debug
##mkdir -p ${ROOT_DIR}
##
##export TF_FORCE_GPU_ALLOW_GROWTH=true
##export WRAPT_DISABLE_EXTENSIONS=true
##GLOBAL_SEED=0
##REVERB_PORT=8000
##REVERB_SERVER_IP=127.0.0.1
##NETLIST_FILE=/rl-perf/rl_perf/domains/circuit_training/circuit_training/environment/test_data/toy_macro_stdcell/netlist.pb.txt
##INIT_PLACEMENT=/rl-perf/rl_perf/domains/circuit_training/circuit_training/environment/test_data/toy_macro_stdcell/initial.plc
#
#echo "NUM_COLLECT_JOBS: $NUM_COLLECT_JOBS"
#echo "ROOT_DIR: $ROOT_DIR"
#echo "TF_FORCE_GPU_ALLOW_GROWTH: $TF_FORCE_GPU_ALLOW_GROWTH"
#echo "WRAPT_DISABLE_EXTENSIONS: $WRAPT_DISABLE_EXTENSIONS"
#echo "GLOBAL_SEED: $GLOBAL_SEED"
#echo "REVERB_PORT: $REVERB_PORT"
#echo "REVERB_SERVER_IP: $REVERB_SERVER_IP"
#echo "NETLIST_FILE: $NETLIST_FILE"
#echo "INIT_PLACEMENT: $INIT_PLACEMENT"
#
#handler() {
#  echo "Caught interrupt signal (likely reverb). Check logs directory ${SCRIPT_LOGS}."
#  echo "Exiting with code ${HANDLER_EXIT_CODE}."
#  exit $HANDLER_EXIT_CODE
#}
#
#usr1_handler() {
#  echo "Collect job failed (SIGUSR1). Check ${SCRIPT_LOGS}/collect_*.log."
#  echo "Exiting with code ${HANDLER_EXIT_CODE}."
#  exit $HANDLER_EXIT_CODE
#}
#
#usr2_handler() {
#  echo "Train job failed (SIGUSR2). Exiting with code ${HANDLER_EXIT_CODE}."
#  exit $HANDLER_EXIT_CODE
#}
#
#trap handler INT
#trap usr1_handler USR1
#trap usr2_handler USR2
#
#start_background() {
#  local -ir pid="$1"
#  shift
#  "$@" || kill -INT -- "$pid"
#}
#
#start_background_collect() {
#  local -ir pid="$1"
#  shift
#  "$@" || kill -SIGUSR1 -- "$pid"
#}
#
#start_background_train() {
#  local -ir pid="$1"
#  shift
#  "$@" || kill -SIGUSR2 -- "$pid"
#}
#
#REVERB_SERVER="${REVERB_SERVER_IP}:${REVERB_PORT}"
#echo "Reverb server set to $REVERB_SERVER"
#echo "Starting Reveb Server in the background."
#
#cd /rl-perf/rl_perf/rlperf_benchmark_submission/circuit_training/debug || exit
#
#start_background "$$" python3.9 -m learning.ppo_reverb_server \
#  --root_dir="$ROOT_DIR" \
#  --global_seed="$GLOBAL_SEED" \
#  --port="$REVERB_PORT" &>${ROOT_DIR}/reverb.log &
#
#echo "Starting $NUM_COLLECT_JOBS collect jobs."
#for i in $(eval echo "{1..$NUM_COLLECT_JOBS}"); do
#  echo "Start collect job $i in the background..."
#  start_background_collect "$$" python3.9 -m learning.ppo_collect \
#    --root_dir="$ROOT_DIR" \
#    --std_cell_placer_mode=dreamplace \
#    --replay_buffer_server_address="$REVERB_SERVER" \
#    --variable_container_server_address="$REVERB_SERVER" \
#    --task_id=0 \
#    --netlist_file="$NETLIST_FILE" \
#    --init_placement="$INIT_PLACEMENT" \
#    --global_seed="$GLOBAL_SEED" &>${ROOT_DIR}/collect_${i}.log &
#  echo "Logging collect job ${i} to ${SCRIPT_LOGS}/collect_${i}.log."
#done
#
#echo "Start Training job in the background but logging to console."
#start_background_train "$$" python3.9 -m learning.train_ppo \
#  --root_dir="$ROOT_DIR" \
#  --std_cell_placer_mode=dreamplace \
#  --replay_buffer_server_address="$REVERB_SERVER" \
#  --variable_container_server_address="$REVERB_SERVER" \
#  --sequence_length=3 \
#  --gin_bindings='train.num_iterations=200' \
#  --gin_bindings='train.num_episodes_per_iteration=32' \
#  --gin_bindings='train.per_replica_batch_size=64' \
#  --gin_bindings='CircuittrainingPPOLearner.summary_interval=12' \
#  --gin_bindings='CircuitPPOAgent.debug_summaries=True' \
#  --netlist_file="$NETLIST_FILE" \
#  --init_placement="$INIT_PLACEMENT" \
#  --global_seed="$GLOBAL_SEED"

REVERB_SERVER="${REVERB_SERVER_IP}:${REVERB_PORT}"
python3.9 -m learning.ppo_reverb_server \
  --root_dir="$ROOT_DIR" \
  --global_seed="$GLOBAL_SEED" \
  --port="$REVERB_PORT" >"$ROOT_DIR/server.log" 2>&1 &

python3.9 -m learning.train_ppo \
  --root_dir="$ROOT_DIR" \
  --std_cell_placer_mode=dreamplace \
  --replay_buffer_server_address="$REVERB_SERVER" \
  --variable_container_server_address="$REVERB_SERVER" \
  --sequence_length=3 \
  --gin_bindings='train.num_iterations=200' \
  --gin_bindings='train.num_episodes_per_iteration=32' \
  --gin_bindings='train.per_replica_batch_size=64' \
  --gin_bindings='CircuittrainingPPOLearner.summary_interval=12' \
  --gin_bindings='CircuitPPOAgent.debug_summaries=True' \
  --netlist_file="$NETLIST_FILE" \
  --init_placement="$INIT_PLACEMENT" \
  --global_seed="$GLOBAL_SEED" >"$ROOT_DIR/train.log" 2>&1 &

train_ppo_pid=$! # Get the process ID of the train_ppo process

for ((i = 0; i < "$NUM_COLLECT_JOBS"; i++)); do
  CUDA_VISIBLE_DEVICES=-1 python3.9 -m learning.ppo_collect \
    --root_dir="$ROOT_DIR" \
    --std_cell_placer_mode=dreamplace \
    --replay_buffer_server_address="$REVERB_SERVER" \
    --variable_container_server_address="$REVERB_SERVER" \
    --task_id=0 \
    --netlist_file="$NETLIST_FILE" \
    --init_placement="$INIT_PLACEMENT" \
    --global_seed="$GLOBAL_SEED" >"$ROOT_DIR/collect_$i.log" 2>&1 &
done

CUDA_VISIBLE_DEVICES=-1 python3.9 -m learning.eval \
  --root_dir="$ROOT_DIR" \
  --variable_container_server_address="$REVERB_SERVER" \
  --netlist_file="$NETLIST_FILE" \
  --init_placement="$INIT_PLACEMENT" \
  --global_seed="$GLOBAL_SEED" \
  --output_placement_save_dir="$ROOT_DIR" >"$ROOT_DIR/eval.log" 2>&1 &

# Wait for the train_ppo process to finish
wait "$train_ppo_pid"
