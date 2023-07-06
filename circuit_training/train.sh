#!/bin/bash

REVERB_SERVER="${REVERB_SERVER_IP}:${REVERB_PORT}"
CUDA_VISIBLE_DEVICES=-1 python3.9 -m learning.ppo_reverb_server \
  --root_dir="$ROOT_DIR" \
  --global_seed="$GLOBAL_SEED" \
  --port="$REVERB_PORT" >"$ROOT_DIR/server.log" 2>&1 &

python3.9 -m learning.train_ppo \
  --root_dir="${ROOT_DIR}" \
  --std_cell_placer_mode=dreamplace \
  --replay_buffer_server_address="${REVERB_SERVER}" \
  --variable_container_server_address="${REVERB_SERVER}" \
  --sequence_length=134 \
  --gin_bindings='train.num_iterations=200' \
  --netlist_file="${NETLIST_FILE}" \
  --init_placement="${INIT_PLACEMENT}" \
  --global_seed="${GLOBAL_SEED}" >"$ROOT_DIR/train.log" 2>&1 &

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
