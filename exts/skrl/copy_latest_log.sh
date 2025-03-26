#!/bin/bash
# SRC_DIR="/home/vatanassov/RL_jumping/legged_gym/logs/test_go1"
SRC_DIR="/home/vassil/skrl/runs/torch/AnymalTerrainPathfindingSkills"
DEST_DIR="${PWD}/runs/torch/AnymalTerrainPathfindingSkills"
# server="vatanassov@deep.tudelft.nl"
server="rick"
# path=$server:$SRC_DIR/$(ssh $server 'ls -t /home/vatanassov/RL_jumping/legged_gym/logs/test_go1 | head -1 ')
# echo $server:$SRC_DIR/$(ssh $server 'ls -t /home/vatanassov/RL_jumping/legged_gym/logs/test_go1 | head -1 ')
# echo $(ls $path)
# echo $server:$SRC_DIR/$(ssh $server 'ls -t /home/vatanassov/RL_jumping/legged_gym/logs/test_go1 | head -1') $DEST_DIR
# scp -r $server:$SRC_DIR/$(ssh $server 'ls -t /home/vatanassov/RL_jumping/legged_gym/logs/test_go1 | head -1') $DEST_DIR
# rsync -avz $server:$SRC_DIR/$(ssh $server 'ls -t /home/vatanassov/RL_jumping/legged_gym/logs/test_go1 | head -1') $DEST_DIR
# rsync -avz $server:$SRC_DIR/$(ssh $server 'ls -t /home/vatanassov/RL_jumping/legged_gym/logs/test_go1 | head -1') $DEST_DIR

# directories=($(ssh "$server" "find '$SRC_DIR' -type d -mindepth 1 -maxdepth 1"))
# directories=($(ssh "$server" "find "$SRC_DIR" -mindepth 1 -maxdepth 1 -type d | tail -n 10"))
directories=($(ssh "$server"  "ls -rt "$SRC_DIR" | tail -n 10"))
# echo ${directories}
# printf "Please select folder:\n"
select d in ${directories[@]}; do test -n "$d" && break; echo ">>> Invalid Selection, selecting most recent log."; d="${directories[-1]}" && break; done
echo "You selected: $d"
# rsync -avz
latest_checkpoint=$(ssh "$server" "ls -rt $SRC_DIR/./$d/checkpoints/ | grep 0.pt | tail -1")
echo "Copying latest checkpoint: $latest_checkpoint"
rsync -avz --relative "$server:$SRC_DIR/./$d/checkpoints/$latest_checkpoint" "$DEST_DIR"
# ls $server:$SRC_DIR/$(ssh $server 'ls -t $d')