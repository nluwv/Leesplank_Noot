## Commands to initialize training on the server

# connecting to server:
ssh -p 22013 hotaisle@ssh.hotaisle.cloud
# you are asked to fill  in your password that was set while creating your ssh keys

# uploading files to server:
# open a new terminal window
scp -P 22013 -r C:\Users\Path\to\the\folder hotaisle@ssh.hotaisle.cloud:/mnt/data/
# you are asked to fill  in your password

# go to folder:
cd /mnt/data

# run a python file that continues running after disconnecting:
# create a session
tmux new -s train_session

# start running script:
torchrun --nproc_per_node=8 train.py

# Detach from the tmux session (leave it running): 
Press Ctrl + b, then d.
# Check if session is still running: 
tmux list-sessions

# To reattach to your session later:
tmux attach -t train_session

# see if script is still running:
ps aux | grep python

# checking gpu usasge
rocm-smi

