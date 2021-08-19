#!/bin/bash

# Loading script arguments
NBCLIENTS="${1:-2}" # Nb of clients launched by the script (defaults to 2)
NBMINCLIENTS="${2:-2}" # Nb min of clients before launching round (defaults to 2)
NBFITCLIENTS="${3:-2}" # Nb of clients sampled for the round (defaults to 2)
NBROUNDS="${4:-3}" # Nb of rounds (defaults to 3)
VBATCHSIZE="${5:-256}" # Virtual batch size (defaults to 256)
BATCHSIZE="${6:-256}" # Batch size (vbatchsize%batchsize=0) (defaults to 256)
LR="${7:-0.01}" # Learning rate (defaults to 0.01)
NM="${8:-1.0}" # Noise multiplier for the Privacy Engine (defaults to 1.0)
MGN="${9:-1.2}" # Max grad norm for the Privacy Engine (defaults to 1.2)
EPS="${10:-3.0}" # Target epsilon for the privacy budget (defaults to 3.0)

python server.py -r $NBROUNDS -nbc $NBCLIENTS -b $VBATCHSIZE -fc $NBFITCLIENTS -ac $NBMINCLIENTS &
sleep 5 # Sleep for N seconds to give the server enough time to start, increase if clients can't connect
for ((nb=0; nb<$NBCLIENTS; nb++))
do
    python client.py -c $nb -nbc $NBCLIENTS -vb $VBATCHSIZE -b $BATCHSIZE -lr $LR -nm $NM -mgn $MGN -eps $EPS &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT
# If still not stopping you can use `killall python` or `killall python3` or ultimately `pkill python`
sleep 86400