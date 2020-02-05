# kuhn_poker, incrrease, linear, different alpha
if [ $# -eq 0 ]
then
    echo "format: ./experiments.sh [game: kuhn_poker | leduc_poker] [adaptive_alpha: 1 | 0] [increase: 1 | 0] [adaptive_policy: 1 | 2] [alpha: 1]"
    printf "[adaptive_alpha]: 1: adaptive, 0: fixed"
    printf "[increase]: 1: increase, 0: decrease\n"
    printf "[adaptive_policy]: 1: linear, 2: exponential\n"
    echo "[alpha]: alpha"
    exit 1
fi

GAME=$1; shift
ADAPTIVE_ALPHA=$1; shift
INCREASE=$1; shift
ADAPTIVE_POLICY=$1; shift
ALPHA=$1
NUM_EXPERIMENTS=5
ITERATIONS=5000
OUT_DIR=ICML_Experiments_Dynamics_Pennies
POLICY=lin

if [ $ADAPTIVE_POLICY -eq 2 ]
then
    POLICY=exp
fi

python icml_experiments_multiprocessing.py \
--num_experiments=$NUM_EXPERIMENTS \
--iterations=$ITERATIONS \
--game=$GAME \
--alpha=$ALPHA \
--adaptive_alpha=$ADAPTIVE_ALPHA  \
--increase=$INCREASE \
--adaptive_policy=$ADAPTIVE_POLICY \
--out=$OUT_DIR/${GAME}_experiments/${GAME}_${ADAPTIVE_ALPHA}_${INCREASE}_${POLICY}_${ALPHA}_${ITERATIONS}.txt &
