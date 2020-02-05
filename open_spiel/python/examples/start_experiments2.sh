# format: 
# ./experiments.sh [game: kuhn_poker | leduc_poker] [adaptive_alpha: 1 | 0] [increase: 1 | 0] [adaptive_policy: 1 | 2 | 3] [alphas: 1 1.1 1.2 1.3 1.4] [semi_percent: 0.2 | 0.5] [exploit_rate: 100] [exp_exploit_rate: 1]
cd /root/Documents/ICML2020/open_spiel/open_spiel/python/examples
export PYTHONPATH=$PYTHONPATH:/root/Documents/ICML2020/open_spiel
export PYTHONPATH=$PYTHONPATH:/root/Documents/ICML2020/open_spiel/build/python
source ../../../venv/bin/activate
mkdir -p /root/Documents/ICML2020/open_spiel/open_spiel/python/examples/ICML_Experiments_Dynamics/kuhn_poker_experiments
mkdir -p /root/Documents/ICML2020/open_spiel/open_spiel/python/examples/ICML_Experiments_Dynamics/leduc_poker_extra_experiments

run_experiment() {
    # kuhn_poker, incrrease, linear, different alpha
    if [ $# -eq 0 ]
    then
        echo "format: ./experiments.sh [game: kuhn_poker | leduc_poker] [adaptive_alpha: 1 | 0] [increase: 1 | 0] [adaptive_policy: 1 | 2] [alpha: 1] [semi_percent: 0.2 | 0.5] [exploit_rate: 100] [exp_exploit_rate: 1]"
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
    ALPHA=$1;shift
    SEMI_PERCENT=$1;shift
    EXPLOIT_RATE=$1;shift
    EXP_EXPLOIT_RATE=$1
    NUM_EXPERIMENTS=5
    ITERATIONS=100
    OUT_DIR=ICML_Experiments_Dynamics
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
    --out=$OUT_DIR/${GAME}_extra_experiments/${GAME}_${ADAPTIVE_ALPHA}_${INCREASE}_${POLICY}_${ALPHA}_${ITERATIONS}_${SEMI_PERCENT}_${EXPLOIT_RATE}_${EXP_EXPLOIT_RATE}.txt \
    --semi_percent=$SEMI_PERCENT \
    --exploit_rate=$EXPLOIT_RATE \
    --exp_exploit_rate=$EXP_EXPLOIT_RATE &
}
# 586
#  tim1-5b5474c799-qtssh 
# 580
if [ $(hostname) = "tim6-666d9955f7-5hcpb" ] 
then
    # experiments for adaptive alpha
    for alpha in 1.02 1.04 1.06 1.08
    do
        for semi_percent in 0.2 0.5
        do
            run_experiment leduc_poker 1 1 1 $alpha $semi_percent 100 1
        done
    done

# 584
elif [ $(hostname) = "tim2-667758c9c5-876lw" ]
then
    # experiments for fixed alpha
    for exp_exploit_rate in 5 10 15 20 25 30 35 40
    do
        run_experiment leduc_poker 1 1 4 1 0.2 100 $exp_exploit_rate
    done

# 582
elif [ $(hostname) = "tim3-77549f558f-7nzdv" ]
then
    # experiments for adaptive alpha
    for semi_percent in 0.2 0.5
    do
        run_experiment leduc_poker 1 1 1 1.1 $semi_percent 100 1
    done
    for exploit_rate in 100 200 300 400
    do
        run_experiment leduc_poker 1 1 3 1 0.5 $exploit_rate 1
    done
    for exp_exploit_rate in 45 50
    do
        run_experiment leduc_poker 1 1 4 1 0.2 100 $exp_exploit_rate
    done

else
    # experiments for fixed alpha
    for exploit_rate in 500 600 700 800 900 1000
    do
        run_experiment leduc_poker 1 1 3 1 0.5 $exploit_rate 1
    done
fi

