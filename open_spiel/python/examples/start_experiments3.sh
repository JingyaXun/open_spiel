# format: 
# ./experiments.sh [game: kuhn_poker | leduc_poker] [adaptive_alpha: 1 | 0] [increase: 1 | 0] [adaptive_policy: 1 | 2 | 3] [alphas: 1 1.1 1.2 1.3 1.4] [semi_percent: 0.2 | 0.5] [exploit_rate: 100] [exp_exploit_rate: 1]
cd /home/open_spiel/open_spiel/python/examples
export PYTHONPATH=$PYTHONPATH:/home/open_spiel
export PYTHONPATH=$PYTHONPATH:/home/open_spiel/build/python
source ../../../venv/bin/activate
mkdir -p /home/open_spiel/open_spiel/python/examples/ICML_Experiments_Appendix/kuhn_poker_players_experiments
# mkdir -p /root/Documents/ICML2020/open_spiel/open_spiel/python/examples/ICML_Experiments_Dynamics_2players/leduc_poker_experiments

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
    EXP_EXPLOIT_RATE=$1;shift
    NUM_PLAYER=$1
    NUM_EXPERIMENTS=5
    ITERATIONS=30000
    OUT_DIR=ICML_Experiments_Appendix
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
    --out=$OUT_DIR/${GAME}_players_experiments/${GAME}_${ADAPTIVE_ALPHA}_${INCREASE}_${POLICY}_${ALPHA}_${ITERATIONS}_${SEMI_PERCENT}_${EXPLOIT_RATE}_${EXP_EXPLOIT_RATE}_${NUM_PLAYER}.txt \
    --semi_percent=$SEMI_PERCENT \
    --exploit_rate=$EXPLOIT_RATE \
    --exp_exploit_rate=$EXP_EXPLOIT_RATE \
    --players=$NUM_PLAYER &
}

# 580 / 574
if [ $(hostname) = "zhihan3-5b96b546d8-gvhk5" ] 
then
    # fixed alpha
    run_experiment kuhn_poker 1 1 1 1.2 0.5 100 1 4

# 584 *
elif [ $(hostname) = "tim2-574f54ccd9-w2wvf" ]
then
    # fixed alpha
    for alpha in 1 1.1 1.2 1.3 1.4
    do
        run_experiment kuhn_poker 0 1 1 $alpha 0.5 100 1 2
    done

    # linear increase
    for alpha in 1.1 1.2 1.3 1.4
    do
        run_experiment kuhn_poker 1 1 1 $alpha 0.5 100 1 2
    done

# 586
# tim1-5b5474c799-qtssh
elif [ $(hostname) = "tim1-5b5474c799-qtssh" ]
then
    # linear increase
    for alpha in 1.1 1.2 1.3 1.4
    do
        run_experiment kuhn_poker 1 1 1 $alpha 0.5 100 1 3
    done

# for big experiments
# 582
elif [ $(hostname) = "zhihan1-7dd8d7ff9c-txdpr" ]
then
    # linear increase
    run_experiment kuhn_poker 1 1 1 1.1 0.5 100 1 4

# for big experiments
#31578
elif [ $(hostname) = "tim7-fc8754657-l6nk2" ]
then
    run_experiment kuhn_poker 0 1 1 1 0.5 100 1 4
# 576
elif [ $(hostname) = "zhihan2-cf974b58f-ftn4q" ]
then
    # fixed alpha
    run_experiment kuhn_poker 1 1 1 1.15 0.5 100 1 4
fi

