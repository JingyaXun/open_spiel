# format: 
# ./experiments.sh [game: kuhn_poker | leduc_poker] [adaptive_alpha: 1 | 0] [increase: 1 | 0] [adaptive_policy: 1 | 2 | 3] [alphas: 1 1.1 1.2 1.3 1.4] [semi_percent: 0.2 | 0.5] [exploit_rate: 100] [exp_exploit_rate: 1]
cd /root/Documents/ICML2020/open_spiel/open_spiel/python/examples
export PYTHONPATH=$PYTHONPATH:/root/Documents/ICML2020/open_spiel
export PYTHONPATH=$PYTHONPATH:/root/Documents/ICML2020/open_spiel/build/python
source ../../../venv/bin/activate
mkdir -p /root/Documents/ICML2020/open_spiel/open_spiel/python/examples/ICML_Experiments_Final/kuhn_poker_players_experiments
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
    NUM_EXPERIMENTS=10
    ITERATIONS=10000
    OUT_DIR=ICML_Experiments_Final
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

# 580
if [ $(hostname) = "tim6-666d9955f7-5hcpb" ] 
then
    # fixed alpha
    for alpha in 1 1.1 1.2 1.3
    do
        run_experiment kuhn_poker 0 1 1 $alpha 0.5 100 1 3
    done
   

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
elif [ $(hostname) = "tim3-77549f558f-7nzdv" ]
then
    # linear increase
    for alpha in 1 1.05 1.1 1.15
    do
        run_experiment kuhn_poker 1 1 1 $alpha 0.5 100 1 4
    done

# for big experiments
#31578
elif [ $(hostname) = "tim7-fc8754657-ndwf2" ]
then
    # lin inc
    for alpha in 1.2 1.25 1.3 1.35
    do
        run_experiment kuhn_poker 1 1 1 $alpha 0.5 100 1 4
    done

#588
else
    # 3 player lin inc 1.4 
    run_experiment kuhn_poker 0 1 1 1.4 0.5 100 1 3
fi

