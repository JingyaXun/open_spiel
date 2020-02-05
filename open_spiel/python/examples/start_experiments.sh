# format: 
# ./experiments.sh [game: kuhn_poker | leduc_poker] [adaptive_alpha: 1 | 0] [increase: 1 | 0] [adaptive_policy: 1 | 2] [alphas: 1 1.1 1.2 1.3 1.4]
cd /root/Documents/ICML2020/open_spiel/open_spiel/python/examples
export PYTHONPATH=$PYTHONPATH:/root/Documents/ICML2020/open_spiel
export PYTHONPATH=$PYTHONPATH:/root/Documents/ICML2020/open_spiel/build/python
source ../../../venv/bin/activate
mkdir -p /root/Documents/ICML2020/open_spiel/open_spiel/python/examples/ICML_Experiments_Dynamics_test/kuhn_poker_experiments
mkdir -p /root/Documents/ICML2020/open_spiel/open_spiel/python/examples/ICML_Experiments_Dynamics_test/leduc_poker_experiments

chmod +x experiments.sh
if [ $(hostname) = "tim1-5b5474c799-qtssh" ]
then
    #31586
    # experiments for adaptive alpha
    for alpha in 1.1 1.2 1.3 1.4
    do
        ./experiments.sh leduc_poker 1 1 1 $alpha
    done


elif [ $(hostname) = "tim2-667758c9c5-876lw" ]
then
    #31584
    # experiments for fixed alpha
    for alpha in 1 1.1 1.2 1.3
    do
        ./experiments.sh leduc_poker 0 1 1 $alpha
    done 
elif [ $(hostname) = "tim3-77549f558f-7nzdv" ]
then
    # 31582
    # experiments for adaptive alpha
    for alpha in 1.1 1.2 1.3 1.4
    do
        ./experiments.sh leduc_poker 1 0 2  $alpha
    done
else
    #31588
    # experiments for fixed alpha
    for alpha in 1.4 1.5
    do
        ./experiments.sh leduc_poker 0 1 1 $alpha
    done
    ./experiments.sh leduc_poker 1 1 1 1.5
    ./experiments.sh leduc_poker 1 1 2 1.5
fi

