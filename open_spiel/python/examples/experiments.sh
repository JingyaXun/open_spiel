# kuhn_poker, incrrease, linear, different alpha
cd /root/Documents/ICML2020/open_spiel/open_spiel/python/examples
export PYTHONPATH=$PYTHONPATH:/root/Documents/ICML2020/open_spiel
export PYTHONPATH=$PYTHONPATH:/root/Documents/ICML2020/open_spiel/build/python
source ../../../venv/bin/activate

python icml_experiments_multiprocessing.py \
--num_experiments=10 \
--iterations=8000 \
--game=kuhn_poker \
--alpha=1 \
--adaptive_alpha=True  \
--increase=1 \
--adaptive_policy=2 \
--out=ICML_Experiments_Dynamics/kuhn_adaptive/kuhn_8000_exp_1.txt &

python icml_experiments_multiprocessing.py \
--num_experiments=10 \
--iterations=8000 \
--game=kuhn_poker \
--alpha=1.1 \
--adaptive_alpha=True  \
--increase=1 \
--adaptive_policy=2 \
--out=ICML_Experiments_Dynamics/kuhn_adaptive/kuhn_8000_exp_inc_1.1.txt &

python icml_experiments_multiprocessing.py \
--num_experiments=10 \
--iterations=8000 \
--game=kuhn_poker \
--alpha=1.2 \
--adaptive_alpha=True  \
--increase=1 \
--adaptive_policy=2 \
--out=ICML_Experiments_Dynamics/kuhn_adaptive/kuhn_8000_exp_inc_1.2.txt &

python icml_experiments_multiprocessing.py \
--num_experiments=10 \
--iterations=8000 \
--game=kuhn_poker \
--alpha=1.3 \
--adaptive_alpha=True  \
--increase=1 \
--adaptive_policy=2 \
--out=ICML_Experiments_Dynamics/kuhn_adaptive/kuhn_8000_exp_inc_1.3.txt &

python icml_experiments_multiprocessing.py \
--num_experiments=10 \
--iterations=8000 \
--game=kuhn_poker \
--alpha=1.4 \
--adaptive_alpha=True  \
--increase=1 \
--adaptive_policy=2 \
--out=ICML_Experiments_Dynamics/kuhn_adaptive/kuhn_8000_exp_inc_1.4.txt
