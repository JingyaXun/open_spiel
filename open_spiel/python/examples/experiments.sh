# kuhn_poker, incrrease, linear, different alpha
python icml_experiments_multiprocessing.py \
--num_experiments=10 \
--iterations=8000 \
--game=kuhn_poker \
--alpha=1 \
--adaptive_alpha=True  \
--increase=1 \
--adaptive_policy=1 \
--out=ICML_Experiments_Dynamics/leduc_adaptive/kuhn_8000_1.txt &

python icml_experiments_multiprocessing.py \
--num_experiments=10 \
--iterations=8000 \
--game=kuhn_poker \
--alpha=1.1 \
--adaptive_alpha=True  \
--increase=1 \
--adaptive_policy=1 \
--out=ICML_Experiments_Dynamics/leduc_adaptive/kuhn_8000_lin_inc_1.1.txt &

python icml_experiments_multiprocessing.py \
--num_experiments=10 \
--iterations=8000 \
--game=kuhn_poker \
--alpha=1.2 \
--adaptive_alpha=True  \
--increase=1 \
--adaptive_policy=1 \
--out=ICML_Experiments_Dynamics/leduc_adaptive/kuhn_8000_lin_inc_1.2.txt &

python icml_experiments_multiprocessing.py \
--num_experiments=10 \
--iterations=8000 \
--game=kuhn_poker \
--alpha=1.3 \
--adaptive_alpha=True  \
--increase=1 \
--adaptive_policy=1 \
--out=ICML_Experiments_Dynamics/leduc_adaptive/kuhn_8000_lin_inc_1.3.txt &

python icml_experiments_multiprocessing.py \
--num_experiments=10 \
--iterations=8000 \
--game=kuhn_poker \
--alpha=1.4 \
--adaptive_alpha=True  \
--increase=1 \
--adaptive_policy=1 \
--out=ICML_Experiments_Dynamics/leduc_adaptive/kuhn_8000_lin_inc_1.4.txt
