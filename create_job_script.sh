TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S")

python create_job_script.py --algorithm bootstrapped_dqn --environment gravitar --penalty_type no_penalty --conda_env dqn_zoo --save_path job_script --num_cpus 32 --memory 64 --num_hours 72 --exp_name $TIMESTAMP --seed 1  --penalty_strength -0.05
python create_job_script.py --algorithm bootstrapped_dqn --environment gravitar --penalty_type hard_coded_penalty --conda_env dqn_zoo --save_path job_script --num_cpus 32 --memory 64 --num_hours 72 --exp_name $TIMESTAMP --seed 1  --penalty_strength -0.05
python create_job_script.py --algorithm bootstrapped_dqn --environment gravitar --penalty_type policy_entropy_penalty --conda_env dqn_zoo --save_path job_script --num_cpus 32 --memory 64 --num_hours 72 --exp_name $TIMESTAMP --seed 1  --penalty_strength -0.05
python create_job_script.py --algorithm bootstrapped_dqn --environment gravitar --penalty_type uncertainty_penalty --conda_env dqn_zoo --save_path job_script --num_cpus 32 --memory 64 --num_hours 72 --exp_name $TIMESTAMP --seed 1  --penalty_strength -0.05

python create_job_script.py --algorithm bootstrapped_dqn --environment hero --penalty_type no_penalty --conda_env dqn_zoo --save_path job_script --num_cpus 32 --memory 64 --num_hours 72 --exp_name $TIMESTAMP --seed 1  --penalty_strength -0.05
python create_job_script.py --algorithm bootstrapped_dqn --environment hero --penalty_type hard_coded_penalty --conda_env dqn_zoo --save_path job_script --num_cpus 32 --memory 64 --num_hours 72 --exp_name $TIMESTAMP --seed 1  --penalty_strength -0.05
python create_job_script.py --algorithm bootstrapped_dqn --environment hero --penalty_type policy_entropy_penalty --conda_env dqn_zoo --save_path job_script --num_cpus 32 --memory 64 --num_hours 72 --exp_name $TIMESTAMP --seed 1  --penalty_strength -0.05
python create_job_script.py --algorithm bootstrapped_dqn --environment hero --penalty_type uncertainty_penalty --conda_env dqn_zoo --save_path job_script --num_cpus 32 --memory 64 --num_hours 72 --exp_name $TIMESTAMP --seed 1  --penalty_strength -0.05

python create_job_script.py --algorithm bootstrapped_dqn --environment asterix --penalty_type no_penalty --conda_env dqn_zoo --save_path job_script --num_cpus 32 --memory 64 --num_hours 72 --exp_name $TIMESTAMP --seed 1  --penalty_strength -0.05
python create_job_script.py --algorithm bootstrapped_dqn --environment asterix --penalty_type hard_coded_penalty --conda_env dqn_zoo --save_path job_script --num_cpus 32 --memory 64 --num_hours 72 --exp_name $TIMESTAMP --seed 1  --penalty_strength -0.05
python create_job_script.py --algorithm bootstrapped_dqn --environment asterix --penalty_type policy_entropy_penalty --conda_env dqn_zoo --save_path job_script --num_cpus 32 --memory 64 --num_hours 72 --exp_name $TIMESTAMP --seed 1  --penalty_strength -0.05
python create_job_script.py --algorithm bootstrapped_dqn --environment asterix --penalty_type uncertainty_penalty --conda_env dqn_zoo --save_path job_script --num_cpus 32 --memory 64 --num_hours 72 --exp_name $TIMESTAMP --seed 1  --penalty_strength -0.05

python create_job_script.py --algorithm bootstrapped_dqn --environment gravitar --penalty_type no_penalty --conda_env dqn_zoo --save_path job_script --num_cpus 32 --memory 64 --num_hours 72 --exp_name $TIMESTAMP --seed 2 --penalty_strength -0.05
python create_job_script.py --algorithm bootstrapped_dqn --environment gravitar --penalty_type hard_coded_penalty --conda_env dqn_zoo --save_path job_script --num_cpus 32 --memory 64 --num_hours 72 --exp_name $TIMESTAMP --seed 2 --penalty_strength -0.05
python create_job_script.py --algorithm bootstrapped_dqn --environment gravitar --penalty_type policy_entropy_penalty --conda_env dqn_zoo --save_path job_script --num_cpus 32 --memory 64 --num_hours 72 --exp_name $TIMESTAMP --seed 2 --penalty_strength -0.05
python create_job_script.py --algorithm bootstrapped_dqn --environment gravitar --penalty_type uncertainty_penalty --conda_env dqn_zoo --save_path job_script --num_cpus 32 --memory 64 --num_hours 72 --exp_name $TIMESTAMP --seed 2 --penalty_strength -0.05

python create_job_script.py --algorithm bootstrapped_dqn --environment hero --penalty_type no_penalty --conda_env dqn_zoo --save_path job_script --num_cpus 32 --memory 64 --num_hours 72 --exp_name $TIMESTAMP --seed 2 --penalty_strength -0.05
python create_job_script.py --algorithm bootstrapped_dqn --environment hero --penalty_type hard_coded_penalty --conda_env dqn_zoo --save_path job_script --num_cpus 32 --memory 64 --num_hours 72 --exp_name $TIMESTAMP --seed 2 --penalty_strength -0.05
python create_job_script.py --algorithm bootstrapped_dqn --environment hero --penalty_type policy_entropy_penalty --conda_env dqn_zoo --save_path job_script --num_cpus 32 --memory 64 --num_hours 72 --exp_name $TIMESTAMP --seed 2 --penalty_strength -0.05
python create_job_script.py --algorithm bootstrapped_dqn --environment hero --penalty_type uncertainty_penalty --conda_env dqn_zoo --save_path job_script --num_cpus 32 --memory 64 --num_hours 72 --exp_name $TIMESTAMP --seed 2 --penalty_strength -0.05

python create_job_script.py --algorithm bootstrapped_dqn --environment asterix --penalty_type no_penalty --conda_env dqn_zoo --save_path job_script --num_cpus 32 --memory 64 --num_hours 72 --exp_name $TIMESTAMP --seed 2 --penalty_strength -0.05
python create_job_script.py --algorithm bootstrapped_dqn --environment asterix --penalty_type hard_coded_penalty --conda_env dqn_zoo --save_path job_script --num_cpus 32 --memory 64 --num_hours 72 --exp_name $TIMESTAMP --seed 2 --penalty_strength -0.05
python create_job_script.py --algorithm bootstrapped_dqn --environment asterix --penalty_type policy_entropy_penalty --conda_env dqn_zoo --save_path job_script --num_cpus 32 --memory 64 --num_hours 72 --exp_name $TIMESTAMP --seed 2 --penalty_strength -0.05
python create_job_script.py --algorithm bootstrapped_dqn --environment asterix --penalty_type uncertainty_penalty --conda_env dqn_zoo --save_path job_script --num_cpus 32 --memory 64 --num_hours 72 --exp_name $TIMESTAMP --seed 2 --penalty_strength -0.05

python create_job_script.py --algorithm bootstrapped_dqn --environment gravitar --penalty_type no_penalty --conda_env dqn_zoo --save_path job_script --num_cpus 32 --memory 64 --num_hours 72 --exp_name $TIMESTAMP --seed 1  --penalty_strength -0.01
python create_job_script.py --algorithm bootstrapped_dqn --environment gravitar --penalty_type hard_coded_penalty --conda_env dqn_zoo --save_path job_script --num_cpus 32 --memory 64 --num_hours 72 --exp_name $TIMESTAMP --seed 1  --penalty_strength -0.01
python create_job_script.py --algorithm bootstrapped_dqn --environment gravitar --penalty_type policy_entropy_penalty --conda_env dqn_zoo --save_path job_script --num_cpus 32 --memory 64 --num_hours 72 --exp_name $TIMESTAMP --seed 1  --penalty_strength -0.01
python create_job_script.py --algorithm bootstrapped_dqn --environment gravitar --penalty_type uncertainty_penalty --conda_env dqn_zoo --save_path job_script --num_cpus 32 --memory 64 --num_hours 72 --exp_name $TIMESTAMP --seed 1  --penalty_strength -0.01

python create_job_script.py --algorithm bootstrapped_dqn --environment hero --penalty_type no_penalty --conda_env dqn_zoo --save_path job_script --num_cpus 32 --memory 64 --num_hours 72 --exp_name $TIMESTAMP --seed 1  --penalty_strength -0.01
python create_job_script.py --algorithm bootstrapped_dqn --environment hero --penalty_type hard_coded_penalty --conda_env dqn_zoo --save_path job_script --num_cpus 32 --memory 64 --num_hours 72 --exp_name $TIMESTAMP --seed 1  --penalty_strength -0.01
python create_job_script.py --algorithm bootstrapped_dqn --environment hero --penalty_type policy_entropy_penalty --conda_env dqn_zoo --save_path job_script --num_cpus 32 --memory 64 --num_hours 72 --exp_name $TIMESTAMP --seed 1  --penalty_strength -0.01
python create_job_script.py --algorithm bootstrapped_dqn --environment hero --penalty_type uncertainty_penalty --conda_env dqn_zoo --save_path job_script --num_cpus 32 --memory 64 --num_hours 72 --exp_name $TIMESTAMP --seed 1  --penalty_strength -0.01

python create_job_script.py --algorithm bootstrapped_dqn --environment asterix --penalty_type no_penalty --conda_env dqn_zoo --save_path job_script --num_cpus 32 --memory 64 --num_hours 72 --exp_name $TIMESTAMP --seed 1  --penalty_strength -0.01
python create_job_script.py --algorithm bootstrapped_dqn --environment asterix --penalty_type hard_coded_penalty --conda_env dqn_zoo --save_path job_script --num_cpus 32 --memory 64 --num_hours 72 --exp_name $TIMESTAMP --seed 1  --penalty_strength -0.01
python create_job_script.py --algorithm bootstrapped_dqn --environment asterix --penalty_type policy_entropy_penalty --conda_env dqn_zoo --save_path job_script --num_cpus 32 --memory 64 --num_hours 72 --exp_name $TIMESTAMP --seed 1  --penalty_strength -0.01
python create_job_script.py --algorithm bootstrapped_dqn --environment asterix --penalty_type uncertainty_penalty --conda_env dqn_zoo --save_path job_script --num_cpus 32 --memory 64 --num_hours 72 --exp_name $TIMESTAMP --seed 1  --penalty_strength -0.01

python create_job_script.py --algorithm bootstrapped_dqn --environment gravitar --penalty_type no_penalty --conda_env dqn_zoo --save_path job_script --num_cpus 32 --memory 64 --num_hours 72 --exp_name $TIMESTAMP --seed 2 --penalty_strength -0.01
python create_job_script.py --algorithm bootstrapped_dqn --environment gravitar --penalty_type hard_coded_penalty --conda_env dqn_zoo --save_path job_script --num_cpus 32 --memory 64 --num_hours 72 --exp_name $TIMESTAMP --seed 2 --penalty_strength -0.01
python create_job_script.py --algorithm bootstrapped_dqn --environment gravitar --penalty_type policy_entropy_penalty --conda_env dqn_zoo --save_path job_script --num_cpus 32 --memory 64 --num_hours 72 --exp_name $TIMESTAMP --seed 2 --penalty_strength -0.01
python create_job_script.py --algorithm bootstrapped_dqn --environment gravitar --penalty_type uncertainty_penalty --conda_env dqn_zoo --save_path job_script --num_cpus 32 --memory 64 --num_hours 72 --exp_name $TIMESTAMP --seed 2 --penalty_strength -0.01

python create_job_script.py --algorithm bootstrapped_dqn --environment hero --penalty_type no_penalty --conda_env dqn_zoo --save_path job_script --num_cpus 32 --memory 64 --num_hours 72 --exp_name $TIMESTAMP --seed 2 --penalty_strength -0.01
python create_job_script.py --algorithm bootstrapped_dqn --environment hero --penalty_type hard_coded_penalty --conda_env dqn_zoo --save_path job_script --num_cpus 32 --memory 64 --num_hours 72 --exp_name $TIMESTAMP --seed 2 --penalty_strength -0.01
python create_job_script.py --algorithm bootstrapped_dqn --environment hero --penalty_type policy_entropy_penalty --conda_env dqn_zoo --save_path job_script --num_cpus 32 --memory 64 --num_hours 72 --exp_name $TIMESTAMP --seed 2 --penalty_strength -0.01
python create_job_script.py --algorithm bootstrapped_dqn --environment hero --penalty_type uncertainty_penalty --conda_env dqn_zoo --save_path job_script --num_cpus 32 --memory 64 --num_hours 72 --exp_name $TIMESTAMP --seed 2 --penalty_strength -0.01

python create_job_script.py --algorithm bootstrapped_dqn --environment asterix --penalty_type no_penalty --conda_env dqn_zoo --save_path job_script --num_cpus 32 --memory 64 --num_hours 72 --exp_name $TIMESTAMP --seed 2 --penalty_strength -0.01
python create_job_script.py --algorithm bootstrapped_dqn --environment asterix --penalty_type hard_coded_penalty --conda_env dqn_zoo --save_path job_script --num_cpus 32 --memory 64 --num_hours 72 --exp_name $TIMESTAMP --seed 2 --penalty_strength -0.01
python create_job_script.py --algorithm bootstrapped_dqn --environment asterix --penalty_type policy_entropy_penalty --conda_env dqn_zoo --save_path job_script --num_cpus 32 --memory 64 --num_hours 72 --exp_name $TIMESTAMP --seed 2 --penalty_strength -0.01
python create_job_script.py --algorithm bootstrapped_dqn --environment asterix --penalty_type uncertainty_penalty --conda_env dqn_zoo --save_path job_script --num_cpus 32 --memory 64 --num_hours 72 --exp_name $TIMESTAMP --seed 2 --penalty_strength -0.01

cp subdir_run.sh results/$TIMESTAMP