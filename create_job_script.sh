TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S")

python create_job_script.py --algorithm bootstrapped_dqn --environment gravitar --penalty_type hard_coded_penalty --conda_env dqn_zoo --save_path job_script --num_cpus 32 --memory 64 --num_hours 72 --exp_name $TIMESTAMP
python create_job_script.py --algorithm bootstrapped_dqn --environment gravitar --penalty_type policy_entropy_penalty --conda_env dqn_zoo --save_path job_script --num_cpus 32 --memory 64 --num_hours 72 --exp_name $TIMESTAMP
python create_job_script.py --algorithm bootstrapped_dqn --environment gravitar --penalty_type uncertainty_penalty --conda_env dqn_zoo --save_path job_script --num_cpus 32 --memory 64 --num_hours 72 --exp_name $TIMESTAMP

python create_job_script.py --algorithm bootstrapped_dqn --environment hero --penalty_type hard_coded_penalty --conda_env dqn_zoo --save_path job_script --num_cpus 32 --memory 64 --num_hours 72 --exp_name $TIMESTAMP
python create_job_script.py --algorithm bootstrapped_dqn --environment hero --penalty_type policy_entropy_penalty --conda_env dqn_zoo --save_path job_script --num_cpus 32 --memory 64 --num_hours 72 --exp_name $TIMESTAMP
python create_job_script.py --algorithm bootstrapped_dqn --environment hero --penalty_type uncertainty_penalty --conda_env dqn_zoo --save_path job_script --num_cpus 32 --memory 64 --num_hours 72 --exp_name $TIMESTAMP

python create_job_script.py --algorithm bootstrapped_dqn --environment asterix --penalty_type hard_coded_penalty --conda_env dqn_zoo --save_path job_script --num_cpus 32 --memory 64 --num_hours 72 --exp_name $TIMESTAMP
python create_job_script.py --algorithm bootstrapped_dqn --environment asterix --penalty_type policy_entropy_penalty --conda_env dqn_zoo --save_path job_script --num_cpus 32 --memory 64 --num_hours 72 --exp_name $TIMESTAMP
python create_job_script.py --algorithm bootstrapped_dqn --environment asterix --penalty_type uncertainty_penalty --conda_env dqn_zoo --save_path job_script --num_cpus 32 --memory 64 --num_hours 72 --exp_name $TIMESTAMP

python create_job_script.py --algorithm bootstrapped_dqn --environment defender --penalty_type hard_coded_penalty --conda_env dqn_zoo --save_path job_script --num_cpus 32 --memory 64 --num_hours 72 --exp_name $TIMESTAMP
python create_job_script.py --algorithm bootstrapped_dqn --environment defender --penalty_type policy_entropy_penalty --conda_env dqn_zoo --save_path job_script --num_cpus 32 --memory 64 --num_hours 72 --exp_name $TIMESTAMP
python create_job_script.py --algorithm bootstrapped_dqn --environment defender --penalty_type uncertainty_penalty --conda_env dqn_zoo --save_path job_script --num_cpus 32 --memory 64 --num_hours 72 --exp_name $TIMESTAMP

cp subdir_run.sh results/$TIMESTAMP