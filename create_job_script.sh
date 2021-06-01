TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S")
python create_job_script.py --algorithm bootstrapped_dqn --environment gravitar --penalty_type policy_entropy_penalty --conda_env dqn_zoo --save_path job_script --num_cpus 32 --memory 64 --num_hours 72 --exp_name $TIMESTAMP
