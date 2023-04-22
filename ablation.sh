# td prioritisation on pong
python gpu_job_script.py --dqn_algorithm ens_qrdqn --environment_name pong --priority td --exp_name td_pong_1 --seed 1
python gpu_job_script.py --dqn_algorithm ens_qrdqn --environment_name pong --priority td --exp_name td_pong_2 --seed 2
python gpu_job_script.py --dqn_algorithm ens_qrdqn --environment_name pong --priority td --exp_name td_pong_3 --seed 3
python gpu_job_script.py --dqn_algorithm ens_qrdqn --environment_name pong --priority td --exp_name td_pong_4 --seed 4
python gpu_job_script.py --dqn_algorithm ens_qrdqn --environment_name pong --priority td --exp_name td_pong_5 --seed 5
python gpu_job_script.py --dqn_algorithm ens_qrdqn --environment_name pong --priority td --exp_name td_pong_6 --seed 6
python gpu_job_script.py --dqn_algorithm ens_qrdqn --environment_name pong --priority td --exp_name td_pong_7 --seed 7
python gpu_job_script.py --dqn_algorithm ens_qrdqn --environment_name pong --priority td --exp_name td_pong_8 --seed 8

# no prioritisation on breakout
python gpu_job_script.py --dqn_algorithm ens_qrdqn --environment_name breakout --exp_name breakout_1 --seed 1
python gpu_job_script.py --dqn_algorithm ens_qrdqn --environment_name breakout --exp_name breakout_2 --seed 2
python gpu_job_script.py --dqn_algorithm ens_qrdqn --environment_name breakout --exp_name breakout_3 --seed 3

# standard td prioritisation on breakout
python gpu_job_script.py --dqn_algorithm ens_qrdqn --environment_name breakout --priority td --exp_name td_breakout_1 --seed 1
python gpu_job_script.py --dqn_algorithm ens_qrdqn --environment_name breakout --priority td --exp_name td_breakout_2 --seed 2
python gpu_job_script.py --dqn_algorithm ens_qrdqn --environment_name breakout --priority td --exp_name td_breakout_3 --seed 3

# uncertainty prioritisation on breakout
python gpu_job_script.py --dqn_algorithm ens_qrdqn --environment_name breakout --priority uncertainty --exp_name uncertainty_breakout_1 --seed 1
python gpu_job_script.py --dqn_algorithm ens_qrdqn --environment_name breakout --priority uncertainty --exp_name uncertainty_breakout_2 --seed 2
python gpu_job_script.py --dqn_algorithm ens_qrdqn --environment_name breakout --priority uncertainty --exp_name uncertainty_breakout_3 --seed 3

# no prioritisation on space_invaders
python gpu_job_script.py --dqn_algorithm ens_qrdqn --environment_name space_invaders --exp_name space_invaders_1 --seed 1
python gpu_job_script.py --dqn_algorithm ens_qrdqn --environment_name space_invaders --exp_name space_invaders_2 --seed 2
python gpu_job_script.py --dqn_algorithm ens_qrdqn --environment_name space_invaders --exp_name space_invaders_3 --seed 3

# standard td prioritisation on space_invaders
python gpu_job_script.py --dqn_algorithm ens_qrdqn --environment_name space_invaders --priority td --exp_name td_space_invaders_1 --seed 1
python gpu_job_script.py --dqn_algorithm ens_qrdqn --environment_name space_invaders --priority td --exp_name td_space_invaders_2 --seed 2
python gpu_job_script.py --dqn_algorithm ens_qrdqn --environment_name space_invaders --priority td --exp_name td_space_invaders_3 --seed 3

# uncertainty prioritisation on space_invaders
python gpu_job_script.py --dqn_algorithm ens_qrdqn --environment_name space_invaders --priority uncertainty --exp_name uncertainty_space_invaders_1 --seed 1
python gpu_job_script.py --dqn_algorithm ens_qrdqn --environment_name space_invaders --priority uncertainty --exp_name uncertainty_space_invaders_2 --seed 2
python gpu_job_script.py --dqn_algorithm ens_qrdqn --environment_name space_invaders --priority uncertainty --exp_name uncertainty_space_invaders_3 --seed 3

