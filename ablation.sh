# td prioritisation on pong
python gpu_job_script.py --dqn_algorithm ens_qrdqn --environment_name pong --priority td --seed 1
python gpu_job_script.py --dqn_algorithm ens_qrdqn --environment_name pong --priority td --seed 2
python gpu_job_script.py --dqn_algorithm ens_qrdqn --environment_name pong --priority td --seed 3
python gpu_job_script.py --dqn_algorithm ens_qrdqn --environment_name pong --priority td --seed 4
python gpu_job_script.py --dqn_algorithm ens_qrdqn --environment_name pong --priority td --seed 5
python gpu_job_script.py --dqn_algorithm ens_qrdqn --environment_name pong --priority td --seed 6
python gpu_job_script.py --dqn_algorithm ens_qrdqn --environment_name pong --priority td --seed 7
python gpu_job_script.py --dqn_algorithm ens_qrdqn --environment_name pong --priority td --seed 8

# no prioritisation on breakout
python gpu_job_script.py --dqn_algorithm ens_qrdqn --environment_name breakout --seed 1
python gpu_job_script.py --dqn_algorithm ens_qrdqn --environment_name breakout --seed 2
python gpu_job_script.py --dqn_algorithm ens_qrdqn --environment_name breakout --seed 3

# standard td prioritisation on breakout
python gpu_job_script.py --dqn_algorithm ens_qrdqn --environment_name breakout --priority td --seed 1
python gpu_job_script.py --dqn_algorithm ens_qrdqn --environment_name breakout --priority td --seed 2
python gpu_job_script.py --dqn_algorithm ens_qrdqn --environment_name breakout --priority td --seed 3

# uncertainty prioritisation on breakout
python gpu_job_script.py --dqn_algorithm ens_qrdqn --environment_name breakout --priority uncertainty --seed 1
python gpu_job_script.py --dqn_algorithm ens_qrdqn --environment_name breakout --priority uncertainty --seed 2
python gpu_job_script.py --dqn_algorithm ens_qrdqn --environment_name breakout --priority uncertainty --seed 3

# no prioritisation on space_invaders
python gpu_job_script.py --dqn_algorithm ens_qrdqn --environment_name space_invaders --seed 1
python gpu_job_script.py --dqn_algorithm ens_qrdqn --environment_name space_invaders --seed 2
python gpu_job_script.py --dqn_algorithm ens_qrdqn --environment_name space_invaders --seed 3

# standard td prioritisation on space_invaders
python gpu_job_script.py --dqn_algorithm ens_qrdqn --environment_name space_invaders --priority td --seed 1
python gpu_job_script.py --dqn_algorithm ens_qrdqn --environment_name space_invaders --priority td --seed 2
python gpu_job_script.py --dqn_algorithm ens_qrdqn --environment_name space_invaders --priority td --seed 3

# uncertainty prioritisation on space_invaders
python gpu_job_script.py --dqn_algorithm ens_qrdqn --environment_name space_invaders --priority uncertainty --seed 1
python gpu_job_script.py --dqn_algorithm ens_qrdqn --environment_name space_invaders --priority uncertainty --seed 2
python gpu_job_script.py --dqn_algorithm ens_qrdqn --environment_name space_invaders --priority uncertainty --seed 3

