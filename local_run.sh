
#python -m dqn_zoo.prioritize_uncertainty.run_atari \
#    --jax_platform_name=gpu \
#    --environment_name=pong \
#    --replay_capacity=1000 \
#    --target_network_update_period=40 \
#    --num_iterations=10 \
#    --num_train_frames=1000 \
#    --num_eval_frames=500
echo "-------------------"
echo "CPU ENSEMBLE QR-DQN"
echo "-------------------"
python -m dqn_zoo.ens_qrdqn.run_atari \
    --jax_platform_name=cpu \
    --environment_name=pong \
    --replay_capacity=10000 \
    --target_network_update_period=40 \
    --num_iterations=40 \
    --num_train_frames=1000 \
    --num_eval_frames=5000 \
    --batch_size=64 
    # --results_path="results/ens_qrdqn_cpu"

echo "-------------------"
echo "CPU QR-DQN"
echo "-------------------"
python -m dqn_zoo.qrdqn.run_atari \
    --jax_platform_name=cpu \
    --environment_name=pong \
    --replay_capacity=10000 \
    --target_network_update_period=40 \
    --num_iterations=40 \
    --num_train_frames=1000 \
    --num_eval_frames=5000 \
    --batch_size=64 
    #--results_path="results/qrdqn_cpu"

echo "-------------------"
echo "CPU DQN"
echo "-------------------"
python -m dqn_zoo.dqn.run_atari \
    --jax_platform_name=cpu \
    --environment_name=pong \
    --replay_capacity=10000 \
    --target_network_update_period=40 \
    --num_iterations=40 \
    --num_train_frames=1000 \
    --num_eval_frames=5000 \
    --batch_size=64 
    #--results_path="results/dqn_cpu"

echo "-------------------"
echo "GPU ENSEMBLE QR-DQN"
echo "-------------------"
python -m dqn_zoo.ens_qrdqn.run_atari \
    --jax_platform_name=gpu \
    --environment_name=pong \
    --replay_capacity=10000 \
    --target_network_update_period=40 \
    --num_iterations=40 \
    --num_train_frames=1000 \
    --num_eval_frames=5000 \
    --batch_size=64 
    #--results_path="results/ens_qrdqn_gpu"

echo "-------------------"
echo "GPU QR-DQN"
echo "-------------------"
python -m dqn_zoo.qrdqn.run_atari \
    --jax_platform_name=gpu \
    --environment_name=pong \
    --replay_capacity=10000 \
    --target_network_update_period=40 \
    --num_iterations=40 \
    --num_train_frames=1000 \
    --num_eval_frames=5000 \
    --batch_size=64 
    #--results_path="results/qrdqn_gpu"

echo "-------------------"
echo "GPU DQN"
echo "-------------------"
python -m dqn_zoo.dqn.run_atari \
    --jax_platform_name=gpu \
    --environment_name=pong \
    --replay_capacity=10000 \
    --target_network_update_period=40 \
    --num_iterations=40 \
    --num_train_frames=1000 \
    --num_eval_frames=5000 \
    --batch_size=64 
    #--results_path="results/dqn_gpu"
