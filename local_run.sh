
python -m dqn_zoo.bootstrapped_dqn.run_atari \
    --jax_platform_name=gpu \
    --environment_name=pong \
    --replay_capacity=1000 \
    --target_network_update_period=40 \
    --num_iterations=10 \
    --num_train_frames=1000 \
    --num_eval_frames=500