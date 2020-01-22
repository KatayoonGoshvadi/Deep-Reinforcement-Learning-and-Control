1. DQN:
For training
    -> run "python DQN_Implementation.py" with the following argument:
    --env "CartPole-v0" or "MountainCar-v0"
    --render 0
    --train 1
    --num-episodes (total number of episods for training, eg. 4000)
    --test-after (number of episodes after which the model will be evaluated)
    --eval-episode (number of episodes for which the model will be evaluated)
    -- lr (learning rate)
    --gamma (discount factor)
    --replay-size (memory size for the replay buffer)
    --burn-in (burn in size)
    --model (for training leave set it to None)
    --vid-ep (episode for which video is being made, not required for training)

The training code will create a directory storing the results and saved model with the following naming scheme:
cwd = current working directory
model_dir = "cwd/dqn_{}_lr_{}_eps_{}_replay_sz_{}_burn_{}".format(args.env,args.lr,args.num_episodes,args.replay_size, args.burn_in)

For making video
Give the same arguments used for training, except
--train 0
--model (the model to be tested, select the model weights file name from the 
            model_dir/saved_weights)
--vid-ep (set the episode number for the video)

eg for training:
python DQN_Implementation.py --env CartPole-v0 --render 0 --train 1 --model None --num-episodes 5000 --test-after 100 --eval-episodes 20 --lr 1e-3 --gamma 0.99 --replay-size 70000 --burn-in 20000

eg for making video:
python DQN_Implementation.py --env CartPole-v0 --render 0 --train 1 --model episode1600_overall_step_27147.h5 --num-episodes 5000 --test-after 100 --eval-episodes 20 --lr 1e-3 --gamma 0.99 --replay-size 70000 --burn-in 20000


2. Duelling DQN:
For training
    -> run "python Duelling_DQN_Implementation.py" with the following argument:
    --env "CartPole-v0" or "MountainCar-v0"
    --render 0
    --train 1
    --num-episodes (total number of episods for training, eg. 4000)
    --test-after (number of episodes after which the model will be evaluated)
    --eval-episode (number of episodes for which the model will be evaluated)
    -- lr (learning rate)
    --gamma (discount factor)
    --replay-size (memory size for the replay buffer)
    --burn-in (burn in size)
    --model (for training leave set it to None)
    --vid-ep (episode for which video is being made, not required for training)

The training code will create a directory storing the results and saved model with the following naming scheme:
cwd = current working directory
model_dir = "cwd/duel_dqn_{}_lr_{}_eps_{}_replay_sz_{}_burn_{}".format(args.env,args.lr,args.num_episodes,args.replay_size, args.burn_in)

For making video
Give the same arguments used for training, except
--train 0
--model (the model to be tested, select the model weights file name from the 
            model_dir/saved_weights)
--vid-ep (set the episode number for the video)

eg for training:
python Duelling_DQN_Implementation.py --env CartPole-v0 --render 0 --train 1 --model None --num-episodes 5000 --test-after 100 --eval-episodes 20 --lr 1e-3 --gamma 0.99 --replay-size 70000 --burn-in 20000

eg for making video:
python Duelling_DQN_Implementation.py --env CartPole-v0 --render 0 --train 1 --model episode1600_overall_step_27147.h5 --num-episodes 5000 --test-after 100 --eval-episodes 20 --lr 1e-3 --gamma 0.99 --replay-size 70000 --burn-in 20000