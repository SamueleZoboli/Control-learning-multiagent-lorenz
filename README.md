# Learning contracting control for a network of homogeneous Lorenz attractors
Code for reproducing experiments of Giaccagli et al. "Synchronization in networks of
nonlinear systems: Contraction analysis via Riemannian metrics and deep-learning for feedback
estimation", section V.\
The 'find_P.py' file generates and trains the metric network according to the contraction conditions.\
The 'find_alpha.py' file generates and trains the alpha network according to the integrability condition.\
The 'test_multiagent.py' file loads the alpha network and tests the proposed controller on a network of chaotic Lorenz attractors.

**Arguments for launching the code:**\
*find_P*: \
        --net : hidden layers dimensions separated by comma\
        --activ : activation function for hidden layers (relu or tanh), shared between layers\
        --dataset_size : total number of samples\
        --batch_size : batch size for training\
        --n_epochs : number of training epochs\
        --learning_rate : learning rate, scheduling follows cosineannealing for the metric training\
        --log_name : name of log folder, which will be stored in 'runs/' folder\
*find_alpha*: \
        --net : hidden layers dimensions separated by comma\
        --activ : activation function for hidden layers (relu or tanh), shared between layers\
        --dataset_size : total number of samples\
        --batch_size : batch size for training\
        --n_epochs : number of training epochs\
        --learning_rate : learning rate, scheduling follows cosineannealing \
        --log_name : name of log folder, which will be stored in 'runs/' folder\
*test_multiagent*: \
        --alpha_path : path to the alpha network
