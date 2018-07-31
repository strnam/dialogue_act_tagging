##### Run pretrain

export GOTIT_DIR="path to gotit data"

export INPUT_DIR="path to create input dir for next step"

export PRETRAIN_DIR = 'path for pretrain dir '

python pretrain/gen_vocab.py \\
    --output_dir=$GOTIT_DIR \\
    --dataset=gotit \\
    --imdb_input_dir=$INPUT_DIR \\
    --lowercase=False
 
python pretrain/gen_vocab.py \\
    --output_dir=$GOTIT_DIR \\
    --dataset=gotit \\
    --gotit_input_dir=$INPUT_DIR \\
    --lowercase=False
    
python pretrain.py \\
    --train_dir=$PRETRAIN_DIR \\
    --data_dir=$GOTIT_DIR \\
    --vocab_size=86934 \          \
    --embedding_dims=100 \\
    --rnn_cell_size=512 \\
    --num_candidate_samples=512 \\
    --batch_size=128 \\
    --learning_rate=0.001 \\
    --learning_rate_decay_factor=0.9999 \\
    --max_steps=100 \\
    --max_grad_norm=1.0 \\
    --num_timesteps=400 \\
    --keep_prob_emb=0.5 \\
    --normalize_embeddings
    
####

python training.py
python prediction.py