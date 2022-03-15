CUDA_VISIBLE_DEVICES=0,1,2 python main.py --task mnist_cnum100_dist1_skew0.2_seed0 --model cnn --algorithm fedavg --num_rounds 50 --num_epochs 5 --learning_rate 0.001 --proportion 0.1 --batch_size 10 --eval_interval 1 --num_threads_per_gpu 3  --num_gpus 3
# python main.py --task mnist_cnum100_dist1_skew0.2_seed0 --model cnn --algorithm fedavg --num_rounds 50 --num_epochs 5 --learning_rate 0.001 --proportion 0.1 --batch_size 10 --eval_interval 1 --gpu 0
# python main.py --task mnist_cnum100_dist1_skew0.2_seed0 --model cnn --algorithm fedprox --num_rounds 50 --num_epochs 5 --learning_rate 0.001 --proportion 0.1 --batch_size 10 --eval_interval 1 --gpu 0

# python main.py --task mnist_cnum10_dist1_skew0.8_seed0 --model cnn --algorithm fedavg --num_rounds 100 --num_epochs 5 --learning_rate 0.001 --proportion 1 --batch_size 10 --eval_interval 1 --gpu 3
# python main.py --task mnist_cnum10_dist1_skew0.8_seed0 --model cnn --algorithm fedprox --num_rounds 100 --num_epochs 5 --learning_rate 0.001 --proportion 1 --batch_size 10 --eval_interval 1 --gpu 3

# python main.py --task mnist_cnum100_dist2_skew0.3_seed0 --model cnn --algorithm fedrl --num_rounds 100 --num_epochs 5 --learning_rate 0.001 --proportion 1 --batch_size 10 --eval_interval 1 --gpu 3
# python main.py --task mnist_cnum100_dist2_skew0.3_seed0 --model cnn --algorithm fedavg --num_rounds 100 --num_epochs 5 --learning_rate 0.001 --proportion 1 --batch_size 10 --eval_interval 1 --gpu 3
# python main.py --task mnist_cnum100_dist2_skew0.3_seed0 --model cnn --algorithm fedprox --num_rounds 100 --num_epochs 5 --learning_rate 0.001 --proportion 1 --batch_size 10 --eval_interval 1 --gpu 3

# python main.py --task mnist_cnum100_dist3_skew0.4_seed0 --model cnn --algorithm fedrl --num_rounds 100 --num_epochs 5 --learning_rate 0.001 --proportion 1 --batch_size 10 --eval_interval 1 --gpu 3
# python main.py --task mnist_cnum100_dist2_skew0.4_seed0 --model cnn --algorithm fedavg --num_rounds 100 --num_epochs 5 --learning_rate 0.001 --proportion 1 --batch_size 10 --eval_interval 1 --gpu 3
# python main.py --task mnist_cnum100_dist2_skew0.4_seed0 --model cnn --algorithm fedprox --num_rounds 100 --num_epochs 5 --learning_rate 0.001 --proportion 1 --batch_size 10 --eval_interval 1 --gpu 3
