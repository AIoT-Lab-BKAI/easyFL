mkdir fedtask

# mnist_clustered_N10_K10 - quantitative
CUDA_VISIBLE_DEVICES=0,1 python main.py --task mnist_clustered_N10_K10 --model cnn --algorithm mp_fedtest --dataidx_filename mnist/client_cluster/MNIST-client-cluster-quantitative.json --num_rounds 500 --num_epochs 5 --learning_rate 0.001 --proportion 1 --batch_size 8 --eval_interval 1 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task mnist_clustered_N10_K10 --model cnn --algorithm mp_fedtestULT --dataidx_filename mnist/client_cluster/MNIST-client-cluster-quantitative.json --num_rounds 500 --num_epochs 5 --learning_rate 0.001 --proportion 1 --batch_size 8 --eval_interval 1 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task mnist_clustered_N10_K10 --model cnn --algorithm scaffold --dataidx_filename mnist/client_cluster/MNIST-client-cluster-quantitative.json --num_rounds 500 --num_epochs 5 --learning_rate 0.001 --proportion 1 --batch_size 8 --eval_interval 1 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task mnist_clustered_N10_K10 --model cnn --algorithm mp_fedkdr --dataidx_filename mnist/client_cluster/MNIST-client-cluster-quantitative.json --num_rounds 500 --num_epochs 5 --learning_rate 0.001 --proportion 1 --batch_size 8 --eval_interval 1 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task mnist_clustered_N10_K10 --model cnn --algorithm mp_fedsdivv5 --dataidx_filename mnist/client_cluster/MNIST-client-cluster-quantitative.json --num_rounds 500 --num_epochs 5 --learning_rate 0.001 --proportion 1 --batch_size 8 --eval_interval 1 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task mnist_clustered_N10_K10 --model cnn --algorithm mo_fedavg --dataidx_filename mnist/client_cluster/MNIST-client-cluster-quantitative.json --num_rounds 500 --num_epochs 5 --learning_rate 0.001 --proportion 1 --batch_size 8 --eval_interval 1 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task mnist_clustered_N10_K10 --model cnn --algorithm mp_fedprox --dataidx_filename mnist/client_cluster/MNIST-client-cluster-quantitative.json --num_rounds 500 --num_epochs 5 --learning_rate 0.001 --proportion 1 --batch_size 8 --eval_interval 1 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0

# mnist_clustered_N100_K10 - quantitative
CUDA_VISIBLE_DEVICES=0,1 python main.py --task mnist_clustered_N100_K10 --model cnn --algorithm mp_fedtest --dataidx_filename mnist/client_cluster/MNIST-100client-cluster-quantitative.json --num_rounds 500 --num_epochs 5 --learning_rate 0.001 --proportion 0.1 --batch_size 8 --eval_interval 1 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task mnist_clustered_N100_K10 --model cnn --algorithm mp_fedtestULT --dataidx_filename mnist/client_cluster/MNIST-100client-cluster-quantitative.json --num_rounds 500 --num_epochs 5 --learning_rate 0.001 --proportion 0.1 --batch_size 8 --eval_interval 1 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task mnist_clustered_N100_K10 --model cnn --algorithm scaffold --dataidx_filename mnist/client_cluster/MNIST-100client-cluster-quantitative.json --num_rounds 500 --num_epochs 5 --learning_rate 0.001 --proportion 0.1 --batch_size 8 --eval_interval 1 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task mnist_clustered_N100_K10 --model cnn --algorithm mp_fedkdr --dataidx_filename mnist/client_cluster/MNIST-100client-cluster-quantitative.json --num_rounds 500 --num_epochs 5 --learning_rate 0.001 --proportion 0.1 --batch_size 8 --eval_interval 1 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task mnist_clustered_N100_K10 --model cnn --algorithm mp_fedsdivv5 --dataidx_filename mnist/client_cluster/MNIST-100client-cluster-quantitative.json --num_rounds 500 --num_epochs 5 --learning_rate 0.001 --proportion 0.1 --batch_size 8 --eval_interval 1 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task mnist_clustered_N100_K10 --model cnn --algorithm mo_fedavg --dataidx_filename mnist/client_cluster/MNIST-100client-cluster-quantitative.json --num_rounds 500 --num_epochs 5 --learning_rate 0.001 --proportion 0.1 --batch_size 8 --eval_interval 1 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task mnist_clustered_N100_K10 --model cnn --algorithm mp_fedprox --dataidx_filename mnist/client_cluster/MNIST-100client-cluster-quantitative.json --num_rounds 500 --num_epochs 5 --learning_rate 0.001 --proportion 0.1 --batch_size 8 --eval_interval 1 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0

# mnist_clustered_N100_K20 - quantitative
CUDA_VISIBLE_DEVICES=0,1 python main.py --task mnist_clustered_N100_K10 --model cnn --algorithm mp_fedtest --dataidx_filename mnist/client_cluster/MNIST-100client-cluster-quantitative.json --num_rounds 500 --num_epochs 5 --learning_rate 0.001 --proportion 0.2 --batch_size 8 --eval_interval 1 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task mnist_clustered_N100_K10 --model cnn --algorithm mp_fedtestULT --dataidx_filename mnist/client_cluster/MNIST-100client-cluster-quantitative.json --num_rounds 500 --num_epochs 5 --learning_rate 0.001 --proportion 0.2 --batch_size 8 --eval_interval 1 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task mnist_clustered_N100_K10 --model cnn --algorithm scaffold --dataidx_filename mnist/client_cluster/MNIST-100client-cluster-quantitative.json --num_rounds 500 --num_epochs 5 --learning_rate 0.001 --proportion 0.2 --batch_size 8 --eval_interval 1 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task mnist_clustered_N100_K10 --model cnn --algorithm mp_fedkdr --dataidx_filename mnist/client_cluster/MNIST-100client-cluster-quantitative.json --num_rounds 500 --num_epochs 5 --learning_rate 0.001 --proportion 0.2 --batch_size 8 --eval_interval 1 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task mnist_clustered_N100_K10 --model cnn --algorithm mp_fedsdivv5 --dataidx_filename mnist/client_cluster/MNIST-100client-cluster-quantitative.json --num_rounds 500 --num_epochs 5 --learning_rate 0.001 --proportion 0.2 --batch_size 8 --eval_interval 1 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task mnist_clustered_N100_K10 --model cnn --algorithm mo_fedavg --dataidx_filename mnist/client_cluster/MNIST-100client-cluster-quantitative.json --num_rounds 500 --num_epochs 5 --learning_rate 0.001 --proportion 0.2 --batch_size 8 --eval_interval 1 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task mnist_clustered_N100_K10 --model cnn --algorithm mp_fedprox --dataidx_filename mnist/client_cluster/MNIST-100client-cluster-quantitative.json --num_rounds 500 --num_epochs 5 --learning_rate 0.001 --proportion 0.2 --batch_size 8 --eval_interval 1 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0

# mnist_clustered_N100_K40 - quantitative
CUDA_VISIBLE_DEVICES=0,1 python main.py --task mnist_clustered_N100_K10 --model cnn --algorithm mp_fedtest --dataidx_filename mnist/client_cluster/MNIST-100client-cluster-quantitative.json --num_rounds 500 --num_epochs 5 --learning_rate 0.001 --proportion 0.4 --batch_size 8 --eval_interval 1 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task mnist_clustered_N100_K10 --model cnn --algorithm mp_fedtestULT --dataidx_filename mnist/client_cluster/MNIST-100client-cluster-quantitative.json --num_rounds 500 --num_epochs 5 --learning_rate 0.001 --proportion 0.4 --batch_size 8 --eval_interval 1 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task mnist_clustered_N100_K10 --model cnn --algorithm scaffold --dataidx_filename mnist/client_cluster/MNIST-100client-cluster-quantitative.json --num_rounds 500 --num_epochs 5 --learning_rate 0.001 --proportion 0.4 --batch_size 8 --eval_interval 1 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task mnist_clustered_N100_K10 --model cnn --algorithm mp_fedkdr --dataidx_filename mnist/client_cluster/MNIST-100client-cluster-quantitative.json --num_rounds 500 --num_epochs 5 --learning_rate 0.001 --proportion 0.4 --batch_size 8 --eval_interval 1 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task mnist_clustered_N100_K10 --model cnn --algorithm mp_fedsdivv5 --dataidx_filename mnist/client_cluster/MNIST-100client-cluster-quantitative.json --num_rounds 500 --num_epochs 5 --learning_rate 0.001 --proportion 0.4 --batch_size 8 --eval_interval 1 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task mnist_clustered_N100_K10 --model cnn --algorithm mo_fedavg --dataidx_filename mnist/client_cluster/MNIST-100client-cluster-quantitative.json --num_rounds 500 --num_epochs 5 --learning_rate 0.001 --proportion 0.4 --batch_size 8 --eval_interval 1 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task mnist_clustered_N100_K10 --model cnn --algorithm mp_fedprox --dataidx_filename mnist/client_cluster/MNIST-100client-cluster-quantitative.json --num_rounds 500 --num_epochs 5 --learning_rate 0.001 --proportion 0.4 --batch_size 8 --eval_interval 1 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0

# mnist_clustered_N100_K80 - quantitative
CUDA_VISIBLE_DEVICES=0,1 python main.py --task mnist_clustered_N100_K10 --model cnn --algorithm mp_fedtest --dataidx_filename mnist/client_cluster/MNIST-100client-cluster-quantitative.json --num_rounds 500 --num_epochs 5 --learning_rate 0.001 --proportion 0.8 --batch_size 8 --eval_interval 1 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task mnist_clustered_N100_K10 --model cnn --algorithm mp_fedtestULT --dataidx_filename mnist/client_cluster/MNIST-100client-cluster-quantitative.json --num_rounds 500 --num_epochs 5 --learning_rate 0.001 --proportion 0.8 --batch_size 8 --eval_interval 1 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task mnist_clustered_N100_K10 --model cnn --algorithm scaffold --dataidx_filename mnist/client_cluster/MNIST-100client-cluster-quantitative.json --num_rounds 500 --num_epochs 5 --learning_rate 0.001 --proportion 0.8 --batch_size 8 --eval_interval 1 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task mnist_clustered_N100_K10 --model cnn --algorithm mp_fedkdr --dataidx_filename mnist/client_cluster/MNIST-100client-cluster-quantitative.json --num_rounds 500 --num_epochs 5 --learning_rate 0.001 --proportion 0.8 --batch_size 8 --eval_interval 1 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task mnist_clustered_N100_K10 --model cnn --algorithm mp_fedsdivv5 --dataidx_filename mnist/client_cluster/MNIST-100client-cluster-quantitative.json --num_rounds 500 --num_epochs 5 --learning_rate 0.001 --proportion 0.8 --batch_size 8 --eval_interval 1 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task mnist_clustered_N100_K10 --model cnn --algorithm mo_fedavg --dataidx_filename mnist/client_cluster/MNIST-100client-cluster-quantitative.json --num_rounds 500 --num_epochs 5 --learning_rate 0.001 --proportion 0.8 --batch_size 8 --eval_interval 1 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task mnist_clustered_N100_K10 --model cnn --algorithm mp_fedprox --dataidx_filename mnist/client_cluster/MNIST-100client-cluster-quantitative.json --num_rounds 500 --num_epochs 5 --learning_rate 0.001 --proportion 0.8 --batch_size 8 --eval_interval 1 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
