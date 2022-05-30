mkdir fedtask

# cifar100_clustered_N10_K10 - quantitative
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_clustered_N10_K10 --model cnn --algorithm fedtestULT --data_folder ./benchmark/cifar100/data --log_folder fedtask --dataidx_filename cifar/cluster/CIFAR_10client_quantitative_cluster.json --num_rounds 500 --num_epochs 5 --proportion 1 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_clustered_N10_K10 --model cnn --algorithm scaffold --data_folder ./benchmark/cifar100/data --log_folder fedtask --dataidx_filename cifar/cluster/CIFAR_10client_quantitative_cluster.json --num_rounds 5 --num_epochs 5 --proportion 1 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_clustered_N10_K10 --model cnn --algorithm fedsdiv_transS --data_folder ./benchmark/cifar100/data --log_folder fedtask --dataidx_filename cifar/cluster/CIFAR_10client_quantitative_cluster.json --num_rounds 500 --num_epochs 5 --proportion 1 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_clustered_N10_K10 --model cnn --algorithm mp_fedavg --data_folder ./benchmark/cifar100/data --log_folder fedtask --dataidx_filename cifar/cluster/CIFAR_10client_quantitative_cluster.json --num_rounds 500 --num_epochs 5 --proportion 1 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_clustered_N10_K10 --model cnn --algorithm mp_fedprox --data_folder ./benchmark/cifar100/data --log_folder fedtask --dataidx_filename cifar/cluster/CIFAR_10client_quantitative_cluster.json --num_rounds 500 --num_epochs 5 --proportion 1 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_clustered_N10_K10 --model cnn --algorithm feddyn --data_folder ./benchmark/cifar100/data --log_folder fedtask --dataidx_filename cifar/cluster/CIFAR_10client_quantitative_cluster.json --num_rounds 500 --num_epochs 5 --proportion 1 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_clustered_N10_K10 --model cnn --algorithm fedfa --alpha 0.5 --data_folder ./benchmark/cifar100/data --log_folder fedtask --dataidx_filename cifar/cluster/CIFAR_10client_quantitative_cluster.json --num_rounds 500 --num_epochs 5 --proportion 1 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_clustered_N10_K10 --model cnn --algorithm fedfv --alpha 0.1 --data_folder ./benchmark/cifar100/data --log_folder fedtask --dataidx_filename cifar/cluster/CIFAR_10client_quantitative_cluster.json --num_rounds 500 --num_epochs 5 --proportion 1 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0

# cifar100_clustered_N100_K10 - quantitative
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_clustered_N100_K10 --model cnn --algorithm fedtestULT --data_folder ./benchmark/cifar100/data --log_folder fedtask --dataidx_filename cifar/100client/cluster/CIFAR_100client_quantitative_cluster.json --num_rounds 500 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_clustered_N100_K10 --model cnn --algorithm scaffold --data_folder ./benchmark/cifar100/data --log_folder fedtask --dataidx_filename cifar/100client/cluster/CIFAR_100client_quantitative_cluster.json --num_rounds 500 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_clustered_N100_K10 --model cnn --algorithm fedsdiv_transS --data_folder ./benchmark/cifar100/data --log_folder fedtask --dataidx_filename cifar/100client/cluster/CIFAR_100client_quantitative_cluster.json --num_rounds 500 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_clustered_N100_K10 --model cnn --algorithm mp_fedavg --data_folder ./benchmark/cifar100/data --log_folder fedtask --dataidx_filename cifar/100client/cluster/CIFAR_100client_quantitative_cluster.json --num_rounds 500 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_clustered_N100_K10 --model cnn --algorithm mp_fedprox --data_folder ./benchmark/cifar100/data --log_folder fedtask --dataidx_filename cifar/100client/cluster/CIFAR_100client_quantitative_cluster.json --num_rounds 500 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_clustered_N100_K10 --model cnn --algorithm feddyn --data_folder ./benchmark/cifar100/data --log_folder fedtask --dataidx_filename cifar/100client/cluster/CIFAR_100client_quantitative_cluster.json --num_rounds 500 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_clustered_N100_K10 --model cnn --algorithm fedfa --alpha 0.5 --data_folder ./benchmark/cifar100/data --log_folder fedtask --dataidx_filename cifar/100client/cluster/CIFAR_100client_quantitative_cluster.json --num_rounds 500 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_clustered_N100_K10 --model cnn --algorithm fedfv --alpha 0.1 --data_folder ./benchmark/cifar100/data --log_folder fedtask --dataidx_filename cifar/100client/cluster/CIFAR_100client_quantitative_cluster.json --num_rounds 500 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0

# cifar100_clustered_N100_K20 - quantitative
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_clustered_N100_K20 --model cnn --algorithm fedtestULT --data_folder ./benchmark/cifar100/data --log_folder fedtask --dataidx_filename cifar/100client/cluster/CIFAR_100client_quantitative_cluster.json --num_rounds 500 --num_epochs 5 --proportion 0.2 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_clustered_N100_K20 --model cnn --algorithm scaffold --data_folder ./benchmark/cifar100/data --log_folder fedtask --dataidx_filename cifar/100client/cluster/CIFAR_100client_quantitative_cluster.json --num_rounds 500 --num_epochs 5 --proportion 0.2 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_clustered_N100_K20 --model cnn --algorithm fedsdiv_transS --data_folder ./benchmark/cifar100/data --log_folder fedtask --dataidx_filename cifar/100client/cluster/CIFAR_100client_quantitative_cluster.json --num_rounds 500 --num_epochs 5 --proportion 0.2 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_clustered_N100_K20 --model cnn --algorithm mp_fedavg --data_folder ./benchmark/cifar100/data --log_folder fedtask --dataidx_filename cifar/100client/cluster/CIFAR_100client_quantitative_cluster.json --num_rounds 500 --num_epochs 5 --proportion 0.2 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_clustered_N100_K20 --model cnn --algorithm mp_fedprox --data_folder ./benchmark/cifar100/data --log_folder fedtask --dataidx_filename cifar/100client/cluster/CIFAR_100client_quantitative_cluster.json --num_rounds 500 --num_epochs 5 --proportion 0.2 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_clustered_N100_K20 --model cnn --algorithm feddyn --data_folder ./benchmark/cifar100/data --log_folder fedtask --dataidx_filename cifar/100client/cluster/CIFAR_100client_quantitative_cluster.json --num_rounds 500 --num_epochs 5 --proportion 0.2 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_clustered_N100_K20 --model cnn --algorithm fedfa --alpha 0.5 --data_folder ./benchmark/cifar100/data --log_folder fedtask --dataidx_filename cifar/100client/cluster/CIFAR_100client_quantitative_cluster.json --num_rounds 500 --num_epochs 5 --proportion 0.2 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_clustered_N100_K20 --model cnn --algorithm fedfv --alpha 0.1 --data_folder ./benchmark/cifar100/data --log_folder fedtask --dataidx_filename cifar/100client/cluster/CIFAR_100client_quantitative_cluster.json --num_rounds 500 --num_epochs 5 --proportion 0.2 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0

# cifar100_clustered_N100_K40 - quantitative
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_clustered_N100_K40 --model cnn --algorithm fedtestULT --data_folder ./benchmark/cifar100/data --log_folder fedtask --dataidx_filename cifar/100client/cluster/CIFAR_100client_quantitative_cluster.json --num_rounds 500 --num_epochs 5 --proportion 0.4 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_clustered_N100_K40 --model cnn --algorithm scaffold --data_folder ./benchmark/cifar100/data --log_folder fedtask --dataidx_filename cifar/100client/cluster/CIFAR_100client_quantitative_cluster.json --num_rounds 500 --num_epochs 5 --proportion 0.4 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_clustered_N100_K40 --model cnn --algorithm fedsdiv_transS --data_folder ./benchmark/cifar100/data --log_folder fedtask --dataidx_filename cifar/100client/cluster/CIFAR_100client_quantitative_cluster.json --num_rounds 500 --num_epochs 5 --proportion 0.4 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_clustered_N100_K40 --model cnn --algorithm mp_fedavg --data_folder ./benchmark/cifar100/data --log_folder fedtask --dataidx_filename cifar/100client/cluster/CIFAR_100client_quantitative_cluster.json --num_rounds 500 --num_epochs 5 --proportion 0.4 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_clustered_N100_K40 --model cnn --algorithm mp_fedprox --data_folder ./benchmark/cifar100/data --log_folder fedtask --dataidx_filename cifar/100client/cluster/CIFAR_100client_quantitative_cluster.json --num_rounds 500 --num_epochs 5 --proportion 0.4 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_clustered_N100_K40 --model cnn --algorithm feddyn --data_folder ./benchmark/cifar100/data --log_folder fedtask --dataidx_filename cifar/100client/cluster/CIFAR_100client_quantitative_cluster.json --num_rounds 500 --num_epochs 5 --proportion 0.4 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_clustered_N100_K40 --model cnn --algorithm fedfa --alpha 0.5 --data_folder ./benchmark/cifar100/data --log_folder fedtask --dataidx_filename cifar/100client/cluster/CIFAR_100client_quantitative_cluster.json --num_rounds 500 --num_epochs 5 --proportion 0.4 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_clustered_N100_K40 --model cnn --algorithm fedfv --alpha 0.1 --data_folder ./benchmark/cifar100/data --log_folder fedtask --dataidx_filename cifar/100client/cluster/CIFAR_100client_quantitative_cluster.json --num_rounds 500 --num_epochs 5 --proportion 0.4 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0

# cifar100_clustered_N100_K80 - quantitative
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_clustered_N100_K80 --model cnn --algorithm fedtestULT --data_folder ./benchmark/cifar100/data --log_folder fedtask --dataidx_filename cifar/100client/cluster/CIFAR_100client_quantitative_cluster.json --num_rounds 500 --num_epochs 5 --proportion 0.8 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_clustered_N100_K80 --model cnn --algorithm scaffold --data_folder ./benchmark/cifar100/data --log_folder fedtask --dataidx_filename cifar/100client/cluster/CIFAR_100client_quantitative_cluster.json --num_rounds 500 --num_epochs 5 --proportion 0.8 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_clustered_N100_K80 --model cnn --algorithm fedsdiv_transS --data_folder ./benchmark/cifar100/data --log_folder fedtask --dataidx_filename cifar/100client/cluster/CIFAR_100client_quantitative_cluster.json --num_rounds 500 --num_epochs 5 --proportion 0.8 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_clustered_N100_K80 --model cnn --algorithm mp_fedavg --data_folder ./benchmark/cifar100/data --log_folder fedtask --dataidx_filename cifar/100client/cluster/CIFAR_100client_quantitative_cluster.json --num_rounds 500 --num_epochs 5 --proportion 0.8 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_clustered_N100_K80 --model cnn --algorithm mp_fedprox --data_folder ./benchmark/cifar100/data --log_folder fedtask --dataidx_filename cifar/100client/cluster/CIFAR_100client_quantitative_cluster.json --num_rounds 500 --num_epochs 5 --proportion 0.8 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_clustered_N100_K80 --model cnn --algorithm feddyn --data_folder ./benchmark/cifar100/data --log_folder fedtask --dataidx_filename cifar/100client/cluster/CIFAR_100client_quantitative_cluster.json --num_rounds 500 --num_epochs 5 --proportion 0.8 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_clustered_N100_K80 --model cnn --algorithm fedfa --alpha 0.5 --data_folder ./benchmark/cifar100/data --log_folder fedtask --dataidx_filename cifar/100client/cluster/CIFAR_100client_quantitative_cluster.json --num_rounds 500 --num_epochs 5 --proportion 0.8 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_clustered_N100_K80 --model cnn --algorithm fedfv --alpha 0.1 --data_folder ./benchmark/cifar100/data --log_folder fedtask --dataidx_filename cifar/100client/cluster/CIFAR_100client_quantitative_cluster.json --num_rounds 500 --num_epochs 5 --proportion 0.8 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
