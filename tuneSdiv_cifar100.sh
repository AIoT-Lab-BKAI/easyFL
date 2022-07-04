# For CIFAR100
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_resnet9_clustered_N10_K10 --model resnet9 --kd_fct 2 -sthr 0.9 --algorithm fedsdiv_transS --data_folder "./benchmark/cifar10/data" --log_folder "fedtask" --dataidx_filename "cifar/cluster/CIFAR_10client_quantitative_cluster.json" --num_rounds 500 --num_epochs 5 --proportion 1 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_resnet9_clustered_N10_K10 --model resnet9 --kd_fct 4 -sthr 0.9 --algorithm fedsdiv_transS --data_folder "./benchmark/cifar10/data" --log_folder "fedtask" --dataidx_filename "cifar/cluster/CIFAR_10client_quantitative_cluster.json" --num_rounds 500 --num_epochs 5 --proportion 1 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_resnet9_clustered_N10_K10 --model resnet9 --kd_fct 6 -sthr 0.9 --algorithm fedsdiv_transS --data_folder "./benchmark/cifar10/data" --log_folder "fedtask" --dataidx_filename "cifar/cluster/CIFAR_10client_quantitative_cluster.json" --num_rounds 500 --num_epochs 5 --proportion 1 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_resnet9_clustered_N10_K10 --model resnet9 --kd_fct 8 -sthr 0.9 --algorithm fedsdiv_transS --data_folder "./benchmark/cifar10/data" --log_folder "fedtask" --dataidx_filename "cifar/cluster/CIFAR_10client_quantitative_cluster.json" --num_rounds 500 --num_epochs 5 --proportion 1 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_resnet9_clustered_N10_K10 --model resnet9 --kd_fct 10 -sthr 0.9 --algorithm fedsdiv_transS --data_folder "./benchmark/cifar10/data" --log_folder "fedtask" --dataidx_filename "cifar/cluster/CIFAR_10client_quantitative_cluster.json" --num_rounds 500 --num_epochs 5 --proportion 1 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0

CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_resnet9_clustered_N100_K10 --model resnet9 --kd_fct 2 -sthr 0.9 --algorithm fedsdiv_transS --data_folder "./benchmark/cifar100/data" --log_folder "fedtask" --dataidx_filename "cifar/100client/cluster/CIFAR_100client_quantitative_cluster.json" --num_rounds 500 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_resnet9_clustered_N100_K10 --model resnet9 --kd_fct 4 -sthr 0.9 --algorithm fedsdiv_transS --data_folder "./benchmark/cifar100/data" --log_folder "fedtask" --dataidx_filename "cifar/100client/cluster/CIFAR_100client_quantitative_cluster.json" --num_rounds 500 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_resnet9_clustered_N100_K10 --model resnet9 --kd_fct 6 -sthr 0.9 --algorithm fedsdiv_transS --data_folder "./benchmark/cifar100/data" --log_folder "fedtask" --dataidx_filename "cifar/100client/cluster/CIFAR_100client_quantitative_cluster.json" --num_rounds 500 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_resnet9_clustered_N100_K10 --model resnet9 --kd_fct 8 -sthr 0.9 --algorithm fedsdiv_transS --data_folder "./benchmark/cifar100/data" --log_folder "fedtask" --dataidx_filename "cifar/100client/cluster/CIFAR_100client_quantitative_cluster.json" --num_rounds 500 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_resnet9_clustered_N100_K10 --model resnet9 --kd_fct 10 -sthr 0.9 --algorithm fedsdiv_transS --data_folder "./benchmark/cifar100/data" --log_folder "fedtask" --dataidx_filename "cifar/100client/cluster/CIFAR_100client_quantitative_cluster.json" --num_rounds 500 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0

CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_resnet9_clustered_N100_K20 --model resnet9 --kd_fct 2 -sthr 0.9 --algorithm fedsdiv_transS --data_folder "./benchmark/cifar100/data" --log_folder "fedtask" --dataidx_filename "cifar/100client/cluster/CIFAR_100client_quantitative_cluster.json" --num_rounds 500 --num_epochs 5 --proportion 0.2 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_resnet9_clustered_N100_K20 --model resnet9 --kd_fct 4 -sthr 0.9 --algorithm fedsdiv_transS --data_folder "./benchmark/cifar100/data" --log_folder "fedtask" --dataidx_filename "cifar/100client/cluster/CIFAR_100client_quantitative_cluster.json" --num_rounds 500 --num_epochs 5 --proportion 0.2 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_resnet9_clustered_N100_K20 --model resnet9 --kd_fct 6 -sthr 0.9 --algorithm fedsdiv_transS --data_folder "./benchmark/cifar100/data" --log_folder "fedtask" --dataidx_filename "cifar/100client/cluster/CIFAR_100client_quantitative_cluster.json" --num_rounds 500 --num_epochs 5 --proportion 0.2 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_resnet9_clustered_N100_K20 --model resnet9 --kd_fct 8 -sthr 0.9 --algorithm fedsdiv_transS --data_folder "./benchmark/cifar100/data" --log_folder "fedtask" --dataidx_filename "cifar/100client/cluster/CIFAR_100client_quantitative_cluster.json" --num_rounds 500 --num_epochs 5 --proportion 0.2 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_resnet9_clustered_N100_K20 --model resnet9 --kd_fct 10 -sthr 0.9 --algorithm fedsdiv_transS --data_folder "./benchmark/cifar100/data" --log_folder "fedtask" --dataidx_filename "cifar/100client/cluster/CIFAR_100client_quantitative_cluster.json" --num_rounds 500 --num_epochs 5 --proportion 0.2 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0

CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_resnet9_clustered_N100_K30 --model resnet9 --kd_fct 2 -sthr 0.9 --algorithm fedsdiv_transS --data_folder "./benchmark/cifar100/data" --log_folder "fedtask" --dataidx_filename "cifar/100client/cluster/CIFAR_100client_quantitative_cluster.json" --num_rounds 500 --num_epochs 5 --proportion 0.3 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_resnet9_clustered_N100_K30 --model resnet9 --kd_fct 4 -sthr 0.9 --algorithm fedsdiv_transS --data_folder "./benchmark/cifar100/data" --log_folder "fedtask" --dataidx_filename "cifar/100client/cluster/CIFAR_100client_quantitative_cluster.json" --num_rounds 500 --num_epochs 5 --proportion 0.3 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_resnet9_clustered_N100_K30 --model resnet9 --kd_fct 6 -sthr 0.9 --algorithm fedsdiv_transS --data_folder "./benchmark/cifar100/data" --log_folder "fedtask" --dataidx_filename "cifar/100client/cluster/CIFAR_100client_quantitative_cluster.json" --num_rounds 500 --num_epochs 5 --proportion 0.3 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_resnet9_clustered_N100_K30 --model resnet9 --kd_fct 8 -sthr 0.9 --algorithm fedsdiv_transS --data_folder "./benchmark/cifar100/data" --log_folder "fedtask" --dataidx_filename "cifar/100client/cluster/CIFAR_100client_quantitative_cluster.json" --num_rounds 500 --num_epochs 5 --proportion 0.3 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_resnet9_clustered_N100_K30 --model resnet9 --kd_fct 10 -sthr 0.9 --algorithm fedsdiv_transS --data_folder "./benchmark/cifar100/data" --log_folder "fedtask" --dataidx_filename "cifar/100client/cluster/CIFAR_100client_quantitative_cluster.json" --num_rounds 500 --num_epochs 5 --proportion 0.3 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0

CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_resnet9_clustered_N100_K40 --model resnet9 --kd_fct 2 -sthr 0.9 --algorithm fedsdiv_transS --data_folder "./benchmark/cifar100/data" --log_folder "fedtask" --dataidx_filename "cifar/100client/cluster/CIFAR_100client_quantitative_cluster.json" --num_rounds 500 --num_epochs 5 --proportion 0.4 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_resnet9_clustered_N100_K40 --model resnet9 --kd_fct 4 -sthr 0.9 --algorithm fedsdiv_transS --data_folder "./benchmark/cifar100/data" --log_folder "fedtask" --dataidx_filename "cifar/100client/cluster/CIFAR_100client_quantitative_cluster.json" --num_rounds 500 --num_epochs 5 --proportion 0.4 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_resnet9_clustered_N100_K40 --model resnet9 --kd_fct 6 -sthr 0.9 --algorithm fedsdiv_transS --data_folder "./benchmark/cifar100/data" --log_folder "fedtask" --dataidx_filename "cifar/100client/cluster/CIFAR_100client_quantitative_cluster.json" --num_rounds 500 --num_epochs 5 --proportion 0.4 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_resnet9_clustered_N100_K40 --model resnet9 --kd_fct 8 -sthr 0.9 --algorithm fedsdiv_transS --data_folder "./benchmark/cifar100/data" --log_folder "fedtask" --dataidx_filename "cifar/100client/cluster/CIFAR_100client_quantitative_cluster.json" --num_rounds 500 --num_epochs 5 --proportion 0.4 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_resnet9_clustered_N100_K40 --model resnet9 --kd_fct 10 -sthr 0.9 --algorithm fedsdiv_transS --data_folder "./benchmark/cifar100/data" --log_folder "fedtask" --dataidx_filename "cifar/100client/cluster/CIFAR_100client_quantitative_cluster.json" --num_rounds 500 --num_epochs 5 --proportion 0.4 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0

CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_resnet9_clustered_N100_K50 --model resnet9 --kd_fct 2 -sthr 0.9 --algorithm fedsdiv_transS --data_folder "./benchmark/cifar100/data" --log_folder "fedtask" --dataidx_filename "cifar/100client/cluster/CIFAR_100client_quantitative_cluster.json" --num_rounds 500 --num_epochs 5 --proportion 0.5 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_resnet9_clustered_N100_K50 --model resnet9 --kd_fct 4 -sthr 0.9 --algorithm fedsdiv_transS --data_folder "./benchmark/cifar100/data" --log_folder "fedtask" --dataidx_filename "cifar/100client/cluster/CIFAR_100client_quantitative_cluster.json" --num_rounds 500 --num_epochs 5 --proportion 0.5 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_resnet9_clustered_N100_K50 --model resnet9 --kd_fct 6 -sthr 0.9 --algorithm fedsdiv_transS --data_folder "./benchmark/cifar100/data" --log_folder "fedtask" --dataidx_filename "cifar/100client/cluster/CIFAR_100client_quantitative_cluster.json" --num_rounds 500 --num_epochs 5 --proportion 0.5 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_resnet9_clustered_N100_K50 --model resnet9 --kd_fct 8 -sthr 0.9 --algorithm fedsdiv_transS --data_folder "./benchmark/cifar100/data" --log_folder "fedtask" --dataidx_filename "cifar/100client/cluster/CIFAR_100client_quantitative_cluster.json" --num_rounds 500 --num_epochs 5 --proportion 0.5 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_resnet9_clustered_N100_K50 --model resnet9 --kd_fct 10 -sthr 0.9 --algorithm fedsdiv_transS --data_folder "./benchmark/cifar100/data" --log_folder "fedtask" --dataidx_filename "cifar/100client/cluster/CIFAR_100client_quantitative_cluster.json" --num_rounds 500 --num_epochs 5 --proportion 0.5 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0

CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_resnet9_featured_N10_K10 --model resnet9 --kd_fct 2 -sthr 0.9 --algorithm fedsdiv_transS --data_folder "./benchmark/cifar100/data" --log_folder "fedtask" --dataidx_filename "cifar/featured/CIFAR-noniid-featured_1.json" --num_rounds 500 --num_epochs 5  --proportion 1 --batch_size 8  --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_resnet9_featured_N10_K10 --model resnet9 --kd_fct 4 -sthr 0.9 --algorithm fedsdiv_transS --data_folder "./benchmark/cifar100/data" --log_folder "fedtask" --dataidx_filename "cifar/featured/CIFAR-noniid-featured_1.json" --num_rounds 500 --num_epochs 5  --proportion 1 --batch_size 8  --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_resnet9_featured_N10_K10 --model resnet9 --kd_fct 6 -sthr 0.9 --algorithm fedsdiv_transS --data_folder "./benchmark/cifar100/data" --log_folder "fedtask" --dataidx_filename "cifar/featured/CIFAR-noniid-featured_1.json" --num_rounds 500 --num_epochs 5  --proportion 1 --batch_size 8  --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_resnet9_featured_N10_K10 --model resnet9 --kd_fct 8 -sthr 0.9 --algorithm fedsdiv_transS --data_folder "./benchmark/cifar100/data" --log_folder "fedtask" --dataidx_filename "cifar/featured/CIFAR-noniid-featured_1.json" --num_rounds 500 --num_epochs 5  --proportion 1 --batch_size 8  --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_resnet9_featured_N10_K10 --model resnet9 --kd_fct 10 -sthr 0.9 --algorithm fedsdiv_transS --data_folder "./benchmark/cifar100/data" --log_folder "fedtask" --dataidx_filename "cifar/featured/CIFAR-noniid-featured_1.json" --num_rounds 500 --num_epochs 5  --proportion 1 --batch_size 8  --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0

CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_resnet9_pareto_N10_K10 --model resnet9 --kd_fct 2 -sthr 0.9 --algorithm fedsdiv_transS --data_folder "./benchmark/cifar100/data" --log_folder "fedtask" --dataidx_filename "cifar/pareto/CIFAR-noniid-fedavg_pareto_1.json" --num_rounds 500 --num_epochs 5  --proportion 1 --batch_size 8  --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_resnet9_pareto_N10_K10 --model resnet9 --kd_fct 4 -sthr 0.9 --algorithm fedsdiv_transS --data_folder "./benchmark/cifar100/data" --log_folder "fedtask" --dataidx_filename "cifar/pareto/CIFAR-noniid-fedavg_pareto_1.json" --num_rounds 500 --num_epochs 5  --proportion 1 --batch_size 8  --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_resnet9_pareto_N10_K10 --model resnet9 --kd_fct 6 -sthr 0.9 --algorithm fedsdiv_transS --data_folder "./benchmark/cifar100/data" --log_folder "fedtask" --dataidx_filename "cifar/pareto/CIFAR-noniid-fedavg_pareto_1.json" --num_rounds 500 --num_epochs 5  --proportion 1 --batch_size 8  --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_resnet9_pareto_N10_K10 --model resnet9 --kd_fct 8 -sthr 0.9 --algorithm fedsdiv_transS --data_folder "./benchmark/cifar100/data" --log_folder "fedtask" --dataidx_filename "cifar/pareto/CIFAR-noniid-fedavg_pareto_1.json" --num_rounds 500 --num_epochs 5  --proportion 1 --batch_size 8  --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_resnet9_pareto_N10_K10 --model resnet9 --kd_fct 10 -sthr 0.9 --algorithm fedsdiv_transS --data_folder "./benchmark/cifar100/data" --log_folder "fedtask" --dataidx_filename "cifar/pareto/CIFAR-noniid-fedavg_pareto_1.json" --num_rounds 500 --num_epochs 5  --proportion 1 --batch_size 8  --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0

CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_resnet9_quantitative_N10_K10 --model resnet9 --kd_fct 2 -sthr 0.9 --algorithm fedsdiv_transS --data_folder "./benchmark/cifar100/data" --log_folder "fedtask" --dataidx_filename "cifar/quantitative/CIFAR-noniid-quantitative_1.json" --num_rounds 500 --num_epochs 5  --proportion 1 --batch_size 8  --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_resnet9_quantitative_N10_K10 --model resnet9 --kd_fct 4 -sthr 0.9 --algorithm fedsdiv_transS --data_folder "./benchmark/cifar100/data" --log_folder "fedtask" --dataidx_filename "cifar/quantitative/CIFAR-noniid-quantitative_1.json" --num_rounds 500 --num_epochs 5  --proportion 1 --batch_size 8  --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_resnet9_quantitative_N10_K10 --model resnet9 --kd_fct 6 -sthr 0.9 --algorithm fedsdiv_transS --data_folder "./benchmark/cifar100/data" --log_folder "fedtask" --dataidx_filename "cifar/quantitative/CIFAR-noniid-quantitative_1.json" --num_rounds 500 --num_epochs 5  --proportion 1 --batch_size 8  --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_resnet9_quantitative_N10_K10 --model resnet9 --kd_fct 8 -sthr 0.9 --algorithm fedsdiv_transS --data_folder "./benchmark/cifar100/data" --log_folder "fedtask" --dataidx_filename "cifar/quantitative/CIFAR-noniid-quantitative_1.json" --num_rounds 500 --num_epochs 5  --proportion 1 --batch_size 8  --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar100_resnet9_quantitative_N10_K10 --model resnet9 --kd_fct 10 -sthr 0.9 --algorithm fedsdiv_transS --data_folder "./benchmark/cifar100/data" --log_folder "fedtask" --dataidx_filename "cifar/quantitative/CIFAR-noniid-quantitative_1.json" --num_rounds 500 --num_epochs 5  --proportion 1 --batch_size 8  --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
