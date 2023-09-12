CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar10_resnet9_quantitative_N100_K10 --model resnet9 --algorithm feddyn --data_folder "./benchmark/cifar10/data" --log_folder "fedtask" --dataidx_filename "cifar10/100client/CIFAR10_100client_quantitative.json" --num_rounds 1000 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar10_resnet9_quantitative_N100_K10 --model resnet9 --algorithm fedfa --data_folder "./benchmark/cifar10/data" --log_folder "fedtask" --dataidx_filename "cifar10/100client/CIFAR10_100client_quantitative.json" --num_rounds 1000 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar10_resnet9_quantitative_N100_K10 --model resnet9 --algorithm fedfv --data_folder "./benchmark/cifar10/data" --log_folder "fedtask" --dataidx_filename "cifar10/100client/CIFAR10_100client_quantitative.json" --num_rounds 1000 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar10_resnet9_quantitative_N100_K10 --model resnet9 --algorithm fedsdiv_transS --data_folder "./benchmark/cifar10/data" --log_folder "fedtask" --dataidx_filename "cifar10/100client/CIFAR10_100client_quantitative.json" --num_rounds 1000 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar10_resnet9_quantitative_N100_K10 --model resnet9 --algorithm mp_fedavg --data_folder "./benchmark/cifar10/data" --log_folder "fedtask" --dataidx_filename "cifar10/100client/CIFAR10_100client_quantitative.json" --num_rounds 1000 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar10_resnet9_quantitative_N100_K10 --model resnet9 --algorithm mp_fedprox --data_folder "./benchmark/cifar10/data" --log_folder "fedtask" --dataidx_filename "cifar10/100client/CIFAR10_100client_quantitative.json" --num_rounds 1000 --num_epochs 5 --proportion 0.1 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0

CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar10_resnet9_quantitative_N100_K20 --model resnet9 --algorithm feddyn --data_folder "./benchmark/cifar10/data" --log_folder "fedtask" --dataidx_filename "cifar10/100client/CIFAR10_100client_quantitative.json" --num_rounds 1000 --num_epochs 5 --proportion 0.2 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar10_resnet9_quantitative_N100_K20 --model resnet9 --algorithm fedfa --data_folder "./benchmark/cifar10/data" --log_folder "fedtask" --dataidx_filename "cifar10/100client/CIFAR10_100client_quantitative.json" --num_rounds 1000 --num_epochs 5 --proportion 0.2 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar10_resnet9_quantitative_N100_K20 --model resnet9 --algorithm fedfv --data_folder "./benchmark/cifar10/data" --log_folder "fedtask" --dataidx_filename "cifar10/100client/CIFAR10_100client_quantitative.json" --num_rounds 1000 --num_epochs 5 --proportion 0.2 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar10_resnet9_quantitative_N100_K20 --model resnet9 --algorithm fedsdiv_transS --data_folder "./benchmark/cifar10/data" --log_folder "fedtask" --dataidx_filename "cifar10/100client/CIFAR10_100client_quantitative.json" --num_rounds 1000 --num_epochs 5 --proportion 1 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar10_resnet9_quantitative_N100_K20 --model resnet9 --algorithm mp_fedavg --data_folder "./benchmark/cifar10/data" --log_folder "fedtask" --dataidx_filename "cifar10/100client/CIFAR10_100client_quantitative.json" --num_rounds 1000 --num_epochs 5 --proportion 0.2 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar10_resnet9_quantitative_N100_K20 --model resnet9 --algorithm mp_fedprox --data_folder "./benchmark/cifar10/data" --log_folder "fedtask" --dataidx_filename "cifar10/100client/CIFAR10_100client_quantitative.json" --num_rounds 1000 --num_epochs 5 --proportion 0.2 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0

CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar10_resnet9_quantitative_N100_K30 --model resnet9 --algorithm feddyn --data_folder "./benchmark/cifar10/data" --log_folder "fedtask" --dataidx_filename "cifar10/100client/CIFAR10_100client_quantitative.json" --num_rounds 1000 --num_epochs 5 --proportion 0.3 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar10_resnet9_quantitative_N100_K30 --model resnet9 --algorithm fedfa --data_folder "./benchmark/cifar10/data" --log_folder "fedtask" --dataidx_filename "cifar10/100client/CIFAR10_100client_quantitative.json" --num_rounds 1000 --num_epochs 5 --proportion 0.3 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar10_resnet9_quantitative_N100_K30 --model resnet9 --algorithm fedfv --data_folder "./benchmark/cifar10/data" --log_folder "fedtask" --dataidx_filename "cifar10/100client/CIFAR10_100client_quantitative.json" --num_rounds 1000 --num_epochs 5 --proportion 0.3 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar10_resnet9_quantitative_N100_K30 --model resnet9 --algorithm fedsdiv_transS --data_folder "./benchmark/cifar10/data" --log_folder "fedtask" --dataidx_filename "cifar10/100client/CIFAR10_100client_quantitative.json" --num_rounds 1000 --num_epochs 5 --proportion 0.3 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar10_resnet9_quantitative_N100_K30 --model resnet9 --algorithm mp_fedavg --data_folder "./benchmark/cifar10/data" --log_folder "fedtask" --dataidx_filename "cifar10/100client/CIFAR10_100client_quantitative.json" --num_rounds 1000 --num_epochs 5 --proportion 0.3 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar10_resnet9_quantitative_N100_K30 --model resnet9 --algorithm mp_fedprox --data_folder "./benchmark/cifar10/data" --log_folder "fedtask" --dataidx_filename "cifar10/100client/CIFAR10_100client_quantitative.json" --num_rounds 1000 --num_epochs 5 --proportion 0.3 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0

CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar10_resnet9_quantitative_N100_K40 --model resnet9 --algorithm feddyn --data_folder "./benchmark/cifar10/data" --log_folder "fedtask" --dataidx_filename "cifar10/100client/CIFAR10_100client_quantitative.json" --num_rounds 1000 --num_epochs 5 --proportion 0.4 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar10_resnet9_quantitative_N100_K40 --model resnet9 --algorithm fedfa --data_folder "./benchmark/cifar10/data" --log_folder "fedtask" --dataidx_filename "cifar10/100client/CIFAR10_100client_quantitative.json" --num_rounds 1000 --num_epochs 5 --proportion 0.4 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar10_resnet9_quantitative_N100_K40 --model resnet9 --algorithm fedfv --data_folder "./benchmark/cifar10/data" --log_folder "fedtask" --dataidx_filename "cifar10/100client/CIFAR10_100client_quantitative.json" --num_rounds 1000 --num_epochs 5 --proportion 0.4 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar10_resnet9_quantitative_N100_K40 --model resnet9 --algorithm fedsdiv_transS --data_folder "./benchmark/cifar10/data" --log_folder "fedtask" --dataidx_filename "cifar10/100client/CIFAR10_100client_quantitative.json" --num_rounds 1000 --num_epochs 5 --proportion 0.4 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar10_resnet9_quantitative_N100_K40 --model resnet9 --algorithm mp_fedavg --data_folder "./benchmark/cifar10/data" --log_folder "fedtask" --dataidx_filename "cifar10/100client/CIFAR10_100client_quantitative.json" --num_rounds 1000 --num_epochs 5 --proportion 0.4 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar10_resnet9_quantitative_N100_K40 --model resnet9 --algorithm mp_fedprox --data_folder "./benchmark/cifar10/data" --log_folder "fedtask" --dataidx_filename "cifar10/100client/CIFAR10_100client_quantitative.json" --num_rounds 1000 --num_epochs 5 --proportion 0.4 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0

CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar10_resnet9_quantitative_N100_K50 --model resnet9 --algorithm feddyn --data_folder "./benchmark/cifar10/data" --log_folder "fedtask" --dataidx_filename "cifar10/100client/CIFAR10_100client_quantitative.json" --num_rounds 1000 --num_epochs 5 --proportion 0.5 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar10_resnet9_quantitative_N100_K50 --model resnet9 --algorithm fedfa --data_folder "./benchmark/cifar10/data" --log_folder "fedtask" --dataidx_filename "cifar10/100client/CIFAR10_100client_quantitative.json" --num_rounds 1000 --num_epochs 5 --proportion 0.5 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar10_resnet9_quantitative_N100_K50 --model resnet9 --algorithm fedfv --data_folder "./benchmark/cifar10/data" --log_folder "fedtask" --dataidx_filename "cifar10/100client/CIFAR10_100client_quantitative.json" --num_rounds 1000 --num_epochs 5 --proportion 0.5 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar10_resnet9_quantitative_N100_K50 --model resnet9 --algorithm fedsdiv_transS --data_folder "./benchmark/cifar10/data" --log_folder "fedtask" --dataidx_filename "cifar10/100client/CIFAR10_100client_quantitative.json" --num_rounds 1000 --num_epochs 5 --proportion 0.5 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar10_resnet9_quantitative_N100_K50 --model resnet9 --algorithm mp_fedavg --data_folder "./benchmark/cifar10/data" --log_folder "fedtask" --dataidx_filename "cifar10/100client/CIFAR10_100client_quantitative.json" --num_rounds 1000 --num_epochs 5 --proportion 0.5 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar10_resnet9_quantitative_N100_K50 --model resnet9 --algorithm mp_fedprox --data_folder "./benchmark/cifar10/data" --log_folder "fedtask" --dataidx_filename "cifar10/100client/CIFAR10_100client_quantitative.json" --num_rounds 1000 --num_epochs 5 --proportion 0.5 --batch_size 8 --num_threads_per_gpu 2  --num_gpus 2 --server_gpu_id 0
