mkdir fedtask

# cifar10_resnet9_featured_N10_K10 - featured
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar10_resnet9_featured_N10_K10 --model resnet9 --algorithm fedtest --neg_fct 2 --data_folder "./benchmark/cifar10/data" --log_folder "fedtask" --dataidx_filename "cifar/featured/CIFAR-noniid-featured_1.json" --num_rounds 500 --num_epochs 5  --proportion 1 --batch_size 8  --num_threads_per_gpu 1  --num_gpus 1 --server_gpu_id 0

# cifar10_resnet9_pareto_N10_K10 - pareto
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar10_resnet9_pareto_N10_K10 --model resnet9 --algorithm fedtest --neg_fct 2 --data_folder "./benchmark/cifar10/data" --log_folder "fedtask" --dataidx_filename "cifar/pareto/CIFAR-noniid-fedavg_pareto_1.json" --num_rounds 500 --num_epochs 5  --proportion 1 --batch_size 8  --num_threads_per_gpu 1  --num_gpus 1 --server_gpu_id 0

# cifar10_resnet9_quantitative_N10_K10 - quantitative
CUDA_VISIBLE_DEVICES=0,1 python main.py --task cifar10_resnet9_quantitative_N10_K10 --model resnet9 --algorithm fedtest --neg_fct 2 --data_folder "./benchmark/cifar10/data" --log_folder "fedtask" --dataidx_filename "cifar/quantitative/CIFAR-noniid-quantitative_1.json" --num_rounds 500 --num_epochs 5  --proportion 1 --batch_size 8  --num_threads_per_gpu 1  --num_gpus 1 --server_gpu_id 0
