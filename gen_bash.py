dist_list = [1,2,3]
skew_list = [0.2, 0.4, 0.6, 0.8]
cnum_list = [10]
algo_list = ["fedavg", "fedprox", "fedtheo", "feddf", "fedreg", "fedgeo", "fedsdiv"]

with open("experiments.sh", "w") as file:
    for dist in dist_list:
        file.write(f"#dist {dist}\n")
        for skew in skew_list:
            file.write(f"#skew {skew}\n")
            for cnum in cnum_list:
                for algo in algo_list:
                    nround = int(50 * skew + 10)
                    command_format = f"CUDA_VISIBLE_DEVICES=0,1 python main.py --task mnist_cnum{cnum}_dist{dist}_skew{skew}_seed0 --model cnn --algorithm {algo} --num_rounds {nround} --num_epochs 4 --learning_rate 0.001 --proportion 1 --batch_size 4 --eval_interval 1 --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0"
                    file.write(f"{command_format}\n")
                    
            file.write(f"\n")
        file.write(f"\n")