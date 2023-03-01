
for noise_rate_ in '0.5'
do
    for proportion_ in 0.2
    do
        for malicious_client_ in 25
        do
            # echo "$malicious_client_"
            for noise_type_ in salt_pepper 
            do
                for aggregate_ in  mean 
                do
                    count=0
                    # echo "$count"
                    dirty_rate_=(0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0)

                    if [ $noise_rate_ == '0.5' ]
                        then
                            dirty_rate_=(0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0 1.0 1.0 1.0 1.0 1.0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
                    elif [ $noise_rate_ == '1.0' ]
                        then 
                            dirty_rate_=(1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
                    fi
                    
                    echo "The number of malicous clients is $malicious_client_"
                    echo "${dirty_rate_[@]}"

                    CUDA_VISIBLE_DEVICES=0,1 python main.py  --task dirtycifar10_iid_50client_25attacker --dirty_rate ${dirty_rate_[@]} --noise_magnitude 0.1 --algorithm mp_fedavg --model resnet18 --data_folder "./benchmark/cifar10/data" --log_folder fedtask --dataidx_filename "cifar10/iid/cifar10_iid_50client_1000data.json" --num_rounds 200 --num_epochs 5 --proportion 0.2 --batch_size 64 --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0 --wandb 1 --uncertainty 0  --aggregate $aggregate_ --noise_type $noise_type_ --num_malicious $malicious_client_ --attacked_class 0 1 2 3 4 5 6 7 8 9 --agg_algorithm "pareto_peak_or_cs_choose_normal_aggregate_attacker_by_cs_(csi+0.05*mincs/maxcs)*ni/N" --mu 0.1 --alpha 0.001 --client_id 7 --outside_noise inside
                    CUDA_VISIBLE_DEVICES=0,1 python main.py  --task dirtycifar10_iid_50client_25attacker --dirty_rate ${dirty_rate_[@]} --noise_magnitude 0.1 --algorithm mp_fedavg --model resnet18 --data_folder "./benchmark/cifar10/data" --log_folder fedtask --dataidx_filename "cifar10/iid/cifar10_iid_50client_1000data.json" --num_rounds 200 --num_epochs 5 --proportion 0.2 --batch_size 64 --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0 --wandb 1 --uncertainty 0  --aggregate $aggregate_ --noise_type $noise_type_ --num_malicious $malicious_client_ --attacked_class 0 1 2 3 4 5 6 7 8 9 --agg_algorithm "pareto_peak_or_cs_choose_normal_aggregate_attacker_by_(tanhe(csi+0.05*mincs/maxcs))*ni/N" --mu 0.1 --alpha 0.001 --client_id 7 --outside_noise inside
                    CUDA_VISIBLE_DEVICES=0,1 python main.py  --task dirtycifar10_iid_50client_25attacker --dirty_rate ${dirty_rate_[@]} --noise_magnitude 0.1 --algorithm mp_fedavg --model resnet18 --data_folder "./benchmark/cifar10/data" --log_folder fedtask --dataidx_filename "cifar10/iid/cifar10_iid_50client_1000data.json" --num_rounds 200 --num_epochs 5 --proportion 0.2 --batch_size 64 --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0 --wandb 1 --uncertainty 0  --aggregate $aggregate_ --noise_type $noise_type_ --num_malicious $malicious_client_ --attacked_class 0 1 2 3 4 5 6 7 8 9 --agg_algorithm "pareto_fedavg" --mu 0.1 --alpha 0.001 --client_id 7 --outside_noise inside
                done
            done
        done
    done
done

for noise_rate_ in '0.5'
do
    for proportion_ in 0.2
    do
        for malicious_client_ in 25
        do
            # echo "$malicious_client_"
            for noise_type_ in salt_pepper 
            do
                for aggregate_ in  median trmean krum mkrum bulyan 
                do
                    count=0
                    # echo "$count"
                    dirty_rate_=(0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0)

                    if [ $noise_rate_ == '0.5' ]
                        then
                            dirty_rate_=(0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0 1.0 1.0 1.0 1.0 1.0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
                    elif [ $noise_rate_ == '1.0' ]
                        then 
                            dirty_rate_=(1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
                    fi
                    
                    echo "The number of malicous clients is $malicious_client_"
                    echo "${dirty_rate_[@]}"

                    CUDA_VISIBLE_DEVICES=0,1 python main.py  --task dirtycifar10_iid_50client_25attacker --dirty_rate ${dirty_rate_[@]} --noise_magnitude 0.1 --algorithm mp_fedavg --model resnet18 --data_folder "./benchmark/cifar10/data" --log_folder fedtask --dataidx_filename "cifar10/iid/cifar10_iid_50client_1000data.json" --num_rounds 200 --num_epochs 5 --proportion 0.2 --batch_size 64 --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0 --wandb 1 --uncertainty 0  --aggregate $aggregate_ --noise_type $noise_type_ --num_malicious $malicious_client_ --attacked_class 0 1 2 3 4 5 6 7 8 9 --agg_algorithm $aggregate_ --mu 0.1 --alpha 0.001 --client_id 7 --outside_noise inside
                done
            done
        done
    done
done