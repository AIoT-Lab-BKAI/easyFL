
for noise_rate_ in '0.5'
do
    for proportion_ in 0.2
    do
        for malicious_client_ in 20
        do
            # echo "$malicious_client_"
            for noise_type_ in gaussian 
            do
                for aggregate_ in  mean 
                do
                    count=0
                    # echo "$count"
                    dirty_rate_=(0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0)

                    if [ $noise_rate_ == '0.5' ]
                        then
                            dirty_rate_=(0 0 0 0.5 0.5 0 0.5 0 0.5 0 0 0 0.5 0.5 0 0.5 0 0.5 0 0.5 0 0 0 0 0 0.5 0.5 0 0 0 0.5 0 0.5 0 0 0 0 0.5 0 0.5 0 0.5 0 0 0 0.5 0.5 0.5 0.5 0)
                    elif [ $noise_rate_ == '1.0' ]
                        then 
                            dirty_rate_=(0 0 0 1.0 1.0 0 1.0 0 1.0 0 0 0 1.0 1.0 0 1.0 0 1.0 0 1.0 0 0 0 0 0 1.0 1.0 0 0 0 1.0 0 1.0 0 0 0 0 1.0 0 1.0 0 1.0 0 0 0 1.0 1.0 1.0 1.0 0)

                    fi
                    # dirty_rate_=()
                    # while [ $count -lt 50 ]
                    # do
                    #     if [ $count -lt $malicious_client_ ]
                    #         then
                    #             dirty_rate_+=($noise_rate_)
                    #             count=$(( $count + 1 ))
                    #     elif [ $count -lt 50 ]
                    #         then 
                    #             dirty_rate_+=(0)
                    #             count=$(( $count + 1 ))
                    #     fi
                    #     # echo "$count"
                    # done
                    echo "The number of malicous clients is $malicious_client_"
                    echo "${dirty_rate_[@]}"

                    CUDA_VISIBLE_DEVICES=0,1 python main.py  --task dirtycifar10_noniid_50client_cluster --dirty_rate ${dirty_rate_[@]} --noise_magnitude 1 --algorithm mp_fedavg --model resnet18 --data_folder "./benchmark/cifar10/data" --log_folder fedtask --dataidx_filename "cifar10/noniid/CIFAR10_50client_cluster.json" --num_rounds 200 --num_epochs 5 --proportion $proportion_ --batch_size 64 --num_threads_per_gpu 1  --num_gpus 1 --server_gpu_id 0 --wandb 0 --uncertainty 0  --aggregate $aggregate_ --noise_type $noise_type_ --num_malicious $malicious_client_ --attacked_class 0 1 2 3 4 5 6 7 8 9 --agg_algorithm cluster_fedavg_aggregate_client_32 --mu 0.1 --alpha 0.001 --client_id 32
                done
            done
        done
    done
done



for noise_rate_ in '0.5'
do
    for proportion_ in 0.2
    do
        for malicious_client_ in 20
        do
            # echo "$malicious_client_"
            for noise_type_ in gaussian 
            do
                for aggregate_ in  mean 
                do
                    count=0
                    # echo "$count"
                    dirty_rate_=(0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0)

                    if [ $noise_rate_ == '0.5' ]
                        then
                            dirty_rate_=(0 0 0 0.5 0.5 0 0.5 0 0.5 0 0 0 0.5 0.5 0 0.5 0 0.5 0 0.5 0 0 0 0 0 0.5 0.5 0 0 0 0.5 0 0.5 0 0 0 0 0.5 0 0.5 0 0.5 0 0 0 0.5 0.5 0.5 0.5 0)
                    elif [ $noise_rate_ == '1.0' ]
                        then 
                            dirty_rate_=(0 0 0 1.0 1.0 0 1.0 0 1.0 0 0 0 1.0 1.0 0 1.0 0 1.0 0 1.0 0 0 0 0 0 1.0 1.0 0 0 0 1.0 0 1.0 0 0 0 0 1.0 0 1.0 0 1.0 0 0 0 1.0 1.0 1.0 1.0 0)

                    fi
                    # dirty_rate_=()
                    # while [ $count -lt 50 ]
                    # do
                    #     if [ $count -lt $malicious_client_ ]
                    #         then
                    #             dirty_rate_+=($noise_rate_)
                    #             count=$(( $count + 1 ))
                    #     elif [ $count -lt 50 ]
                    #         then 
                    #             dirty_rate_+=(0)
                    #             count=$(( $count + 1 ))
                    #     fi
                    #     # echo "$count"
                    # done
                    echo "The number of malicous clients is $malicious_client_"
                    echo "${dirty_rate_[@]}"

                    CUDA_VISIBLE_DEVICES=0,1 python main.py  --task dirtycifar10_noniid_50client_cluster --dirty_rate ${dirty_rate_[@]} --noise_magnitude 1 --algorithm mp_fedavg --model resnet18 --data_folder "./benchmark/cifar10/data" --log_folder fedtask --dataidx_filename "cifar10/noniid/CIFAR10_50client_cluster.json" --num_rounds 200 --num_epochs 5 --proportion $proportion_ --batch_size 64 --num_threads_per_gpu 1  --num_gpus 1 --server_gpu_id 0 --wandb 0 --uncertainty 0  --aggregate $aggregate_ --noise_type $noise_type_ --num_malicious $malicious_client_ --attacked_class 0 1 2 3 4 5 6 7 8 9 --agg_algorithm cluster_fedavg_aggregate_client_40 --mu 0.1 --alpha 0.001 --client_id 40
                done
            done
        done
    done
done
