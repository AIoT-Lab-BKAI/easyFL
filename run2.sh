# #noise effect
# #change noise rate
# for noise_rate_ in '0' '0.25' '0.5' '0.75' '1.0'
# do
#     for proportion_ in 0.2
#     do
#         for malicious_client_ in 25
#         do
#             # echo "$malicious_client_"
#             for noise_type_ in gaussian salt_pepper speckle poisson  
#             do
#                 for aggregate_ in  mean 
#                 do
#                     count=0
#                     # echo "$count"
#                     dirty_rate_=(0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0)

#                     if [ $noise_rate_ == '0' ]
#                         then
#                             dirty_rate_=(0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
#                     elif [ $noise_rate_ == '0.25' ]
#                         then 
#                             dirty_rate_=(0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
#                     elif [ $noise_rate_ == '0.5' ]
#                         then 
#                             dirty_rate_=(0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
#                     elif [ $noise_rate_ == '0.75' ]
#                         then 
#                             dirty_rate_=(0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
#                     elif [ $noise_rate_ == '1.0' ]
#                         then 
#                             dirty_rate_=(1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
#                     fi
                    
#                     echo "The number of malicous clients is $malicious_client_"
#                     echo "${dirty_rate_[@]}"
#                     noise_magnitude_=1.0
#                     if [ $noise_type_ == 'salt_pepper' ]
#                         then
#                             noise_magnitude_=0.1
#                     fi

#                     # CUDA_VISIBLE_DEVICES=0,1 python main.py  --task dirtycifar10_iid_solution --dirty_rate ${dirty_rate_[@]} --noise_magnitude 0.1 --algorithm mp_fedavg --model resnet18 --data_folder "./benchmark/cifar10/data" --log_folder fedtask --dataidx_filename "cifar10/iid/cifar10_iid_50client_1000data.json" --num_rounds 2 --num_epochs 5 --proportion 0.2 --batch_size 64 --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0 --wandb 1 --uncertainty 0  --aggregate $aggregate_ --noise_type $noise_type_ --num_malicious $malicious_client_ --attacked_class 0 1 2 3 4 5 6 7 8 9 --agg_algorithm "peak_and_cs_choose_attacker_aggregate_attacker_by_(csi+0.05*mincs/maxcs)" --mu 0.1 --alpha 001 --client_id 7 --outside_noise inside
#                     # CUDA_VISIBLE_DEVICES=0,1 python main.py  --task dirtycifar10_iid_solution --dirty_rate ${dirty_rate_[@]} --noise_magnitude 0.1 --algorithm mp_fedavg --model resnet18 --data_folder "./benchmark/cifar10/data" --log_folder fedtask --dataidx_filename "cifar10/iid/cifar10_iid_50client_1000data.json" --num_rounds 2 --num_epochs 5 --proportion 0.2 --batch_size 64 --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0 --wandb 1 --uncertainty 0  --aggregate $aggregate_ --noise_type $noise_type_ --num_malicious $malicious_client_ --attacked_class 0 1 2 3 4 5 6 7 8 9 --agg_algorithm "peak_and_cs_choose_attacker_aggregate_attacker_by_(tanhe(csi+0.05*mincs/maxcs))" --mu 0.1 --alpha 001 --client_id 7 --outside_noise inside
#                     CUDA_VISIBLE_DEVICES=0,1 python main.py  --task dirtycifar10_iid_change_noise_rate --dirty_rate ${dirty_rate_[@]} --noise_magnitude $noise_magnitude_ --algorithm mp_fedavg --model resnet18 --data_folder "./benchmark/cifar10/data" --log_folder fedtask --dataidx_filename "cifar10/iid/cifar10_iid_50client_1000data.json" --num_rounds 2 --num_epochs 5 --proportion 0.2 --batch_size 64 --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0 --wandb 1 --uncertainty 0  --aggregate $aggregate_ --noise_type $noise_type_ --num_malicious $malicious_client_ --attacked_class 0 1 2 3 4 5 6 7 8 9 --agg_algorithm "fedavg" --mu 0.1 --alpha 001 --client_id 7 --outside_noise inside
#                 done
#             done
#         done
#     done
# done

# #change #attacker
# for noise_rate_ in '1.0'
# do
#     for proportion_ in 0.2
#     do
#         for malicious_client_ in 0 10 20 30 40
#         do
#             # echo "$malicious_client_"
#             for noise_type_ in gaussian salt_pepper speckle poisson  
#             do
#                 for aggregate_ in  mean 
#                 do
#                     count=0
#                     # echo "$count"
#                     dirty_rate_=(0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0)

#                     if [ $malicious_client_ == '0' ]
#                         then
#                             dirty_rate_=(0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
#                     elif [ $malicious_client_ == '10' ]
#                         then 
#                             dirty_rate_=(1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
#                     elif [ $malicious_client_ == '20' ]
#                         then 
#                             dirty_rate_=(1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
#                     elif [ $malicious_client_ == '30' ]
#                         then 
#                             dirty_rate_=(1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
#                     elif [ $malicious_client_ == '40' ]
#                         then 
#                             dirty_rate_=(1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 0 0 0 0 0 0 0 0 0 0)
#                     fi
                    
#                     echo "The number of malicous clients is $malicious_client_"
#                     echo "${dirty_rate_[@]}"
#                     noise_magnitude_=1.0
#                     if [ $noise_type_ == 'salt_pepper' ]
#                         then
#                             noise_magnitude_=0.1
#                     fi

#                     # CUDA_VISIBLE_DEVICES=0,1 python main.py  --task dirtycifar10_iid_solution --dirty_rate ${dirty_rate_[@]} --noise_magnitude 0.1 --algorithm mp_fedavg --model resnet18 --data_folder "./benchmark/cifar10/data" --log_folder fedtask --dataidx_filename "cifar10/iid/cifar10_iid_50client_1000data.json" --num_rounds 2 --num_epochs 5 --proportion 0.2 --batch_size 64 --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0 --wandb 1 --uncertainty 0  --aggregate $aggregate_ --noise_type $noise_type_ --num_malicious $malicious_client_ --attacked_class 0 1 2 3 4 5 6 7 8 9 --agg_algorithm "peak_and_cs_choose_attacker_aggregate_attacker_by_(csi+0.05*mincs/maxcs)" --mu 0.1 --alpha 001 --client_id 7 --outside_noise inside
#                     # CUDA_VISIBLE_DEVICES=0,1 python main.py  --task dirtycifar10_iid_solution --dirty_rate ${dirty_rate_[@]} --noise_magnitude 0.1 --algorithm mp_fedavg --model resnet18 --data_folder "./benchmark/cifar10/data" --log_folder fedtask --dataidx_filename "cifar10/iid/cifar10_iid_50client_1000data.json" --num_rounds 2 --num_epochs 5 --proportion 0.2 --batch_size 64 --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0 --wandb 1 --uncertainty 0  --aggregate $aggregate_ --noise_type $noise_type_ --num_malicious $malicious_client_ --attacked_class 0 1 2 3 4 5 6 7 8 9 --agg_algorithm "peak_and_cs_choose_attacker_aggregate_attacker_by_(tanhe(csi+0.05*mincs/maxcs))" --mu 0.1 --alpha 001 --client_id 7 --outside_noise inside
#                     CUDA_VISIBLE_DEVICES=0,1 python main.py  --task dirtycifar10_iid_change_num_attacker --dirty_rate ${dirty_rate_[@]} --noise_magnitude $noise_magnitude_ --algorithm mp_fedavg --model resnet18 --data_folder "./benchmark/cifar10/data" --log_folder fedtask --dataidx_filename "cifar10/iid/cifar10_iid_50client_1000data.json" --num_rounds 2 --num_epochs 5 --proportion 0.2 --batch_size 64 --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0 --wandb 1 --uncertainty 0  --aggregate $aggregate_ --noise_type $noise_type_ --num_malicious $malicious_client_ --attacked_class 0 1 2 3 4 5 6 7 8 9 --agg_algorithm "fedavg" --mu 0.1 --alpha 001 --client_id 7 --outside_noise inside
#                 done
#             done
#         done
#     done
# done

# #change attacked classes
# for noise_rate_ in '1.0'
# do
#     for proportion_ in 0.2
#     do
#         for malicious_client_ in 25
#         do
#             # echo "$malicious_client_"
#             for noise_type_ in gaussian salt_pepper speckle poisson  
#             do
#                 for aggregate_ in  mean 
#                 do
#                     count=0
#                     # echo "$count"
#                     dirty_rate_=(0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0)

#                     if [ $noise_rate_ == '0' ]
#                         then
#                             dirty_rate_=(0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
#                     elif [ $noise_rate_ == '0.25' ]
#                         then 
#                             dirty_rate_=(0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
#                     elif [ $noise_rate_ == '0.5' ]
#                         then 
#                             dirty_rate_=(0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
#                     elif [ $noise_rate_ == '0.75' ]
#                         then 
#                             dirty_rate_=(0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
#                     elif [ $noise_rate_ == '1.0' ]
#                         then 
#                             dirty_rate_=(1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
#                     fi
                    
#                     echo "The number of malicous clients is $malicious_client_"
#                     echo "${dirty_rate_[@]}"
#                     noise_magnitude_=1.0
#                     if [ $noise_type_ == 'salt_pepper' ]
#                         then
#                             noise_magnitude_=0.1
#                     fi

#                     CUDA_VISIBLE_DEVICES=0,1 python main.py  --task dirtycifar10_iid_change_attacked_classes --dirty_rate ${dirty_rate_[@]} --noise_magnitude $noise_magnitude_ --algorithm mp_fedavg --model resnet18 --data_folder "./benchmark/cifar10/data" --log_folder fedtask --dataidx_filename "cifar10/iid/cifar10_iid_50client_1000data.json" --num_rounds 2 --num_epochs 5 --proportion 0.2 --batch_size 64 --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0 --wandb 1 --uncertainty 0  --aggregate $aggregate_ --noise_type $noise_type_ --num_malicious $malicious_client_ --attacked_class 0 1  --agg_algorithm "fedavg" --mu 0.1 --alpha 001 --client_id 7 --outside_noise inside
#                     CUDA_VISIBLE_DEVICES=0,1 python main.py  --task dirtycifar10_iid_change_attacked_classes --dirty_rate ${dirty_rate_[@]} --noise_magnitude $noise_magnitude_ --algorithm mp_fedavg --model resnet18 --data_folder "./benchmark/cifar10/data" --log_folder fedtask --dataidx_filename "cifar10/iid/cifar10_iid_50client_1000data.json" --num_rounds 2 --num_epochs 5 --proportion 0.2 --batch_size 64 --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0 --wandb 1 --uncertainty 0  --aggregate $aggregate_ --noise_type $noise_type_ --num_malicious $malicious_client_ --attacked_class 0 1 2 3  --agg_algorithm "fedavg" --mu 0.1 --alpha 001 --client_id 7 --outside_noise inside
#                     CUDA_VISIBLE_DEVICES=0,1 python main.py  --task dirtycifar10_iid_change_attacked_classes --dirty_rate ${dirty_rate_[@]} --noise_magnitude $noise_magnitude_ --algorithm mp_fedavg --model resnet18 --data_folder "./benchmark/cifar10/data" --log_folder fedtask --dataidx_filename "cifar10/iid/cifar10_iid_50client_1000data.json" --num_rounds 2 --num_epochs 5 --proportion 0.2 --batch_size 64 --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0 --wandb 1 --uncertainty 0  --aggregate $aggregate_ --noise_type $noise_type_ --num_malicious $malicious_client_ --attacked_class 0 1 2 3 4 5  --agg_algorithm "fedavg" --mu 0.1 --alpha 001 --client_id 7 --outside_noise inside
#                     CUDA_VISIBLE_DEVICES=0,1 python main.py  --task dirtycifar10_iid_change_attacked_classes --dirty_rate ${dirty_rate_[@]} --noise_magnitude $noise_magnitude_ --algorithm mp_fedavg --model resnet18 --data_folder "./benchmark/cifar10/data" --log_folder fedtask --dataidx_filename "cifar10/iid/cifar10_iid_50client_1000data.json" --num_rounds 2 --num_epochs 5 --proportion 0.2 --batch_size 64 --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0 --wandb 1 --uncertainty 0  --aggregate $aggregate_ --noise_type $noise_type_ --num_malicious $malicious_client_ --attacked_class 0 1 2 3 4 5 6 7  --agg_algorithm "fedavg" --mu 0.1 --alpha 001 --client_id 7 --outside_noise inside
#                     CUDA_VISIBLE_DEVICES=0,1 python main.py  --task dirtycifar10_iid_change_attacked_classes --dirty_rate ${dirty_rate_[@]} --noise_magnitude $noise_magnitude_ --algorithm mp_fedavg --model resnet18 --data_folder "./benchmark/cifar10/data" --log_folder fedtask --dataidx_filename "cifar10/iid/cifar10_iid_50client_1000data.json" --num_rounds 2 --num_epochs 5 --proportion 0.2 --batch_size 64 --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0 --wandb 1 --uncertainty 0  --aggregate $aggregate_ --noise_type $noise_type_ --num_malicious $malicious_client_ --attacked_class 0 1 2 3 4 5 6 7 8 9  --agg_algorithm "fedavg" --mu 0.1 --alpha 001 --client_id 7 --outside_noise inside
#                 done
#             done
#         done
#     done
# done

#solution 
for noise_rate_ in '0' '0.25' '0.5' '0.75' '1.0'
do
    for proportion_ in 0.2
    do
        for malicious_client_ in 25
        do
            # echo "$malicious_client_"
            for noise_type_ in gaussian salt_pepper speckle poisson  
            do
                for aggregate_ in  mean 
                do
                    count=0
                    # echo "$count"
                    dirty_rate_=(0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0)

                    if [ $noise_rate_ == '0' ]
                        then
                            dirty_rate_=(0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
                    elif [ $noise_rate_ == '0.25' ]
                        then 
                            dirty_rate_=(0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
                    elif [ $noise_rate_ == '0.5' ]
                        then 
                            dirty_rate_=(0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
                    elif [ $noise_rate_ == '0.75' ]
                        then 
                            dirty_rate_=(0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
                    elif [ $noise_rate_ == '1.0' ]
                        then 
                            dirty_rate_=(1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
                    fi
                    
                    echo "The number of malicous clients is $malicious_client_"
                    echo "${dirty_rate_[@]}"
                    noise_magnitude_=1.0
                    if [ $noise_type_ == 'salt_pepper' ]
                        then
                            noise_magnitude_=0.1
                    fi

                    CUDA_VISIBLE_DEVICES=0,1 python main.py  --task dirtycifar10_iid_solution --dirty_rate ${dirty_rate_[@]} --noise_magnitude 0.1 --algorithm mp_fedavg --model resnet18 --data_folder "./benchmark/cifar10/data" --log_folder log_result --dataidx_filename "cifar10/iid/cifar10_iid_50client_1000data.json" --num_rounds 2 --num_epochs 5 --proportion 0.2 --batch_size 64 --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0 --wandb 0 --uncertainty 0  --aggregate $aggregate_ --noise_type $noise_type_ --num_malicious $malicious_client_ --attacked_class 0 1 2 3 4 5 6 7 8 9 --agg_algorithm "peak_and_cs_choose_attacker_aggregate_attacker_by_(csi+0.05*mincs/maxcs)" --mu 0.1 --alpha 001 --outside_noise inside
                    CUDA_VISIBLE_DEVICES=0,1 python main.py  --task dirtycifar10_iid_solution --dirty_rate ${dirty_rate_[@]} --noise_magnitude 0.1 --algorithm mp_fedavg --model resnet18 --data_folder "./benchmark/cifar10/data" --log_folder fedtask --dataidx_filename "cifar10/iid/cifar10_iid_50client_1000data.json" --num_rounds 2 --num_epochs 5 --proportion 0.2 --batch_size 64 --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0 --wandb 1 --uncertainty 0  --aggregate $aggregate_ --noise_type $noise_type_ --num_malicious $malicious_client_ --attacked_class 0 1 2 3 4 5 6 7 8 9 --agg_algorithm "peak_and_cs_choose_attacker_aggregate_attacker_by_(tanhe(csi+0.05*mincs/maxcs))" --mu 0.1 --alpha 001 --client_id 7 --outside_noise inside
                    CUDA_VISIBLE_DEVICES=0,1 python main.py  --task dirtycifar10_iid_solution --dirty_rate ${dirty_rate_[@]} --noise_magnitude $noise_magnitude_ --algorithm mp_fedavg --model resnet18 --data_folder "./benchmark/cifar10/data" --log_folder fedtask --dataidx_filename "cifar10/iid/cifar10_iid_50client_1000data.json" --num_rounds 2 --num_epochs 5 --proportion 0.2 --batch_size 64 --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0 --wandb 1 --uncertainty 0  --aggregate $aggregate_ --noise_type $noise_type_ --num_malicious $malicious_client_ --attacked_class 0 1 2 3 4 5 6 7 8 9 --agg_algorithm "fedavg" --mu 0.1 --alpha 001 --client_id 7 --outside_noise inside
                done
            done
        done
    done
done

for noise_rate_ in '0' '0.25' '0.5' '0.75' '1.0'
do
    for proportion_ in 0.2
    do
        for malicious_client_ in 25
        do
            # echo "$malicious_client_"
            for noise_type_ in gaussian salt_pepper speckle poisson 
            do
                for aggregate_ in  median trmean krum mkrum bulyan 
                do
                    count=0
                    # echo "$count"
                    dirty_rate_=(0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0)

                    if [ $noise_rate_ == '0' ]
                        then
                            dirty_rate_=(0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
                    elif [ $noise_rate_ == '0.25' ]
                        then 
                            dirty_rate_=(0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
                    elif [ $noise_rate_ == '0.5' ]
                        then 
                            dirty_rate_=(0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
                    elif [ $noise_rate_ == '0.75' ]
                        then 
                            dirty_rate_=(0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0.75 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
                    elif [ $noise_rate_ == '1.0' ]
                        then 
                            dirty_rate_=(1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0)
                    fi
                    
                    echo "The number of malicous clients is $malicious_client_"
                    echo "${dirty_rate_[@]}"
                    noise_magnitude_=1.0
                    if [ $noise_type_ == 'salt_pepper' ]
                        then
                            noise_magnitude_=0.1
                    fi

                    CUDA_VISIBLE_DEVICES=0,1 python main.py  --task dirtycifar10_iid_solution --dirty_rate ${dirty_rate_[@]} --noise_magnitude 1.0 --algorithm mp_fedavg --model resnet18 --data_folder "./benchmark/cifar10/data" --log_folder fedtask --dataidx_filename "cifar10/iid/cifar10_iid_50client_1000data.json" --num_rounds 2 --num_epochs 5 --proportion 0.2 --batch_size 64 --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0 --wandb 1 --uncertainty 0  --aggregate $aggregate_ --noise_type $noise_type_ --num_malicious $malicious_client_ --attacked_class 0 1 2 3 4 5 6 7 8 9  --agg_algorithm $aggregate_ --mu 0.1 --alpha 001 --client_id 7 --outside_noise inside
                done
            done
        done
    done
done
