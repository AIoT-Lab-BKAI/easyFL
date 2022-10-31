for malicious_client_ in 0 10 20 30 40 50
do
    # echo "$malicious_client_"
    for noise_type_ in salt_pepper
    do
        for aggregate_ in  bulyan
        do
            count=0
            # echo "$count"

            dirty_rate_=()
            while [ $count -lt 50 ]
            do
                if [ $count -lt $malicious_client_ ]
                    then
                        dirty_rate_+=(0.5)
                        count=$(( $count + 1 ))
                elif [ $count -lt 50 ]
                    then 
                        dirty_rate_+=(0)
                        count=$(( $count + 1 ))
                fi
                # echo "$count"
            done
            echo "The number of malicous clients is $malicious_client_"
            echo "${dirty_rate_[@]}"

            CUDA_VISIBLE_DEVICES=1,3 python main.py  --task dirtycifar10_iid_50client_1000data --dirty_rate ${dirty_rate_[@]} --noise_magnitude 0.1 --algorithm mp_fedavg --model cnn --data_folder "./benchmark/cifar10/data" --log_folder fedtask --dataidx_filename "cifar10/iid/cifar10_iid_50client_1000data.json" --num_rounds 100 --num_epochs 3 --proportion 0.2 --batch_size 64 --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0 --wandb 1  --uncertainty 0  --aggregate $aggregate_ --noise_type $noise_type_ --num_malicious $malicious_client_
        done
    done
done


for malicious_client_ in 0 10 20 30 40 50
do
    # echo "$malicious_client_"
    for noise_type_ in gaussian poisson speckle
    do
        for aggregate_ in  bulyan
        do
            count=0
            # echo "$count"

            dirty_rate_=()
            while [ $count -lt 50 ]
            do
                if [ $count -lt $malicious_client_ ]
                    then
                        dirty_rate_+=(0.5)
                        count=$(( $count + 1 ))
                elif [ $count -lt 50 ]
                    then 
                        dirty_rate_+=(0)
                        count=$(( $count + 1 ))
                fi
                # echo "$count"
            done
            echo "The number of malicous clients is $malicious_client_"
            echo "${dirty_rate_[@]}"

            CUDA_VISIBLE_DEVICES=1,3 python main.py  --task dirtycifar10_iid_50client_1000data --dirty_rate ${dirty_rate_[@]} --noise_magnitude 1 --algorithm mp_fedavg --model cnn --data_folder "./benchmark/cifar10/data" --log_folder fedtask --dataidx_filename "cifar10/iid/cifar10_iid_50client_1000data.json" --num_rounds 100 --num_epochs 3 --proportion 0.2 --batch_size 64 --num_threads_per_gpu 1  --num_gpus 2 --server_gpu_id 0 --wandb 1  --uncertainty 0  --aggregate $aggregate_ --noise_type $noise_type_ --num_malicious $malicious_client_
        done
    done
done
