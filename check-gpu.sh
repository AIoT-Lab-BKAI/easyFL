for (( i=1; i<=10; i++ ))
do
    echo "Iteration $i"
    srun --nodelist="slurmnode$i" nvidia-smi
done