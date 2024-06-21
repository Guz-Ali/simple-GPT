if [[ $1 -eq 0 ]] ; then
    echo 'missing argument: number of gpus'
    exit 1
fi

python fineweb.py
ls edu_fineweb10B | wc -l
torchrun --standalone --nproc_per_node=$1 train_gpt2.py