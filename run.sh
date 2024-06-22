if [[ $1 -eq 0 ]] ; then
    echo 'missing argument: number of gpus'
    exit 1
fi

sleep 5
echo "after sleep"
exit 0


pip install tiktoken
pip install datasets
pip install tqdm

python fineweb.py
ls edu_fineweb10B | wc -l
torchrun --standalone --nproc_per_node=$1 train_gpt2.py