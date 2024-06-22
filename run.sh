if [[ $1 -eq 0 ]] ; then
    echo 'missing argument: number of gpus'
    exit 1
fi

pip install tiktoken
pip install datasets
pip install tqdm
pip install transformers

python fineweb.py
echo "number of shards:"
ls edu_fineweb10B | wc -l
torchrun --standalone --nproc_per_node=$1 train_gpt2.py

git add log/log-train-eval.txt
git add gpt2-finetuned.pth
git commit -am "finetuning complete"
git push
