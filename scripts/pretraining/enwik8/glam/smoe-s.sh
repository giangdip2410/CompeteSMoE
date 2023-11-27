mkdir -p checkpoints/enwik8/glam-smoe-s

args="
--data /home/gtruong/Project/ICML2/data/enwik8 \
--base_arch glam \
--architecture sgsfsgsfsgsf \
--gate_name smoe \
--nlayers 3 \
--hid-sz 264 \
--inner-hid-sz 264 \
--nheads 8 \
--block-sz 512 \
--attn-span 2048 \
--dropout 0.1 \
--load_balance 0.01 \
--optim adam \
--lr 0.0007 \
--lr-warmup 3000 \
--niter 1 \
--batch-sz 48 \
--batch-split 2 \
--nbatches 60000 \
--checkpoint checkpoints/enwik8/glam-smoe-s/smoe.pt \
"

echo "Training ..."
python train.py $args

echo "Evaluation ..."
python train.py $args --full-eval-mode --batch-sz 8