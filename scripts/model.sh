gpu=2
seq_len=96 #lookback length
pred_len=96 #oredict length

# #WaveCMNet
##################################################
model_name=WaveCMNet
for random_seed in 2021 2022 2023 2024 2025; do
   python -u ../run_longExp.py \
        --model $model_name \
        --random_seed $random_seed \
        --seq_len $seq_len \
        --label_len 48 \
        --pred_len $pred_len \
        --gpu $gpu\
        --lradj   TST
done
##################################################


# #PatchTST
##################################################
model_name=PatchTST
for random_seed in 2021 2022 2023 2024 2025; do
   python -u ../run_longExp.py \
        --model $model_name \
        --random_seed $random_seed \
        --seq_len $seq_len \
        --label_len 48 \
        --gpu $gpu\
        --pred_len $pred_len\
        --lradj   TST
        
done
# ##################################################


#TimesNet
##################################################
model_name=TimesNet
for random_seed in 2021 2022 2023 2024 2025; do
   python -u ../run_longExp.py \
        --model $model_name \
        --random_seed $random_seed \
        --seq_len $seq_len \
        --label_len 48 \
        --gpu $gpu\
        --pred_len $pred_len \
        --lradj   TST
done
##################################################


#MICN
##################################################
model_name=MICN
for random_seed in 2021 2022 2023 2024 2025; do
   python -u ../run_longExp.py \
        --model $model_name \
        --random_seed $random_seed \
        --seq_len $seq_len \
        --label_len 48 \
        --gpu $gpu\
        --pred_len $pred_len \
        --lradj   TST
done
##################################################


#FreTS
##################################################
model_name=FreTSLinear
for random_seed in 2021 2022 2023 2024 2025; do
   python -u ../run_longExp.py \
        --model $model_name \
        --random_seed $random_seed \
        --seq_len $seq_len \
        --label_len 48 \
        --gpu $gpu\
        --pred_len $pred_len\
        --lradj   TST
done
# ##################################################

#FEDformer
##################################################
model_name=FEDformer
for random_seed in 2021 2022 2023 2024 2025; do
   python -u ../run_longExp.py \
        --model $model_name \
        --random_seed $random_seed \
        --seq_len $seq_len \
        --label_len 48 \
        --gpu $gpu\
        --pred_len $pred_len\
        --lradj   TST
done
# ##################################################

#DCDN
##################################################
model_name=DCDN
for random_seed in 2021 2022 2023 2024 2025; do
   python -u ../run_longExp.py \
        --model $model_name \
        --random_seed $random_seed \
        --seq_len $seq_len \
        --label_len 48 \
        --gpu $gpu\
        --pred_len $pred_len\
        --lradj   TST
done
# ##################################################