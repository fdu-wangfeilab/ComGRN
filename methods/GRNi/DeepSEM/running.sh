echo "denoised data"

for loop in 0 1 2 3 4 5 6 7 8 9
do
    python data_transfer.py \
    --data_path ./demo_data/GRN_inference/input/real_data/mHSC/2000_mHSC/

    python main.py \
    --task non_celltype_GRN \
    --setting test \
    --alpha 100 \
    --beta 1 \
    --data_file ./demo_data/GRN_inference/input/real_data/mHSC/2000_mHSC/denoised_ExpressionDataT.csv \
    --net_file ./demo_data/GRN_inference/input/real_data/mHSC/2000_mHSC/network.csv \
    --save_name ./demo_data/GRN_inference/output \

    python eval.py \
    --ref_path ./demo_data/GRN_inference/input/real_data/mHSC/2000_mHSC/ \
    --pred_path ./demo_data/GRN_inference/output/GRN_inference_result.tsv

done

for loop in 0 # 1 2 3 4 5 6 7 8 9
do
    python data_transfer.py \
    --data_path ./demo_data/GRN_inference/input/real_data/mHSC/2000_mHSC/denoised_

    python main.py \
    --task celltype_GRN \
    --setting test \
    --alpha 10 \
    --beta 0.1 \
    --data_file ./demo_data/GRN_inference/input/real_data/mHSC/2000_mHSC/denoised_ExpressionDataT.csv \
    --net_file ./demo_data/GRN_inference/input/real_data/mHSC/2000_mHSC/network.csv \
    --save_name ./demo_data/GRN_inference/output \

    python eval.py \
    --ref_path ./demo_data/GRN_inference/input/real_data/mHSC/2000_mHSC/ \
    --pred_path ./demo_data/GRN_inference/output/GRN_inference_result.tsv

done