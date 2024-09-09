CUDA_VISIBLE_DEVICES="0" python3 testReal.py --cuda --dataRoot ./data/iiw-dataset/data --imList IIWTest.txt \
    --testRoot IIWTest --isLight --isBS --level 2 \
    --experiment0 experiments/models/check_cascadeNYU0 --nepoch0 2 \
    --experimentLight0 experiments/models/check_cascadeLight0_sg12_offset1.0 --nepochLight0 10 \
    --experimentBS0 experiments/models/checkBs_cascade0_w320_h240 \
    --experiment1 experiments/models/check_cascadeNYU1 --nepoch1 3 \
    --experimentLight1 experiments/models/check_cascadeLight1_sg12_offset1.0 --nepochLight1 10 \
    --experimentBS1 experiments/models/checkBs_cascade1_w320_h240 \



CUDA_VISIBLE_DEVICES="1" python3 testReal.py --cuda --dataRoot ./data/paper_comparison/internet --imList "" \
    --testRoot out --isLight --isBS --level 2 \
    --experiment0 experiments/models/check_cascadeNYU0 --nepoch0 2 \
    --experimentLight0 experiments/models/check_cascadeLight0_sg12_offset1.0 --nepochLight0 10 \
    --experimentBS0 experiments/models/checkBs_cascade0_w320_h240 \
    --experiment1 experiments/models/check_cascadeNYU1 --nepoch1 3 \
    --experimentLight1 experiments/models/check_cascadeLight1_sg12_offset1.0 --nepochLight1 10 \
    --experimentBS1 experiments/models/checkBs_cascade1_w320_h240 \
    --imHeight0 1024 --imHeight1 1024 --imWidth0 1024 --imWidth1 1024 \
    --envRow 512 --envCol 512
