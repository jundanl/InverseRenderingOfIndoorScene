CUDA_VISIBLE_DEVICES="6" python3 infer_lighting.py --cuda --dataRoot ./data/ --imList "" \
    --testRoot out/real_model_384X512 --isLight --level 2 \
    --experiment0 models/check_cascade0_w320_h240 --nepoch0 14 \
    --experimentLight0 models/check_cascadeLight0_sg12_offset1.0 --nepochLight0 10 \
    --experimentBS0 models/checkBs_cascade0_w320_h240 \
    --experiment1 models/check_cascade1_w320_h240 --nepoch1 7 \
    --experimentLight1 models/check_cascadeLight1_sg12_offset1.0 --nepochLight1 10 \
    --experimentBS1 models/checkBs_cascade1_w320_h240 \
    --imHeight0 384 --imHeight1 384 --imWidth0 512 --imWidth1 512 \
    --envRow 192 --envCol 256


CUDA_VISIBLE_DEVICES="0" python3 infer_lighting.py --cuda --dataRoot ./data/ --imList "" \
    --testRoot out/real_model_default --isLight --level 2 \
    --experiment0 models/check_cascade0_w320_h240 --nepoch0 14 \
    --experimentLight0 models/check_cascadeLight0_sg12_offset1.0 --nepochLight0 10 \
    --experimentBS0 models/checkBs_cascade0_w320_h240 \
    --experiment1 models/check_cascade1_w320_h240 --nepoch1 7 \
    --experimentLight1 models/check_cascadeLight1_sg12_offset1.0 --nepochLight1 10 \
    --experimentBS1 models/checkBs_cascade1_w320_h240 \
    --envHeight 8 --envWidth 16 \
    --vis_tm_linear_envmap

