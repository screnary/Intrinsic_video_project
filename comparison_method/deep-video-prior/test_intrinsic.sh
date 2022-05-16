# test my intrinsic processed data
# # For the video with unimodal inconsistency:
#python main_IRT.py --max_epoch 25 --input PATH_TO_YOUR_INPUT_FOLDER --processed PATH_TO_YOUR_PROCESSED_FOLDER --model NAME_OF_YOUR_MODEL --with_IRT 0 --IRT_initialization 0 --output ./result/OWN_DATA
# # For the video with multimodal inconsistency:
#python main_IRT.py --max_epoch 25 --input PATH_TO_YOUR_INPUT_FOLDER --processed PATH_TO_YOUR_PROCESSED_FOLDER --model NAME_OF_YOUR_MODEL --with_IRT 1 --IRT_initialization 1 --output ./result/OWN_DATA
#python main_IRT.py --max_epoch 25 --input ../../YF-intrinsic/datasets/MPI_refined/refined_gs/input/market_5 --processed ../../YF-intrinsic/datasets/MPI_refined/refined_gs/demo/market_5 --model smooth --with_IRT 1 --IRT_initialization 1 --output ../../YF-intrinsic/datasets/MPI_refined/refined_gs/output/market_5

# python main_IRT.py --max_epoch 50 --input ../../YF-intrinsic/datasets/MPI_refined/refined_gs/input/market_5 --processed ../../YF-intrinsic/datasets/MPI_refined/refined_gs/input/market_5 --model smooth --with_IRT 1 --IRT_initialization 1 --output ../../YF-intrinsic/datasets/MPI_refined/refined_gs/output/market_5
echo market_5
python main_IRT_intrinsic.py --max_epoch 25 --synname market_5 --channel reflect
python main_IRT_intrinsic.py --max_epoch 25 --synname market_5 --channel shading

echo temple_3
python main_IRT_intrinsic.py --max_epoch 25 --synname temple_3 --channel reflect
python main_IRT_intrinsic.py --max_epoch 25 --synname temple_3 --channel shading

echo sleeping_2
python main_IRT_intrinsic.py --max_epoch 25 --synname sleeping_2 --channel reflect
python main_IRT_intrinsic.py --max_epoch 25 --synname sleeping_2 --channel shading

echo cave_4
python main_IRT_intrinsic.py --max_epoch 25 --synname cave_4 --channel reflect
python main_IRT_intrinsic.py --max_epoch 25 --synname cave_4 --channel shading

echo bandage_2
python main_IRT_intrinsic.py --max_epoch 25 --synname bandage_2 --channel reflect
python main_IRT_intrinsic.py --max_epoch 25 --synname bandage_2 --channel shading

echo bamboo_2
python main_IRT_intrinsic.py --max_epoch 25 --synname bamboo_2 --channel reflect
python main_IRT_intrinsic.py --max_epoch 25 --synname bamboo_2 --channel shading

echo alley_2
python main_IRT_intrinsic.py --max_epoch 25 --synname alley_2 --channel reflect
python main_IRT_intrinsic.py --max_epoch 25 --synname alley_2 --channel shading

echo mountain_1
python main_IRT_intrinsic.py --max_epoch 25 --synname mountain_1 --channel reflect
python main_IRT_intrinsic.py --max_epoch 25 --synname mountain_1 --channel shading
