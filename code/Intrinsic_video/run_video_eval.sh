echo lambda_r=1, lambda_s=1
python main_mpi_video.py --phase test --best_ep 250 --w_flow 50 --lambda_r 1 --lambda_s 1
echo lambda_r=4, lambda_s=0
python main_mpi_video.py --phase test --best_ep 250 --w_flow 50 --lambda_r 4 --lambda_s 0