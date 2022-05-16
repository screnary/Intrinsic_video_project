echo train from v8 pre_trained
python main_mpi_video.py --w_flow 50 --lambda_r 10 --lambda_s 5 --is_continue --from_ep 200 --save_per_n_ep 1 --total_ep 5 --save_train_img
echo test from v8 pre_trained
python main_mpi_video.py --w_flow 50 --lambda_r 10 --lambda_s 5 --phase test --best_ep 5
