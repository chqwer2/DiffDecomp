



python  train.py --time_steps 50 --train_steps 700000 \
            --save_folder ./results_cifar10 \
            --data_path ../deblurring-diffusion-pytorch/root_celebA_128_train_new/ \
            --train_routine Final --sampling_routine default \
            --remove_time_embed --residual --loss_type l1 \
            --initial_mask 11 --kernel_std 0.15 --reverse



