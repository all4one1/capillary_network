# capillary_network


Instructions for SLURM:
1) Compilation: 
nvcc kernel.cu --generate-code arch=compute_35,code=compute_35 -w -O2 -o run
2) run:
sbatch -N 1 --gres=gpu:1 -C k40m -t 1200 --wrap='./run'

maybe -std=c++11 needed 
