#!/bin/bash

for i in {1}
do
	nohup python3 train_grid.py apple_world_simple -n 4 -e 8 -t 1 --sigma_init 1.0 --optimizer=CMAES --peak=1.0 > p.1.00.out &
	sleep .1
	nohup python3 train_grid.py apple_world_simple -n 4 -e 8 -t 1 --sigma_init 1.0 --optimizer=CMAES --peak=0.9 > p.0.90.out &
	sleep .1
	nohup python3 train_grid.py apple_world_simple -n 4 -e 8 -t 1 --sigma_init 1.0 --optimizer=CMAES --peak=0.8 > p.0.80.out &
	sleep .1
	nohup python3 train_grid.py apple_world_simple -n 4 -e 8 -t 1 --sigma_init 1.0 --optimizer=CMAES --peak=0.7 > p.0.70.out &
	sleep .1
	nohup python3 train_grid.py apple_world_simple -n 4 -e 8 -t 1 --sigma_init 1.0 --optimizer=CMAES --peak=0.6 > p.0.60.out &
	sleep .1
	nohup python3 train_grid.py apple_world_simple -n 4 -e 8 -t 1 --sigma_init 1.0 --optimizer=CMAES --peak=0.5 > p.0.50.out &
	sleep .1
	nohup python3 train_grid.py apple_world_simple -n 4 -e 8 -t 1 --sigma_init 1.0 --optimizer=CMAES --peak=0.4 > p.0.40.out &
	sleep .1
	nohup python3 train_grid.py apple_world_simple -n 4 -e 8 -t 1 --sigma_init 1.0 --optimizer=CMAES --peak=0.3 > p.0.30.out &
	sleep .1
	nohup python3 train_grid.py apple_world_simple -n 4 -e 8 -t 1 --sigma_init 1.0 --optimizer=CMAES --peak=0.2 > p.0.20.out &
	sleep .1
	nohup python3 train_grid.py apple_world_simple -n 4 -e 8 -t 1 --sigma_init 1.0 --optimizer=CMAES --peak=0.1 > p.0.10.out &
	sleep .1
	nohup python3 train_grid.py apple_world_simple -n 4 -e 8 -t 1 --sigma_init 1.0 --optimizer=CMAES --peak=0.0 > p.0.00.out &
	sleep 2120
done
