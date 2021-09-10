#!/usr/bin/env python3

import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import argparse
import cProfile
import json
import multiprocessing as mp
import os
import re

import numpy as np
from PIL import Image
from pgpelib import PGPE

from utils import (img2arr, arr2img, rgba2rgb, save_as_png, EasyDict)
from painter import TrianglesPainter
from es import (get_tell_fn, get_best_params_fn, PrintStepHook, PrintCostHook, SaveCostHook, StoreImageHook, ShowImageHook)


def parse_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, default='es_bitmap_out')
    parser.add_argument('--height', type=int, default=200, help='Height of the canvas. -1 for inference.')
    parser.add_argument('--width', type=int, default=-1, help='Width of the canvas.  -1 for inference.')
    parser.add_argument('--target_fn', type=str, required=True)
    parser.add_argument('--n_triangle', type=int, default=50)
    parser.add_argument('--loss_type', type=str, default='l2')
    parser.add_argument('--alpha_scale', type=float, default=0.5)
    parser.add_argument('--coordinate_scale', type=float, default=1.0)
    parser.add_argument('--fps', type=int, default=12)
    parser.add_argument('--n_population', type=int, default=256)
    parser.add_argument('--n_iterations', type=int, default=10000)
    parser.add_argument('--mp_batch_size', type=int, default=1)
    parser.add_argument('--solver', type=str, default='pgpe', choices=['pgpe'])
    parser.add_argument('--report_interval', type=int, default=50)
    parser.add_argument('--step_report_interval', type=int, default=50)
    parser.add_argument('--save_as_gif_interval', type=int, default=50)
    parser.add_argument('--profile', type=bool, default=False)
    cmd_args = parser.parse_args()
    return cmd_args


def parse_args(cmd_args):
    args = EasyDict()

    args.out_dir = cmd_args.out_dir
    args.height = cmd_args.height
    args.width = cmd_args.width
    args.target_fn = cmd_args.target_fn
    args.n_triangle = cmd_args.n_triangle
    args.loss_type = cmd_args.loss_type
    args.alpha_scale = cmd_args.alpha_scale
    args.coordinate_scale = cmd_args.coordinate_scale
    args.fps = cmd_args.fps
    args.n_population = cmd_args.n_population
    args.n_iterations = cmd_args.n_iterations
    args.mp_batch_size = cmd_args.mp_batch_size
    args.solver = cmd_args.solver
    args.report_interval = cmd_args.report_interval
    args.step_report_interval = cmd_args.step_report_interval
    args.save_as_gif_interval = cmd_args.save_as_gif_interval
    args.profile = cmd_args.profile

    return args


def pre_training_loop(args):
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    assert os.path.isdir(out_dir)
    prev_ids = [re.match(r'^\d+', fn) for fn in os.listdir(out_dir)]
    new_id = 1 + max([-1] + [int(id_.group()) if id_ else -1 for id_ in prev_ids])
    desc = f'{os.path.splitext(os.path.basename(args.target_fn))[0]}-' \
           f'{args.n_triangle}-triangles-' \
           f'{args.n_iterations}-iterations-' \
           f'{args.n_population}-population-' \
           f'{args.solver}-solver-' \
           f'{args.loss_type}-loss'
    args.working_dir = os.path.join(out_dir, f'{new_id:04d}-{desc}')

    os.makedirs(args.working_dir)
    args_dump_fn = os.path.join(args.working_dir, 'args.json')
    with open(args_dump_fn, 'w') as f:
        json.dump(args, f, indent=4)


def load_target(fn, resize):
    img = Image.open(fn)
    img = rgba2rgb(img)
    h, w = resize
    img = img.resize((w, h), Image.LANCZOS)
    img_arr = img2arr(img)
    return img_arr


def fitness_fn(params, painter, target_arr, loss_type):
    NUM_ROLLOUTS = 5
    losses = []
    for _ in range(NUM_ROLLOUTS):
        rendered_arr = painter.render(params)
        rendered_arr_rgb = rendered_arr[..., :3]
        rendered_arr_rgb = rendered_arr_rgb.astype(np.float32) / 255.

        target_arr_rgb = target_arr[..., :3]
        target_arr_rgb = target_arr_rgb.astype(np.float32) / 255.

        if loss_type == 'l2':
            pixelwise_l2_loss = (rendered_arr_rgb - target_arr_rgb)**2
            l2_loss = pixelwise_l2_loss.mean()
            loss = l2_loss
        elif loss_type == 'l1':
            pixelwise_l1_loss = np.abs(rendered_arr_rgb - target_arr_rgb)
            l1_loss = pixelwise_l1_loss.mean()
            loss = l1_loss
        else:
            raise ValueError(f'Unsupported loss type \'{loss_type}\'')
        losses.append(loss)

    return -np.mean(losses)  # pgpe *maximizes*


worker_assets = None


def init_worker(painter, target_arr, loss_type):
    global worker_assets
    worker_assets = {'painter': painter, 'target_arr': target_arr, 'loss_type': loss_type}


def fitness_fn_by_worker(params):
    global worker_assets
    painter = worker_assets['painter']
    target_arr = worker_assets['target_arr']
    loss_type = worker_assets['loss_type']

    return fitness_fn(params, painter, target_arr, loss_type)


def batch_fitness_fn_by_workers(params_batch):
    return [fitness_fn_by_worker(params) for params in params_batch]


def infer_height_and_width(hint_height, hint_width, fn):
    fn_width, fn_height = Image.open(fn).size
    if hint_height <= 0:
        if hint_width <= 0:
            inferred_height, inferred_width = fn_height, fn_width  # use target image's size
        else:  # hint_width is valid
            inferred_width = hint_width
            inferred_height = hint_width * fn_height // fn_width
    else:  # hint_height is valid
        if hint_width <= 0:
            inferred_height = hint_height
            inferred_width = hint_height * fn_width // fn_height
        else:  # hint_width is valid
            inferred_height, inferred_width = hint_height, hint_width  # use hint size

    print(f'Inferring height and width. '
          f'Hint: {hint_height, hint_width}, File: {fn_width, fn_height}, Inferred: {inferred_height, inferred_width}')

    return inferred_height, inferred_width


def training_loop(args):
    height, width = infer_height_and_width(args.height, args.width, args.target_fn)

    painter = TrianglesPainter(
        h=height,
        w=width,
        n_triangle=args.n_triangle,
        alpha_scale=args.alpha_scale,
        coordinate_scale=args.coordinate_scale,
    )

    target_arr = load_target(args.target_fn, (height, width))
    save_as_png(os.path.join(args.working_dir, 'target'), arr2img(target_arr))

    hooks = [
        (args.step_report_interval, PrintStepHook()),
        (args.report_interval, PrintCostHook()),
        (args.report_interval, SaveCostHook(save_fp=os.path.join(args.working_dir, 'cost.txt'))),
        (
            args.report_interval,
            StoreImageHook(
                render_fn=lambda params: painter.render(params, background='white'),
                save_fp=os.path.join(args.working_dir, 'animate-background=white'),
                fps=args.fps,
                save_interval=args.save_as_gif_interval,
            ),
        ),
        (args.report_interval, ShowImageHook(render_fn=lambda params: painter.render(params, background='white'))),
    ]

    allowed_solver = ['pgpe']
    if args.solver not in allowed_solver:
        raise ValueError(f'Only following solver(s) is/are supported: {allowed_solver}')

    solver = None
    if args.solver == 'pgpe':
        solver = PGPE(
            solution_length=painter.n_params,
            popsize=args.n_population,
            optimizer='clipup',
            optimizer_config={'max_speed': 0.15},
        )
    else:
        raise ValueError()

    tell_fn = get_tell_fn(args.solver)
    best_params_fn = get_best_params_fn(args.solver)
    loss_type = args.loss_type
    # fitnesses_fn is OK to be inefficient as it's for hook's use only.
    fitnesses_fn = lambda fitness_fn, solutions: [fitness_fn(_, painter, target_arr, loss_type) for _ in solutions]
    n_iterations = args.n_iterations
    mp_batch_size = args.mp_batch_size
    proc_pool = mp.Pool(processes=mp.cpu_count(), initializer=init_worker, initargs=(painter, target_arr, loss_type))

    for i in range(1, 1 + n_iterations):
        solutions = solver.ask()

        batch_it = (solutions[start:start + mp_batch_size] for start in range(0, len(solutions), mp_batch_size))
        batch_output = proc_pool.imap(func=batch_fitness_fn_by_workers, iterable=batch_it)
        fitnesses = [item for batch in batch_output for item in batch]

        tell_fn(solver, solutions, fitnesses)

        for hook in hooks:
            trigger_itervel, hook_fn_or_obj = hook
            if i % trigger_itervel == 0:
                hook_fn_or_obj(i, solver, fitness_fn, fitnesses_fn, best_params_fn)

    for hook in hooks:
        _, hook_fn_or_obj = hook
        if hasattr(hook_fn_or_obj, 'close') and callable(hook_fn_or_obj.close):
            hook_fn_or_obj.close()

    proc_pool.close()
    proc_pool.join()


def main():
    cmd_args = parse_cmd_args()
    args = parse_args(cmd_args)
    pre_training_loop(args)

    if args.profile:
        cProfile.runctx('training_loop(args)', globals(), locals(), sort='cumulative')
    else:
        training_loop(args)


if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()
