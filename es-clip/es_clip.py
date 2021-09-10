#!/usr/bin/env python3

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import argparse
import json
import multiprocessing as mp
import os
import re

import clip
import torch
import numpy as np
from pgpelib import PGPE
import torchvision.transforms as transforms

from utils import EasyDict
from painter import TrianglesPainter
from es import (PrintStepHook, PrintCostHook, SaveCostHook, StoreImageHook, ShowImageHook)


def parse_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, default='es_clip_out')
    parser.add_argument('--height', type=int, default=200)
    parser.add_argument('--width', type=int, default=200)
    parser.add_argument('--n_triangle', type=int, default=50)
    parser.add_argument('--alpha_scale', type=float, default=0.5)
    parser.add_argument('--coordinate_scale', type=float, default=1.0)
    parser.add_argument('--fps', type=int, default=12)
    parser.add_argument('--n_population', type=int, default=256)
    parser.add_argument('--n_iterations', type=int, default=2000)
    parser.add_argument('--report_interval', type=int, default=20)
    parser.add_argument('--step_report_interval', type=int, default=20)
    parser.add_argument('--save_as_gif_interval', type=int, default=20)
    parser.add_argument('--gpus', nargs='*', type=int, default=[])
    parser.add_argument('--thread_per_clip', type=int, default=1)
    parser.add_argument('--prompt', type=str, required=True)

    cmd_args = parser.parse_args()
    return cmd_args


def parse_args(cmd_args):
    args = EasyDict()

    # args copied from cmd_args
    args.out_dir = cmd_args.out_dir
    args.height = cmd_args.height
    args.width = cmd_args.width
    args.n_triangle = cmd_args.n_triangle
    args.alpha_scale = cmd_args.alpha_scale
    args.coordinate_scale = cmd_args.coordinate_scale
    args.fps = cmd_args.fps
    args.n_population = cmd_args.n_population
    args.n_iterations = cmd_args.n_iterations
    args.report_interval = cmd_args.report_interval
    args.step_report_interval = cmd_args.step_report_interval
    args.save_as_gif_interval = cmd_args.save_as_gif_interval
    args.gpus = cmd_args.gpus
    args.thread_per_clip = cmd_args.thread_per_clip
    args.prompt = cmd_args.prompt

    return args


def pre_training_loop(args):
    # Pick working directory.
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    assert os.path.isdir(out_dir)
    prev_ids = [re.match(r'^\d+', fn) for fn in os.listdir(out_dir)]
    new_id = 1 + max([-1] + [int(id_.group()) if id_ else -1 for id_ in prev_ids])
    desc = f'[{args.prompt}]-prompt-' \
           f'{args.n_triangle}-triangles-' \
           f'{args.n_iterations}-iterations-' \
           f'{args.n_population}-population'
    args.working_dir = os.path.join(out_dir, f'{new_id:04d}-{desc}')

    # Prepare working directory.
    os.makedirs(args.working_dir)
    args_dump_fn = os.path.join(args.working_dir, 'args.json')
    with open(args_dump_fn, 'w') as f:
        json.dump(args, f, indent=4)


worker_assets = None


def init_worker(gpu_queue, text, painter):
    global worker_assets
    gpu = gpu_queue.get()
    device_str = f'cuda:{gpu}' if gpu >= 0 else 'cpu'
    device = torch.device(device_str)
    model, preprocess = clip.load('ViT-B/32', device, jit=True)  # change jit=True
    text_input = clip.tokenize(text).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_input)
    worker_assets = {
        'device': device,
        'model': model,
        'preprocess': preprocess,
        'text_features': text_features,
        'painter': painter,
    }
    print(f'Worker initiated with ' f'device {device} ')


def batch_fitness_fn_by_worker(solutions):
    device = worker_assets['device']
    model = worker_assets['model']
    text_features = worker_assets['text_features']
    painter = worker_assets['painter']

    NUM_AUGS = 4

    n_solutions = len(solutions)
    arrs = [painter.render(solution, 'white') for solution in solutions]

    t = np.stack(arrs, axis=0).transpose(0, 3, 1, 2)
    t = torch.tensor(t).to(device)
    t = t.type(torch.float32) / 255.
    t = t.repeat_interleave(NUM_AUGS, dim=0)
    new_augment_trans = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 0.9)),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    t = new_augment_trans(t)
    im_batch = t

    with torch.no_grad():
        image_features = model.encode_image(im_batch)
        similiarities = torch.cosine_similarity(image_features, text_features, axis=-1)
        similiarities = torch.reshape(similiarities, (n_solutions, NUM_AUGS)).mean(axis=-1)

    similiarities = similiarities.to('cpu').tolist()
    return similiarities


def fitness_fn_by_worker(solution):
    solutions = [solution]
    similarities = batch_fitness_fn_by_worker(solutions)
    return similarities[0]


def training_loop(args):
    painter = TrianglesPainter(
        h=args.height,
        w=args.width,
        n_triangle=args.n_triangle,
        alpha_scale=args.alpha_scale,
        coordinate_scale=args.coordinate_scale,
    )

    solver = PGPE(
        solution_length=painter.n_params,
        popsize=args.n_population,
        optimizer='clipup',
        optimizer_config={'max_speed': 0.15},
    )

    hooks = [
        (
            args.step_report_interval,
            PrintStepHook(),
        ),
        (
            args.report_interval,
            PrintCostHook(fitnesses_fn_is_wrapper=False),
        ),
        (
            args.report_interval,
            SaveCostHook(
                fitnesses_fn_is_wrapper=False,
                save_fp=os.path.join(args.working_dir, 'cost.txt'),
            ),
        ),
        (
            args.report_interval,
            StoreImageHook(
                render_fn=lambda params: painter.render(params, background='white'),
                save_fp=os.path.join(args.working_dir, 'animate-background=white'),
                fps=args.fps,
                save_interval=args.save_as_gif_interval,
            ),
        ),
        (
            args.report_interval,
            ShowImageHook(render_fn=lambda params: painter.render(params, background='white')),
        ),
    ]

    n_iterations = args.n_iterations
    if len(args.gpus) > 0:
        n_worker = len(args.gpus) * args.thread_per_clip
        gpu_queue = mp.Queue()
        for _ in range(args.thread_per_clip):
            for gpu in args.gpus:
                gpu_queue.put(gpu)
    else:
        n_worker = args.thread_per_clip
        gpu_queue = mp.Queue()
        for _ in range(args.thread_per_clip):
            gpu_queue.put(-1)
    text = args.prompt
    worker_pool = mp.Pool(
        processes=n_worker,
        initializer=init_worker,
        initargs=(gpu_queue, text, painter),
    )

    hook_fitnesses_fn = lambda solutions: worker_pool.map(fitness_fn_by_worker, solutions)
    hook_best_params_fn = lambda solver: solver.center

    for i in range(1, 1 + n_iterations):
        solutions = solver.ask()

        batch_size = (len(solutions) + n_worker - 1) // n_worker  # = ceil( solutions / n_worker )
        batch_batches = (solutions[start:start + batch_size] for start in range(0, len(solutions), batch_size))
        batch_output = worker_pool.map(batch_fitness_fn_by_worker, batch_batches)
        fitnesses = [item for batch in batch_output for item in batch]

        solver.tell(fitnesses)

        for hook in hooks:
            trigger_itervel, hook_fn_or_obj = hook
            if i % trigger_itervel == 0:
                hook_fn_or_obj(
                    i,
                    solver,
                    fitness_fn=None,
                    fitnesses_fn=hook_fitnesses_fn,
                    best_params_fn=hook_best_params_fn,
                )

    for hook in hooks:
        _, hook_fn_or_obj = hook
        if hasattr(hook_fn_or_obj, 'close') and callable(hook_fn_or_obj.close):
            hook_fn_or_obj.close()

    worker_pool.close()
    worker_pool.join()


def main():
    cmd_args = parse_cmd_args()
    args = parse_args(cmd_args)
    pre_training_loop(args)
    training_loop(args)


if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()
