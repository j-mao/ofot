from __future__ import annotations

import functools
import itertools
import os
from dataclasses import dataclass
from typing import Generator, Optional

import numpy as np
import torch
from PIL import Image
from scipy import sparse as sps  # type: ignore
from torch import nn
from tqdm.auto import tqdm  # type: ignore

from dataset import get_ground_truth, get_image, get_predicted_flow, list_tasks, write_flow
from visualizer import visualize

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@dataclass
class DomainSpecification:
    tile_shape: tuple[int, int]
    tile_count: tuple[int, int]

    def __iter__(self) -> Generator[tuple[int, int], None, None]:
        yield from itertools.product(*(range(z) for z in self.tile_count))


@dataclass
class OptimalTransportProblem:
    cost: np.ndarray | sps.spmatrix  # (HxW, HxW), any omitted values are inf
    a: np.ndarray                    # (HxW,)
    b: np.ndarray                    # (HxW,)
    coords: np.ndarray               # (HxW, 2)
    domain_coord: tuple[int, int]


class OptimalTransportSolver(nn.Module):
    def __init__(self, *, epsilon: float, lmbda: float, n_iter: int) -> None:
        super().__init__()
        self.epsilon = nn.Parameter(torch.tensor(epsilon))
        self.lmbda   = nn.Parameter(torch.tensor(lmbda))
        self.n_iter  = n_iter

    def forward(self, problem: OptimalTransportProblem) -> torch.Tensor:
        if sps.issparse(problem.cost):
            assert isinstance(problem.cost, sps.spmatrix)
            cost = (
                torch.sparse_coo_tensor(
                    torch.from_numpy(np.vstack((problem.cost.row, problem.cost.col))),
                    problem.cost.data,
                    problem.cost.shape,
                )
                .to(device)
            )
            k = torch.sparse_coo_tensor(
                cost._indices(),
                torch.exp(-cost._values() / self.epsilon),
                cost.size(),
            )
        else:
            cost = torch.from_numpy(problem.cost).to(device)
            k = torch.exp(-cost / self.epsilon)

        a = torch.from_numpy(problem.a).to(device)
        b = torch.from_numpy(problem.b).to(device)
        u = torch.ones_like(a) / a.numel()
        v = torch.ones_like(b) / b.numel()
        p = self.lmbda / (self.lmbda + self.epsilon)
        for i in range(self.n_iter):
            u = torch.pow(a / (k     @ v), p)
            v = torch.pow(b / (k.t() @ u), p)

        coords = torch.from_numpy(problem.coords.astype(float)).to(device)

        if k.is_sparse:
            idxs = k._indices()
            t = torch.sparse_coo_tensor(
                idxs,
                k._values() * u[idxs[0,:]] * v[idxs[1,:]],
                k.size(),
            )
            return torch.sparse.mm(t, coords) / torch.sparse.sum(t, dim=-1).to_dense()[..., None]
        else:
            t = torch.einsum("ij,i,j->ij", k, u, v)
            return t @ coords / t.sum(dim=-1)[..., None]


@functools.lru_cache(maxsize=None)
def make_coords_and_indices(h: int, w: int, max_displacement: int) -> tuple[np.ndarray, np.ndarray]:
    coords = np.stack(np.meshgrid(np.arange(h), np.arange(w)), axis=-1).reshape(-1, 2)
    offsets_1 = np.arange(-max_displacement, max_displacement+1)
    offsets_2 = np.stack(np.meshgrid(offsets_1, offsets_1), axis=-1).reshape(-1, 2)
    indices = coords[:, None, :] + offsets_2[None, :, :]
    indices = np.stack(np.broadcast_arrays(coords[:, None, :], indices), axis=2).reshape(-1, 2, 2)
    indices = indices[
        (indices[:, 1, 0] >= 0) & (indices[:, 1, 0] < h) &
        (indices[:, 1, 1] >= 0) & (indices[:, 1, 1] < w)
    ]
    return coords, indices


@torch.no_grad()
def make_problems(
    img1: Image.Image,
    img2: Image.Image,
    *,
    alpha: float,
    beta: float,
    domain: DomainSpecification,
    max_displacement: int,
    sparse: bool,
) -> Generator[OptimalTransportProblem, None, None]:
    raw1 = np.asarray(img1).copy()
    raw2 = np.asarray(img2).copy()
    for domain_coord in tqdm(list(domain), desc="Generating problems"):
        D = tuple(
            slice(c * s, (c+1) * s)
            for c, s in zip(domain_coord, domain.tile_shape)
        )
        # Do this on CUDA for parallelization. I should use cupy, but I'm lazy.
        data1 = torch.from_numpy(raw1[D]).double().to(device)
        data2 = torch.from_numpy(raw2[D]).double().to(device)
        h, w, _ = data1.shape
        assert data1.shape == data2.shape

        coords, indices_np = make_coords_and_indices(h, w, max_displacement)
        indices = torch.from_numpy(indices_np).to(device)

        px1 = data1[indices[:, 0, 0], indices[:, 0, 1]]
        px2 = data2[indices[:, 1, 0], indices[:, 1, 1]]
        displacement = torch.linalg.vector_norm((indices[:, 0, :] - indices[:, 1, :]).double(), dim=-1)
        colordiff = torch.linalg.vector_norm(px1 - px2, dim=-1)
        costs = torch.sqrt(alpha * displacement**2 + beta * colordiff**2).cpu().numpy()

        coo_i = (indices[:, 0, 0] + indices[:, 0, 1] * h).cpu().numpy()
        coo_j = (indices[:, 1, 0] + indices[:, 1, 1] * h).cpu().numpy()

        if sparse:
            cost = sps.coo_matrix((costs, (coo_i, coo_j)), shape=(h * w, h * w))
        else:
            cost = np.full((h * w, h * w), np.inf)
            cost[coo_i, coo_j] = costs
        yield OptimalTransportProblem(
            cost=cost,
            a=np.ones(h * w) / (h * w),
            b=np.ones(h * w) / (h * w),
            coords=coords,
            domain_coord=domain_coord,
        )


def distance_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(pred - target, dim=-1).mean()


def train(*,
    solver: OptimalTransportSolver,
    optim: torch.optim.Optimizer,
    task: str,
    frame: int,
    alpha: float,
    beta: float,
    domain: DomainSpecification,
    max_displacement: int,
) -> float:
    solver.train()
    img1 = get_image(task=task, frame=frame)
    img2 = get_image(task=task, frame=frame+1)

    problems = make_problems(
        img1=img1,
        img2=img2,
        alpha=alpha,
        beta=beta,
        domain=domain,
        max_displacement=max_displacement,
        sparse=False,
    )

    total_loss = 0.0
    for problem in problems:
        barycenter = solver(problem)
        ground_truth = torch.from_numpy(get_ground_truth(task=task, frame=frame)).to(device)[problem.coords.T]  # type: ignore
        loss = distance_loss(barycenter, ground_truth + torch.from_numpy(problem.coords).to(device))
        optim.zero_grad()
        loss.backward()
        optim.step()
        total_loss += loss.item()
    return total_loss


@torch.no_grad()
def evaluate(*,
    solver: OptimalTransportSolver,
    task: str,
    frame: int,
    alpha: float,
    beta: float,
    domain: DomainSpecification,
    max_displacement: int,
) -> tuple[np.ndarray, np.ndarray]:
    solver.eval()
    img1 = get_image(task=task, frame=frame)
    img2 = get_image(task=task, frame=frame+1)

    problems = make_problems(
        img1=img1,
        img2=img2,
        alpha=alpha,
        beta=beta,
        domain=domain,
        max_displacement=max_displacement,
        sparse=False,
    )

    loss = np.empty(domain.tile_count)
    flow = np.empty(domain.tile_count + domain.tile_shape + (2,))
    for problem in problems:
        barycenter = solver(problem)
        ground_truth = torch.from_numpy(get_ground_truth(task=task, frame=frame)).to(device)[problem.coords.T]  # type: ignore
        loss[problem.domain_coord] = distance_loss(barycenter, ground_truth + torch.from_numpy(problem.coords).to(device)).item()

        flow_raw = barycenter.cpu().numpy() - problem.coords
        flow[(*problem.domain_coord, problem.coords[:,0], problem.coords[:,1])] = flow_raw

    flow = flow.transpose((0, 2, 1, 3, 4))
    flow = flow.reshape((flow.shape[0]*flow.shape[1], flow.shape[2]*flow.shape[3], 2))
    return loss, flow


def run(*,
    solver: OptimalTransportSolver,
    optim: torch.optim.Optimizer,
    n_epoch: int,
    alpha: float,
    beta: float,
    domain: DomainSpecification,
    max_displacement: int,
) -> None:
    for z in range(n_epoch):
        pbar = tqdm(range(20))
        for i in pbar:
            train_loss = train(
                solver=solver,
                optim=optim,
                task="bamboo_1",
                frame=i+1,
                alpha=alpha,
                beta=beta,
                domain=domain,
                max_displacement=max_displacement,
            )
            pbar.set_description(f"Training loss: {train_loss:.4f}")
        test_loss, _ = evaluate(
            solver=solver,
            task="bamboo_1",
            frame=30,
            alpha=alpha,
            beta=beta,
            domain=domain,
            max_displacement=max_displacement,
        )
        print(f"Evaluation loss: {test_loss.mean():.4f}")


def main(
    task: str,
    frame: int,
    alpha: float,
    beta: float,
    epsilon: float,
    lmbda: float,
    save: Optional[str],
) -> None:
    solver = OptimalTransportSolver(n_iter=20, epsilon=epsilon, lmbda=lmbda).to(device)
    domain = DomainSpecification(
        tile_shape=(109, 128),
        tile_count=(4, 8),
    )
    max_displacement = 32

    loss, flow = evaluate(
        solver=solver,
        task=task,
        frame=frame,
        alpha=alpha,
        beta=beta,
        domain=domain,
        max_displacement=max_displacement,
    )
    visualize(flow, save=save)
    visualize(get_ground_truth(task=task, frame=frame), save=None)
    write_flow(flow, task=task, frame=frame, cmt=str(int(lmbda)))
    print(loss.mean())
    print(loss)


if __name__ == "__main__":
    for task, frame in list_tasks():
        print(f"Task {task}/{frame}")
        main(
            task=task,
            frame=frame,
            alpha=1.0,
            beta=1.0,
            epsilon=1.0,
            lmbda=10000.0,
            save=os.path.join("data", f"{task}_{frame:04}_flo_10000.png"),
        )
