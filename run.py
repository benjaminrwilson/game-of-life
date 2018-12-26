import argparse

import cv2
import torch
from torch import ByteTensor, Tensor
from torch.nn import Conv2d, Parameter
from torch.nn.init import zeros_


def step(state, get_neighbors):
    # Get neighbor counts of cells
    neighbors = get_neighbors(state)[0, ...]

    # Alive cell with less than two neighbors should die
    rule1 = (neighbors < 2).type(Tensor)
    mask1 = (rule1 * state[0, ...]).type(ByteTensor)

    # Alive cell with more than two neighbors should die
    rule2 = (neighbors > 3).type(Tensor)
    mask2 = (rule2 * state[0, ...]).type(ByteTensor)

    # Dead cell with exactly three neighbors should spawn
    rule3 = (neighbors == 3).type(Tensor)
    mask3 = (rule3 * (1 - state[0, ...])).type(ByteTensor)

    # Update state
    state[0, mask1] = 0
    state[0, mask2] = 0
    state[0, mask3] = 1
    return state


def run_world(size, prob, tick_ratio, device):
    step_count = 0
    with torch.no_grad():
        channels = 1
        state = init_world(size, channels, prob).to(device)
        get_neighbors = get_neighbors_map(channels).to(device)
        while True:
            if should_step(step_count, tick_ratio):
                cv2.imshow("Game of Life", state.numpy())
                state = step(image2state(state), get_neighbors)
                state = state2image(state)
            step_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


def init_world(size, channels, prob):
    return torch.distributions.Bernoulli(
        Tensor([prob])).sample(torch.Size([size, size, channels])).squeeze(-1)


def image2state(image):
    return image.permute(2, 0, 1).unsqueeze(0)


def state2image(state):
    return state.squeeze(0).permute(1, 2, 0)


def get_neighbors_map(d):
    neighbors_filter = Conv2d(d, d, 3, padding=1)
    neighbors_filter.weight = Parameter(Tensor([[[[1, 1, 1],
                                                  [1, 0, 1],
                                                  [1, 1, 1]]]]), requires_grad=False)
    neighbors_filter.bias = zeros_(neighbors_filter.bias)
    return neighbors_filter


def should_step(step_count, tick_ratio):
    return step_count % tick_ratio == 0


def main():
    opts = argparse.ArgumentParser(description='Game of Life')
    opts.add_argument(
        '-s',
        '--size',
        help='Size of world grid',
        default=500)
    opts.add_argument(
        '-p',
        '--prob',
        help='Probability of life in the initial seed',
        default=.15)
    opts.add_argument(
        '-tr',
        '--tick_ratio',
        help='Ticks needed to update on time step in game',
        default=1)
    opts = opts.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_world(opts.size, opts.prob, opts.tick_ratio, device)


if __name__ == "__main__":
    main()
