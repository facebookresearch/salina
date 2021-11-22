#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import torch


def _index(tensor_3d, tensor_2d):
    x, y, z = tensor_3d.size()
    t = tensor_3d.reshape(x * y, z)
    tt = tensor_2d.reshape(x * y)
    v = t[torch.arange(x * y), tt]
    v = v.reshape(x, y)
    return v


def cumulated_reward(reward, done):
    T, B = done.size()
    done = done.detach().clone()

    v_done, index_done = done.float().max(0)
    assert v_done.eq(
        1.0
    ).all(), "[agents.rl.functional.cumulated_reward] Computing cumulated reward over unfinished trajectories"
    arange = torch.arange(T, device=done.device).unsqueeze(-1).repeat(1, B)
    index_done = index_done.unsqueeze(0).repeat(T, 1)

    mask = arange.le(index_done)
    reward = (reward * mask.float()).sum(0)
    return reward.mean().item()


def temporal_difference(critic, reward, done, discount_factor):
    target = (
        discount_factor * critic[1:].detach() * (1.0 - done[1:].float()) + reward[1:]
    )
    td = target - critic[:-1]
    to_add = torch.zeros(1, td.size()[1]).to(td.device)
    td = torch.cat([td, to_add], dim=0)
    return td


def doubleqlearning_temporal_difference(
    q, action, q_target, reward, done, discount_factor
):
    action_max = q.max(-1)[1]
    q_target_max = _index(q_target, action_max).detach()[1:]

    done = done.float()
    target = reward[1:] + discount_factor * q_target_max * (1 - done[1:])

    q = _index(q, action)[:-1]
    td = target - q
    to_add = torch.zeros(1, td.size()[1], device=td.device)
    td = torch.cat([td, to_add], dim=0)
    return td


def gae(critic, reward, done, discount_factor, gae_coef):
    r = reward[1:]
    v = critic[1:].detach()
    d = done.float()
    td = r + discount_factor * (1.0 - d[1:]) * v - critic[:-1]
    # handling td0 case
    if gae_coef == 0.0:
        return td

    T = td.shape[0]
    gae = td[-1]
    gaes = [gae]
    for t in range(T - 2, -1, -1):
        gae = td[t] + discount_factor * gae_coef * (1.0 - d[:-1][t]) * gae
        gaes.append(gae)
    gaes = list([g.unsqueeze(0) for g in reversed(gaes)])
    gaes = torch.cat(gaes, dim=0)
    return gaes


def compute_reinforce_loss(
    reward, action_probabilities, baseline, action, done, discount_factor
):

    batch_size = reward.size()[1]

    # Find the first done occurence fir each element in the batch
    v_done, trajectories_length = done.float().max(0)
    trajectories_length += 1
    assert v_done.eq(1.0).all()
    max_trajectories_length = trajectories_length.max().item()
    # Shorten trajectories for faster computation
    reward = reward[:max_trajectories_length]
    action_probabilities = action_probabilities[:max_trajectories_length]
    baseline = baseline[:max_trajectories_length]
    action = action[:max_trajectories_length]

    # Create a binary mask to mask useless values (of size max_trajectories_length x batch_size)
    arange = (
        torch.arange(max_trajectories_length, device=done.device)
        .unsqueeze(-1)
        .repeat(1, batch_size)
    )
    mask = arange.lt(
        trajectories_length.unsqueeze(0).repeat(max_trajectories_length, 1)
    )
    reward = reward * mask

    # Compute discounted cumulated reward
    cumulated_reward = [torch.zeros_like(reward[-1])]
    for t in range(max_trajectories_length - 1, 0, -1):
        cumulated_reward.append(discount_factor + cumulated_reward[-1] + reward[t])
    cumulated_reward.reverse()
    cumulated_reward = torch.cat([c.unsqueeze(0) for c in cumulated_reward])

    # baseline loss
    g = baseline - cumulated_reward
    baseline_loss = (g) ** 2
    baseline_loss = (baseline_loss * mask).mean()

    # policy loss
    log_probabilities = _index(action_probabilities, action).log()
    policy_loss = log_probabilities * -g.detach()
    policy_loss = policy_loss * mask
    policy_loss = policy_loss.mean()

    # entropy loss
    entropy = torch.distributions.Categorical(action_probabilities).entropy() * mask
    entropy_loss = entropy.mean()

    return {
        "baseline_loss": baseline_loss,
        "policy_loss": policy_loss,
        "entropy_loss": entropy_loss,
    }


# def soft_update_params(net, target_net, tau):
#     for param, target_param in zip(net.parameters(), target_net.parameters()):
#         target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
