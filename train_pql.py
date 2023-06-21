import isaacgym

assert isaacgym

import time
from collections import deque
from copy import deepcopy
from itertools import count

import hydra
import ray
import torch
from omegaconf import DictConfig

import pql
# from loguru import logger
from pql.algo.pql_actor import PQLActor
from pql.algo.pql_p_learner import PQLPLearner, asyn_p_learner
from pql.algo.pql_v_learner import PQLVLearner, asyn_v_learner
from pql.utils.common import set_random_seed, capture_keyboard_interrupt, preprocess_cfg
from pql.utils.evaluator import Evaluator
from pql.utils.isaacgym_util import create_task_env


@hydra.main(config_path=pql.LIB_PATH.joinpath('cfg').as_posix(), config_name="default")
def main(cfg: DictConfig):
    from ml_logger import logger

    print("Dashboard:", logger.get_dash_url())
    logger.log_text("""
    keys:
    - task.name
    - algo.name
    charts:
    - yKey: "train/critic_loss"
      xKey: step
    - yKey: "train/actor_loss"
      xKey: step
    - yKey: "train/return"
      xKey: step
    """, ".charts.yml", True, True)
    logger.log_params(algo=dict(name=cfg.algo.name), task=dict(name=cfg.task.name))

    ray.init(num_gpus=cfg.algo.num_gpus, num_cpus=cfg.algo.num_cpus, include_dashboard=False)

    preprocess_cfg(cfg)
    capture_keyboard_interrupt()
    set_random_seed(cfg.seed)
    # wandb_run = init_wandb(cfg)
    env = create_task_env(cfg)

    sim_device = torch.device(f"{cfg.sim_device}")
    v_learner_device = torch.device(f"cuda:{cfg.algo.v_learner_gpu}")
    p_learner_device = torch.device(f"cuda:{cfg.algo.p_learner_gpu}")

    pql_actor = PQLActor(env, cfg)
    pql_v_learner = PQLVLearner.remote(env.observation_space.shape, env.action_space.shape[0], cfg)
    pql_p_learner = PQLPLearner.remote(env.observation_space.shape, env.action_space.shape[0], cfg)
    critic, critic_update_times, critic_loss = ray.get(pql_v_learner.start.remote())
    actor, actor_update_times, actor_loss = ray.get(pql_p_learner.start.remote())
    pql_actor.actor = deepcopy(actor).to(sim_device)

    global_steps = 0
    # evaluator = Evaluator(cfg=cfg, wandb_run=wandb_run)
    evaluator = Evaluator(cfg=cfg, wandb_run=None)

    pql_actor.reset_agent()
    logger.print('Warm up the replay buffer', color='green')
    p_data, v_data, steps = pql_actor.explore_env(env, cfg.algo.warm_up, random=True)
    global_steps += steps

    cri_rms = pql_actor.obs_rms.get_states(v_learner_device)
    pol_rms = pql_actor.obs_rms.get_states(p_learner_device)

    old_cri_id = pql_v_learner.update.remote(actor.to(v_learner_device), v_data, cri_rms, 0)
    old_act_id = pql_p_learner.update.remote(critic.to(p_learner_device), p_data, pol_rms, 0)
    asyn_v_learner.remote(pql_v_learner, cfg)
    asyn_p_learner.remote(pql_p_learner, cfg)

    '''measure speed'''
    sim_count = 0
    sim_wait_time = 0
    actor_wait_time = 0
    critic_wait_time = 0

    counter_len = 100
    counter = deque(maxlen=counter_len)
    counter.append({'time': time.time(),
                    'sim': sim_count,
                    'critic': critic_update_times,
                    'actor': actor_update_times,
                    'sim_wait_time': sim_wait_time,
                    'critic_wait_time': critic_wait_time,
                    'actor_wait_time': actor_wait_time})

    logger.print(f"{'Steps':>12s}"
                 f"{'Time':>12s}"
                 f"{'critic_loss':>12s}"
                 f"{'actor_loss':>12s}"
                 f"{'v-updates':>12s}"
                 f"{'p-updates':>12s}", color="cyan")

    for iter_t in count():
        p_data, v_data, steps = pql_actor.explore_env(env, cfg.algo.horizon_len, random=False)
        global_steps += steps
        sim_count += 1

        ready, _ = ray.wait([old_cri_id], timeout=0)
        if len(ready) != 0:
            critic, critic_loss, critic_update_times = ray.get(ready[0])
            old_cri_id = None

        ready, _ = ray.wait([old_act_id], timeout=0)
        if len(ready) != 0:
            actor, actor_loss, actor_update_times = ray.get(ready[0])
            old_act_id = None
            pql_actor.actor = deepcopy(actor).to(sim_device)

        cri_rms = pql_actor.obs_rms.get_states(v_learner_device)
        pol_rms = pql_actor.obs_rms.get_states(p_learner_device)

        cri_id = pql_v_learner.update.remote(
            actor.to(v_learner_device), v_data, cri_rms, critic_wait_time)

        act_id = pql_p_learner.update.remote(
            critic.to(p_learner_device), p_data, pol_rms, actor_wait_time)

        if old_cri_id is None:
            old_cri_id = cri_id

        if old_act_id is None:
            old_act_id = act_id

        if len(counter) >= 10 and actor_update_times != 0 and critic_update_times != 0:
            time_interval = time.time() - counter[0]['time']
            sim_unit_time = time_interval / (sim_count - counter[0]['sim'])
            critic_unit_time = time_interval / (critic_update_times - counter[0]['critic'])
            actor_unit_time = time_interval / (actor_update_times - counter[0]['actor'])

            wait_time = sim_unit_time / cfg.algo.critic_sample_ratio - critic_unit_time
            if wait_time > 0:
                if sim_wait_time == 0:
                    critic_wait_time = counter[0]['critic_wait_time'] + wait_time
                else:
                    sim_wait_time = max(0, counter[0]['sim_wait_time'] - wait_time)
            else:
                if critic_wait_time == 0:
                    sim_wait_time = counter[0]['sim_wait_time'] - wait_time
                else:
                    critic_wait_time = max(0, counter[0]['critic_wait_time'] + wait_time)

            wait_time = critic_unit_time * cfg.algo.critic_actor_ratio - actor_unit_time
            if wait_time > 0:
                actor_wait_time = counter[0]['actor_wait_time'] + wait_time
            else:
                actor_wait_time = max(0, counter[0]['actor_wait_time'] + wait_time)

        counter.append({'time': time.time(),
                        'sim': sim_count,
                        'critic': critic_update_times,
                        'actor': actor_update_times,
                        'sim_wait_time': sim_wait_time,
                        'critic_wait_time': critic_wait_time,
                        'actor_wait_time': actor_wait_time})
        time.sleep(sim_wait_time)

        log_info = {
            "train/critic_loss": critic_loss,
            "train/actor_loss": actor_loss,
            "train/return": pql_actor.return_tracker.mean(),
            "train/episode_length": pql_actor.step_tracker.mean(),
            "train/critic_update_times": critic_update_times,
            "train/actor_update_times": actor_update_times,
            "train/global_steps": global_steps
        }

        if evaluator.parent.poll():
            ret_mean, step_mean = evaluator.parent.recv()
            # note: switch to store_metrics soon.
            logger.store_metrics(**{'eval/return': ret_mean, 'eval/episode_length': step_mean})
        if iter_t % cfg.algo.log_freq == 0:
            # note: consider switching to store_metrics soon.
            logger.log(**log_info, step=global_steps, flush=True)
            # wandb.log(log_info, step=global_steps)
        if iter_t % cfg.algo.eval_freq == 0:
            logger.print(f"{global_steps:12.2e}"
                         f"{time.time() - evaluator.start_time:>12.1f}"
                         f"{log_info['train/critic_loss']:12.2f}"
                         f"{log_info['train/actor_loss']:12.2f}"
                         f"{log_info['train/critic_update_times']:12.2f}"
                         f"{log_info['train/actor_update_times']:12.2f}", color="cyan")
            evaluator.eval_policy(pql_actor.actor, critic, normalizer=pql_actor.obs_rms,
                                  step=global_steps)

        if evaluator.check_if_should_stop(global_steps):
            break


if __name__ == '__main__':
    main()
