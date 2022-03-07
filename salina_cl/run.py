#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import hydra
from salina import instantiate_class
import torch
import time
import os
from os.path import exists
import numpy as np
import pickle

@hydra.main(config_path="configs/", config_name="ppo_finetune_cartpole.yaml")
def main(cfg):
    if not exists(cfg.perf_path):
        os.makedirs(os.path.dirname(cfg.perf_path), exist_ok=True)
        with open(cfg.perf_path, "wb") as f:
            pickle.dump({},f)
    logger = instantiate_class(cfg.logger)
    logger.save_hps(cfg, verbose =False)
    model = instantiate_class(cfg.model)
    scenario = instantiate_class(cfg.scenario)
    logger_evaluation = logger.get_logger("evaluation/")
    logger_evaluation.logger.modulo = 1
    stage=0
    for train_task in scenario.train_tasks():
        model.train(train_task,logger)
        evaluation = model.evaluate(scenario.test_tasks(),logger_evaluation)
        for tid in evaluation:
            for k,v in evaluation[tid].items():
                logger_evaluation.add_scalar(str(tid)+"/"+k,v,stage)
        m_size = model.memory_size()
        for k,v in m_size.items():
            logger_evaluation.add_scalar("memory/"+k,v,stage)
        stage+=1
    if cfg.save_model:
        torch.save(model.policy_agent,os.getcwd()+"/policy.dat")
    perf = np.mean([v['avg_reward'] for v in evaluation.values()])
    d_perf = {os.getcwd():perf}
    with open(cfg.perf_path, "rb") as f:
        data = pickle.load(f)
        data.update(d_perf)
    with open(cfg.perf_path, "wb") as f:
        pickle.dump(data,f)

    logger.close()
    print("....done !")

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn")

    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        v = torch.ones(1, device="cuda:0")
    _start = time.time()
    main()
    print("time elapsed:",round((time.time()-_start),0),"sec")
