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
#from salina_cl.results.evaluation_subspace import *

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
    ds = {}
    #if cfg.final_evaluation:
    #    print("Starting evaluation")
    #    eval = evaluator(cfg.evaluation)
    #    policy_agent = model.policy_agent
    #    policy_agent.agents = policy_agent.agents[1:] #deleting alpha agent
    #    while policy_agent[1].n_anchors > 0:
    #        d = {}
    #        for test_task in scenario.test_tasks():
    #            print("\t- Task",test_task._task_id)
    #            critic_agent = torch.load(os.getcwd()+"/critic_"+str(policy_agent[1].n_anchors-1)+".dat")
    #            d["task_"+str(test_task._task_id)] = eval.evaluate(policy_agent,critic_agent,test_task)
    #        ds[str(policy_agent[1].n_anchors)+"_anchors"] = d
    #        policy_agent = remove_anchor(policy_agent)
    #        print("- set anchor agent to ",policy_agent[1].n_anchors)
    #    print("....done !")
    #    with open(os.getcwd()+"/eval.pkl", "wb") as f:
    #        pickle.dump(ds, f)

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn")

    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        v = torch.ones(1, device="cuda:0")
    _start = time.time()
    main()
    print("time elapsed:",round((time.time()-_start),0),"sec")
