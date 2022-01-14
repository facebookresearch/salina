import hydra
from omegaconf import DictConfig, OmegaConf
from salina import instantiate_class
import torch

@hydra.main(config_path=".", config_name="pendulum.yaml")
def main(cfg):
    logger = instantiate_class(cfg.logger)
    logger.save_hps(cfg)
    model = instantiate_class(cfg.model)
    scenario = instantiate_class(cfg.scenario)
    logger_evaluation=logger.get_logger("evaluation/")
    stage=0
    for train_task in scenario.train_tasks():
        model.train(train_task,logger)
        evaluation=model.evaluate(scenario.test_tasks(),logger_evaluation)
        print(evaluation)
        for tid in evaluation:
            for k,v in evaluation[tid].items():
                logger_evaluation.add_scalar(str(tid)+"/"+k,v,stage)
        stage+=1
    print("....done !")

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn")

    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        v = torch.ones(1, device="cuda:0")

    main()
