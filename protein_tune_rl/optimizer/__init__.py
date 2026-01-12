def create_optimizer(name, model, **kwargs):
    if name == "reinforce":
        from protein_tune_rl.optimizer.reinforce import Reinforce

        return Reinforce(policy=model, **kwargs)
    elif name == "ppo":
        from protein_tune_rl.optimizer.ppo import PPO

        return PPO(policy=model, **kwargs)
    elif name == "a2c":
        from protein_tune_rl.optimizer.a2c import A2C

        return A2C(policy=model, **kwargs)
