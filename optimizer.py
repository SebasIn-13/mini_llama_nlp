from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
            max_grad_norm: float = None,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias, max_grad_norm=max_grad_norm)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            # DONE: Clip gradients if max_grad_norm is set
            if group['max_grad_norm'] is not None:
                torch.nn.utils.clip_grad_norm_(group['params'], group['max_grad_norm'])
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary
                state = self.state[p]

                # Initialize state dictionary
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                # DONE: Access hyperparameters from the `group` dictionary
                lr = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]
                correct_bias = group["correct_bias"]
                
                
                # TODO: Update first and second moments of the gradients
                state["step"] += 1
                # Se utiliza el paso 7 y 8 del Algoritmo 2
                state["exp_avg"] = beta1 * state["exp_avg"] + (1 - beta1) * grad
                state["exp_avg_sq"] = beta2 * state["exp_avg_sq"] + (1 - beta2) * (grad * grad)
                

                # TODO: Bias correction
                # Please note that we are using the "efficient version" given in Algorithm 2 
                # https://arxiv.org/pdf/1711.05101

                # Se obtiene un learning rate equivalente a la modificación del bias
                if correct_bias:
                    # Se aplican los pasos 9 y 10 del Algoritmo 2
                    mt=state["exp_avg"]/(1 - beta1**state["step"])
                    vt=state["exp_avg_sq"]/(1 - beta2**state["step"])
                else:
                    mt=state["exp_avg"]
                    vt=state["exp_avg_sq"]

                # DONE: Update parameters
                # Se aplica el paso 12 del Algoritmo 2
                p.data -= (lr*mt / (vt**0.5+eps))

                # DONE: Add weight decay after the main gradient-based updates.
                # Se utiliza el learning rate en lugar del Scheduled Optimizer sugerido en el paper
                if weight_decay != 0:
                    p.data = p.data - lr * weight_decay * p.data

        return loss