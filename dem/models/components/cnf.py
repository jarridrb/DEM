import torch
from torch.func import jacrev
from torchdiffeq import odeint


def exact_div_fn(u):
    """Accepts a function u:R^D -> R^D."""
    J = jacrev(u)

    def div(x):
        return torch.trace(J(x.unsqueeze(0)).squeeze())

    return div


def hutch_div_fn(u):
    """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""

    def div_fn(x):
        epsilon = torch.randn_like(x)
        with torch.enable_grad():
            fn_eps = torch.sum(u(x) * epsilon)
        grad_fn_eps = torch.autograd.grad(fn_eps, x, create_graph=True)[0]
        return torch.sum(grad_fn_eps * epsilon, dim=tuple(range(1, len(x.shape))))

    return div_fn


def hutch_trace(x_in, x_out, noise=None):
    jvp = torch.autograd.grad(x_out, x_in, noise, create_graph=True)[0]
    return torch.einsum("bi,bi->b", (jvp, noise))


def hutchinson_divergence(self, f, x, dist="gaussian", grad_outputs=None):
    """
    inputs:
        f (N x D)
        x (N x D+1)

    output:
        div (N)
    """
    N = f.shape[0]
    D = f.shape[1]
    div = torch.zeros(N).to(f)
    assert dist in ["rademacher", "gaussian"], "invalid hutchinson distribution"
    if grad_outputs is None:
        if dist == "rademacher":
            grad_outputs = (torch.randn_like(f) < 0) * 2.0 - 1.0
        elif dist == "gaussian":
            grad_outputs = torch.randn_like(f)
    f_reduced = torch.einsum("ij,ij->i", f, grad_outputs)
    grad = self.gradient(f_reduced, x)
    div = torch.einsum("ij,ij->i", grad, grad_outputs)
    return div


class CNF(torch.nn.Module):
    def __init__(
        self,
        vf,
        is_diffusion,
        use_exact_likelihood=True,
        noise_schedule=None,
        method="dopri5",
        atol=1e-5,
        rtol=1e-5,
        max_steps_till_fallback=3000,
        num_steps=100,
    ):
        super().__init__()

        self.vf = vf
        self.is_diffusion = is_diffusion
        self.use_exact_likelihood = use_exact_likelihood
        self.nfe = 0.0
        self.noise_schedule = noise_schedule
        self.method = method
        self.atol = atol
        self.rtol = rtol
        self.max_steps_till_fallback = max_steps_till_fallback
        self.num_steps = num_steps
        if method == "dopri5":
            self.num_steps = 1

    def forward(self, t, x):
        if self.nfe > self.max_steps_till_fallback:
            raise RuntimeError("Too many integration steps")
        # if (self.nfe > 50) & (self.nfe % 100 == 0):
        #    print(f"Large NFE: {self.nfe}")
        x = x[..., :-1].clone().detach().requires_grad_(True)

        def vecfield(x):
            # PF ODE requires dividing the reverse (VE) drift by 2.
            # If we use VP (or a different noising SDE) we need the
            # forward drift too.
            shaped_t = torch.ones(x.shape[0], device=x.device) * t
            if self.is_diffusion:
                return 0.5 * self.vf(shaped_t, x) * self.noise_schedule.g(t) ** 2
            else:
                return self.vf(shaped_t, x)

        if self.use_exact_likelihood:
            dx = vecfield(x)
            div_fn = exact_div_fn
            div = torch.vmap(div_fn(vecfield), randomness="different")(x)
        else:
            with torch.enable_grad():
                dx = vecfield(x)
            div = hutch_trace(x, dx, torch.randn_like(x))

        self.nfe += 1
        # print(div.mean())
        return torch.cat([dx.detach(), div[:, None].detach()], dim=-1)

    def integrate(self, x):
        method = self.method
        end_time = int(self.is_diffusion)
        start_time = 1.0 - end_time

        time = torch.linspace(start_time, end_time, self.num_steps + 1, device=x.device)
        try:
            return odeint(self, x, t=time, method=method, atol=self.atol, rtol=self.rtol)

        except (RuntimeError, AssertionError) as e:
            print(e)
            print("Falling back on fixed-step integration")
            self.nfe = 0.0
            time = torch.linspace(
                start_time, end_time, self.max_steps_till_fallback + 1, device=x.device
            )
            return odeint(self, x, t=time, method="euler")

    def generate(self, x):
        method = self.method

        def reverse_wrapper(model):
            def fxn(t, x, args=None):
                if t.ndim == 0:
                    t = t.unsqueeze(0)

                return model(t.repeat(len(x)), x)

            return fxn

        end_time = 1 - int(self.is_diffusion)
        start_time = 1.0 - end_time

        time = torch.linspace(start_time, end_time, self.num_steps + 1, device=x.device)

        try:
            return odeint(
                reverse_wrapper(self.vf),
                x,
                t=time,
                method=method,
                atol=self.atol,
                rtol=self.rtol,
            )

        except (RuntimeError, AssertionError) as e:
            print(e)
            print("Falling back on fixed-step integration")
            self.nfe = 0.0
            time = torch.linspace(
                start_time, end_time, self.max_steps_till_fallback + 1, device=x.device
            )
            return odeint(reverse_wrapper(self.vf), x, t=time, method="euler")
