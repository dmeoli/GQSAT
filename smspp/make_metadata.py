"""Writes the METADATA of each problem directory given (colon-separated) for
SMSppMaxSATEnv: the steps the rule of the cores of SMS++ takes on each
problem, with the incumbent at the end checked against the optimum of an
optima.txt, if the directory has one."""
import sys
from os import listdir
from os.path import join

sys.path.insert(0, __file__.rsplit("/", 1)[0] + "/build")
import _smspp_env  # noqa: E402

max_iter = int(sys.argv[2]) if len(sys.argv) > 2 else 20
for d in sys.argv[1].split(":"):
    optima = {}
    try:
        for line in open(join(d, "optima.txt")):
            if line.strip() and not line.startswith("#"):
                optima[line.split()[0]] = float(line.split()[1])
    except OSError:
        pass
    env = _smspp_env.BnBEnv("CaDiCaLSATSolver", max_iter, 1, 0.1)
    with open(join(d, "METADATA"), "w") as out:
        for f in sorted(listdir(d)):
            if not (f.endswith(".wcnf") or f.endswith(".cnf")):
                continue
            _, done = env.reset(join(d, f))
            steps = 0
            while not done:
                _, _, done = env.step(-1)
                steps += 1
            if f in optima:
                assert abs(env.incumbent() - optima[f]) < 1e-6, f
            out.write(f"{f},{steps},{steps}\n")
            print(f, steps, env.nodes(), env.incumbent(), flush=True)
