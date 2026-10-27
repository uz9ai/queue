# Team 7 Slurm cluster installer

Sets up a Slurm cluster on bare EC2 instances from one file, `cluster.toml`: one CPU controller (`ctl-01`) and GPU nodes (`train-01` for training, `eval-01` for evaluation). The same file is installed on every host at `/etc/team7/cluster.toml`, so training code reads the same values Slurm was configured with (requeue exit code, signal lead time, checkpoint paths).

The design behind the settings is in `../queue_design.md`. Every TOML key is documented in [DOCS.md](DOCS.md).

## Contents

```
cluster/
  install.sh               entry point (runs the installer with the system python3)
  cluster.example.toml     documented template; copy to cluster.toml
  installer/               installer, standard library only
  team7config/             TOML loader + validation, used by the installer AND training code
  pyenv/                   training environment: pyproject.toml + uv.lock (torch, DeepSpeed)
  tests/                   test_limits (T01-T11), test_requeue (T12, T12b), collect_evidence
  README.md, DOCS.md
```

## What gets installed where

| Path | Hosts | What |
|---|---|---|
| `/etc/team7/cluster.toml` (+ `.sha256`) | all | the config used for this install |
| `/etc/team7/team7.env`, `/etc/profile.d/team7.sh` | all | `TEAM7_CONFIG`, `CUDA_HOME`, `TORCH_EXTENSIONS_DIR`, ... |
| `/usr/local/bin/team7-config` | all | `team7-config get slurm.requeue_exit` for shell scripts |
| `/opt/team7/cluster/` | all | copy of this directory (tests, team7config, pyenv) |
| `/opt/team7/src/` | all | your code, if `[code].source` is `local` or `git` |
| `/opt/team7/jobs/{train,eval,debug}.sbatch` | all | job templates generated from the TOML |
| `/etc/slurm/{slurm,gres,cgroup,slurmdbd}.conf` | controller | Slurm config (nodes fetch it: configless mode) |
| `/root/team7/munge.key` | controller | team munge key (mode 0400), for copying to nodes |
| `/etc/team7/secrets/db_password` | controller | MariaDB password for slurmdbd, never leaves the host |
| `/opt/team7/venv/` | GPU nodes | Python venv with torch and DeepSpeed |
| `/opt/team7/torch_extensions/` | GPU nodes | pre-built DeepSpeed ops |
| `/scratch` (`/scratch/ckpt`, `/scratch/logs`) | GPU nodes | instance-store NVMe, mounted at every boot |
| `/opt/team7/slurm/{prolog,healthcheck}.sh` | GPU nodes | only when `[health].enabled = true` |

## 1. AWS setup (once)

**Instances**

| Host | AMI | Type | Root volume |
|---|---|---|---|
| `ctl-01` | Ubuntu Server 24.04 LTS | `t3.medium` or larger | 30 GB |
| `train-01`, `eval-01` | Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04) | `g4dn.xlarge` or `g4dn.2xlarge` | 100 GB (the venv and uv cache take ~15 GB) |

All hosts must run the same Ubuntu release, so all get the same Slurm version from apt (23.11 on 24.04).

**Fixed private IPs.** Launch each instance with the private IP from `cluster.toml` (console: *Network settings → Advanced → Primary IP*; CLI: `--private-ip-address 10.0.1.20`). The IPs must be inside the subnet's CIDR. With fixed IPs, a replaced instance needs no config change.

**Security group.** Put all three instances in one security group and allow:

| Port | From | Why |
|---|---|---|
| TCP 6817 | the group | slurmctld |
| TCP 6818 | the group | slurmd |
| TCP 60001-63000 | the group | `srun` callbacks (`[slurm].srun_port_range`) |
| TCP 22 | your IP | SSH |
| TCP 2049 | the group, on the EFS security group | only with `[storage.shared].type = "efs"` |

Allowing all traffic from the group itself is simpler and covers the first three rows.

**IAM instance role** (only if you use these):
- `munge_key_source = "ssm"`: `ssm:GetParameter` on the parameter for all hosts, plus `ssm:PutParameter` for the controller. The AWS-managed KMS key for SSM works by default.
- `checkpoints.durable = "s3"`: read/write on the bucket prefix for `train-01`, read for `eval-01`.

## 2. Configure

```bash
cd cluster
cp cluster.example.toml cluster.toml
$EDITOR cluster.toml          # IPs, node sizes, users and their SSH public keys, S3 URI
./install.sh validate
./install.sh render           # writes everything that would be installed to ./out/ for review
```

Check the node sizes. `cpus`, `sockets`, `cores_per_socket`, `threads_per_core` and `real_memory_mb` must match `slurmd -C` on the instance type. The example has values for `g4dn.xlarge`. If they're wrong, the node install stops and prints the right values.

Commit `cluster.toml`. It holds no secrets.

## 3. Install

Copy the repository (the directory that contains `cluster/`) to each host, either with `git clone` at the same commit or with `rsync -a ./ ubuntu@<host>:team7/`. Run every command from the `cluster/` directory.

**Controller first:**

```bash
ssh ubuntu@<controller>
cd team7/cluster
sudo ./install.sh controller
```

This takes a few minutes. It ends by printing the commands for the next step and the sha256 of `cluster.toml`.

**Copy the munge key** (only with `munge_key_source = "file"`), from your laptop:

```bash
ssh ubuntu@<controller> 'sudo cat /root/team7/munge.key' > munge.key
scp munge.key ubuntu@<train-01>:/tmp/munge.key
scp munge.key ubuntu@<eval-01>:/tmp/munge.key
rm munge.key
```

**Each GPU node** (both can run at the same time):

```bash
ssh ubuntu@<train-01>
cd team7/cluster
sudo ./install.sh node train-01 --munge-key /tmp/munge.key
```

Then run the same on `eval-01` with `node eval-01`. The node install deletes `/tmp/munge.key` after installing it. It takes longer than the controller, mostly for the torch download and the DeepSpeed op build. The training venv is built before `slurmd` starts, so no job lands on a node without its environment.

**Verify, on the controller:**

```bash
sudo ./install.sh verify
```

This runs a 1-GPU job on each node as a cluster user and checks that each node's `/etc/team7/cluster.toml` matches the controller's.

The installer refuses to run on the wrong host. `controller` checks that this host's IP is `[controller].ip`. `node train-01` checks the IP, the GPU count and the GPU model.

## 4. Commands

```
sudo ./install.sh controller                  set up the controller
sudo ./install.sh node <name> [--munge-key F] set up a GPU node
sudo ./install.sh apply                       push cluster.toml changes (controller)
sudo ./install.sh accounts                    reconcile QOS/accounts/users only (controller)
sudo ./install.sh verify                      test job on every node (controller)
     ./install.sh render [--out DIR]          generated files to DIR; installs nothing
     ./install.sh validate                    check cluster.toml
     ./install.sh steps                       list each command's steps
```

| Option | Effect |
|---|---|
| `--config PATH` | use another TOML (default `cluster/cluster.toml`) |
| `--dry-run` | print commands and file diffs, change nothing |
| `--only a,b` / `--skip a,b` | run or skip named steps (`install.sh steps` lists them) |
| `--force` | restart daemons even if nothing changed |

Every step checks the current state first. After a failure, fix the cause and rerun the same command.

## 5. Using the cluster

Users SSH into the controller with the keys listed in `[[users]].ssh_keys` and submit from there:

```bash
sbatch /opt/team7/jobs/train.sbatch                 # tr_user*: lands on train-01
sbatch /opt/team7/jobs/eval.sbatch                  # ev_user*: lands on eval-01
sbatch -p train /opt/team7/jobs/debug.sbatch        # 10-min GPU/environment check
squeue; sacct -X -S today
```

Arguments after the script name are passed on to the training or eval script. To run another script with the same template: `TEAM7_TRAIN_SCRIPT=/opt/team7/src/other.py sbatch --export=ALL /opt/team7/jobs/train.sbatch`.

Job output goes to `[logs].dir` **on the GPU node** (`/scratch/logs/<job-name>-<jobid>.out`). Until shared storage exists, read it with `ssh train-01`, or with a debug job: `srun -p train -q debug -t 2 cat /scratch/logs/train-123.out`.

### What `train.py` must do

The train template uses the DeepSpeed launcher and forwards `SIGUSR1` only to the training processes (the launcher would die on it). Your `train.py` must:

```python
import os, signal, sys
from pathlib import Path
import torch, torch.distributed as dist
import team7config

cfg = team7config.load()                       # reads $TEAM7_CONFIG
pid_dir = Path(os.environ.get("TEAM7_PID_DIR") or cfg.pid_dir())
pid_dir.mkdir(parents=True, exist_ok=True)
(pid_dir / f"rank-{os.environ.get('RANK', '0')}.pid").write_text(str(os.getpid()))

stop = False
def on_usr1(signum, frame):
    global stop
    stop = True                                # only set a flag in the handler
signal.signal(signal.SIGUSR1, on_usr1)

for step in range(start_step, total_steps):
    train_one_step()
    flag = torch.tensor([int(stop)], device="cuda")
    dist.all_reduce(flag)                      # all ranks stop at the same step
    if flag.item():
        save_checkpoint(step)                  # atomic write + COMPLETE marker
        dist.barrier()                         # every rank done before any exits
        sys.exit(cfg.requeue_exit)             # 75: Slurm requeues the job
```

On start, resume from the newest checkpoint with a `COMPLETE` marker. The requeued job keeps its job ID, and `SLURM_RESTART_COUNT` tells you which segment you're in. Any other non-zero exit is treated as a failure and is not requeued.

Other useful values: `cfg.get("checkpoints.local_dir")`, `cfg.get("checkpoints.s3_uri")`, `cfg.get("training.precision")`. Shell scripts can use `team7-config get <key>`.

## 6. Changing the configuration

| You changed | Run |
|---|---|
| QOS, accounts, users' account or limits | `sudo ./install.sh apply` on the controller |
| Partitions, priority weights, timeouts | `sudo ./install.sh apply` (it runs `scontrol reconfigure`) |
| Node definitions, ports, plugins | `apply` restarts slurmctld and tells you to restart slurmd on the nodes |
| New Unix user or SSH key | `apply` on the controller, then `sudo ./install.sh node <name>` on each node |
| `[training]`, `[eval]`, `[checkpoints]`, job templates | rerun `node <name>` on each node (and `apply` on the controller, which holds the templates users submit) |
| `pyenv/pyproject.toml` | `uv lock` in `pyenv/`, commit `uv.lock`, rerun `node <name>` |

Commit the change, pull it on every host, then run the commands above. `install.sh verify` confirms that every node has the same `cluster.toml` as the controller.

`apply` never deletes accounts, users or QOS from Slurm. It reports any that are in Slurm but not in the TOML.

## 7. Tests (queue_design.md section 12)

On the controller, with both nodes idle:

```bash
sudo python3 /opt/team7/cluster/tests/test_limits.py        # T01-T11, a few minutes
sudo python3 /opt/team7/cluster/tests/test_requeue.py       # T12, T12b, ~15 min each with 3 cycles
sudo /opt/team7/cluster/tests/collect_evidence.sh           # state snapshot for the admin
```

Each command writes its evidence to `./evidence/<timestamp>/`: every command, the user that ran it, its output, and a `summary.md`. T07 and T08 temporarily enable the `limits-test` partition and change the train account's GPU limit. The script restores both from `cluster.toml` at the end, even if a test fails.

T13 to T17 (node failure, health drain, controller and database restarts) are done by hand. See `queue_design.md`.

## 8. Operations

- **Reboot** keeps the NVMe data. **Stop/start or a new instance** gives a blank NVMe: `team7-scratch.service` formats and mounts it again, but local checkpoints are gone. This is why `[checkpoints].durable` should be `s3` or `shared`.
- **Replace a GPU node:** launch it with the same private IP, copy the code and the munge key, and run `sudo ./install.sh node <name> --munge-key /tmp/munge.key`. The controller needs no change.
- **Resume a drained node** after fixing it: `sudo scontrol update nodename=train-01 state=resume`.
- **Controller backups:** `/var/spool/slurmctld` (job state) and the MariaDB database `slurm_acct_db`. Snapshot the root EBS volume, or `mysqldump --single-transaction slurm_acct_db`.

## 9. Troubleshooting

| Symptom | Likely cause |
|---|---|
| Node step stops at "hardware does not match" | node sizes in `cluster.toml` differ from `slurmd -C`; use the printed values, `apply`, rerun |
| Node never registers | ports 6817/6818 blocked, wrong munge key (`sha256sum /etc/munge/munge.key` on both hosts), clocks off (`chronyc tracking`) |
| Node `drain` with reason `Low RealMemory` | `real_memory_mb` too high |
| Node `drain` with reason `Prolog error` or `health: ...` | health check failed; see `/var/log/team7/` on the node |
| `srun` hangs, or the job starts but gets no output | `srun_port_range` not open from the nodes to the controller |
| Job pending with `AssocGrpGRES`, `QOSMax...` | a limit is working as configured; `scontrol show job <id>` shows which |
| `sbatch: Invalid account or account/partition combination` | user not in that account, or the partition doesn't allow it |
| DeepSpeed rebuilds ops in every job | job env differs from install (`TORCH_CUDA_ARCH_LIST`, `CUDA_HOME`); use the templates |
| `nvcc is CUDA 13.x but torch is built for CUDA 12.6` | set `[python].cuda_home` to a 12.x toolkit, e.g. `/usr/local/cuda-12.6` (`ls /usr/local`) |
| Logs: | `/var/log/slurm/` (all hosts), `journalctl -u slurmd`/`slurmctld`/`slurmdbd`, `journalctl -u team7-scratch` |

## 10. Status

Checked so far: config validation and its error messages, `render` output, `--dry-run` of the controller and node installs, `bash -n` on every generated script, `uv lock` of the training environment (torch 2.14.1+cu126, DeepSpeed 0.19.8), and a local simulation of the `SIGUSR1` path through a stand-in for the two DeepSpeed launcher processes.

**Not yet run on EC2.** No real apt, systemd, Slurm or GPU step has been executed. The first real install is the first full test. Expect small fixes, for example in the Ubuntu unit file details or in `sacctmgr` output parsing.
