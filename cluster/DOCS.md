# cluster.toml reference

Every key the installer and `team7config` understand, with its default, allowed values, and what changing it does. Keys you leave out take the default shown. `install.sh validate` checks the rules described here, and `install.sh render` shows the files a change produces.

**Used by** says who reads a key:
- **installer:** used when configuring hosts. A change takes effect after `apply` or rerunning `node`.
- **jobs:** baked into the generated `/opt/team7/jobs/*.sbatch` templates.
- **training:** for your code through `team7config`. The installer only validates it.

Slurm time values use Slurm's formats: `MM`, `MM:SS`, `HH:MM:SS`, `D-HH`, `D-HH:MM`, `D-HH:MM:SS`, or `UNLIMITED`.

TRES tables look like `{ "gres/gpu" = 1, "cpu" = 4, "mem" = "16G" }`. Allowed keys are `cpu`, `mem`, `node`, `billing`, `energy` and `gres/<name>[:<type>]`. A GPU TRES only works if it's in `[slurm].accounting_tres`.

---

## [cluster]

| Key | Default | Meaning and options | Implications |
|---|---|---|---|
| `name` | required | Slurm `ClusterName` and the accounting cluster. Lowercase letters, digits, `-`, `_`, at most 32 characters. | Renaming an installed cluster creates a new, empty accounting cluster; job history stays under the old name. Pick it once. |
| `os_version` | `"24.04"` | Ubuntu release every host must run. | The installer refuses other releases. All hosts need the same release so apt gives one Slurm version. |
| `slurm_version` | `"23.11"` | Prefix the installed `slurmctld -V` / `slurmd -V` must match. `""` disables the check. | Mixed Slurm versions between controller and nodes cause registration and protocol errors. |
| `ntp_server` | `"169.254.169.123"` | Time source added to chrony (Amazon Time Sync Service). | Munge rejects credentials when clocks differ by more than a few minutes. Outside AWS, use your NTP server. |
| `timezone` | `""` | If set (e.g. `"UTC"`), applied with `timedatectl`. | Only affects log timestamps. |

## [controller]

| Key | Default | Meaning and options | Implications |
|---|---|---|---|
| `name` | required | Short hostname of the controller. The installer sets it. | Used in `SlurmctldHost`, `AccountingStorageHost`, `/etc/hosts`, and the nodes' `--conf-server`. |
| `ip` | required | Private IPv4 of the controller. | `install.sh controller` refuses to run on a host without this IP. Launch the instance with it as a fixed private IP. |

## [[nodes]]

One table per compute node.

| Key | Default | Meaning and options | Implications |
|---|---|---|---|
| `name` | required | Short hostname and Slurm `NodeName`. | `install.sh node <name>` selects the entry by this name. |
| `role` | required | `"train"` or `"eval"`. Added as a Slurm node feature. | Validation warns if `[training].partition` contains a non-train node, or `[eval].partition` a non-eval node. |
| `ip` | required | Private IPv4 (`NodeAddr`). | The node install checks it. Keep it fixed so replacements need no config change. |
| `gpus` | `1` | Number of GPUs. `0` makes a CPU-only node. | Sets `Gres=gpu:<type>:<n>` and `gres.conf`. The node install checks `nvidia-smi -L` shows exactly this many. |
| `gpu_type` | `""` | GRES type, e.g. `"t4"`, `"a10g"`, `"h100"`. Required if `gpus > 0`. Lowercase. | Users may request `--gres=gpu:t4:1` or just `--gres=gpu:1`. Changing it needs `apply` plus a slurmd restart. |
| `gpu_name_match` | `""` | Substring that must appear in the `nvidia-smi -L` name. Empty means `gpu_type`. | Catches launching the wrong instance type. |
| `cpus` | required | Logical CPUs, from `slurmd -C` `CPUs=`. | Must equal `sockets × cores_per_socket × threads_per_core`. If Slurm sees fewer CPUs than configured, it drains the node. |
| `sockets`, `cores_per_socket`, `threads_per_core` | none | CPU topology from `slurmd -C`. Set all three or none. | Without them Slurm can misread hyperthreads. With `CR_Core_Memory`, a 1-CPU request then takes a whole core (2 CPUs on g4dn). |
| `real_memory_mb` | required | Memory Slurm may hand out, in MB. At most `slurmd -C` `RealMemory`. | Set ~10% below the reported value to leave RAM for the OS. A higher value drains the node with `Low RealMemory`. |
| `features` | `[]` | Extra Slurm features (role and GPU type are added automatically). | Users can target them with `--constraint`. |
| `weight` | `1` | Slurm `Weight`. Lower weight gets picked first. | Only matters when a partition has several nodes, e.g. for the fallback in queue_design ADR-03. |

g4dn values: `g4dn.xlarge` is 4 CPUs (1×2×2) with 16 GiB (`real_memory_mb ≈ 14000`), and `g4dn.2xlarge` is 8 CPUs (1×4×2) with 32 GiB (`≈ 30000`). Every single-GPU g4dn size has one T4.

## [unix]

| Key | Default | Meaning and options | Implications |
|---|---|---|---|
| `group` | `"team7"` | Primary group for all cluster users. | Owns `checkpoints.local_dir` and the torch extension cache. |
| `gid` | `2000` | GID of that group, the same on every host. | If the group already exists with another GID, the installer stops. |
| `slurm_uid`, `slurm_gid` | `64030` | UID and GID of the `slurm` user, created before the packages are installed. | Must be the same on every host. If `slurm` already exists with another UID, the installer only warns. |
| `shell` | `"/bin/bash"` | Login shell for cluster users. | |

## [secrets]

The munge key is the only secret shared between hosts. The MariaDB password is generated on the controller and stays there.

| Key | Default | Meaning and options | Implications |
|---|---|---|---|
| `munge_key_source` | `"file"` | `"file"`: the controller generates the key and writes it to `munge_key_file`, and you copy it to each node and pass `--munge-key`. `"ssm"`: stored in SSM Parameter Store, and the nodes fetch it. | `file` needs no AWS setup but involves a manual copy. `ssm` needs an IAM role on every host and installs the AWS CLI (snap) if it's missing. |
| `munge_key_file` | `"/root/team7/munge.key"` | Where the controller keeps the team key (mode 0400). On a node, used if it exists and `--munge-key` isn't given. | Back this file up. Losing it means re-keying every host. |
| `ssm_parameter` | `"/team7/munge_key"` | SecureString name, stored as base64. | The controller creates it if missing and never overwrites an existing one. |
| `ssm_region` | `""` | Region of the parameter. Required for `ssm`. | |

## [database]

| Key | Default | Meaning and options | Implications |
|---|---|---|---|
| `name` | `"slurm_acct_db"` | MariaDB database for slurmdbd. | Renaming starts empty accounting (no job history). |
| `user` | `"slurm"` | Database user, only allowed from `localhost`. | |
| `innodb_buffer_pool_size` | `"512M"` | MariaDB buffer pool size. | Raise it on a larger controller. Slurm recommends at least this much. |
| `innodb_lock_wait_timeout` | `900` | Seconds. Slurm's documentation asks for 900. | Lower values can make slurmdbd upgrades and large purges fail. |

## [slurm]

| Key | Default | Meaning and options | Implications |
|---|---|---|---|
| `conf_dir` | `"/etc/slurm"` | Where the Ubuntu packages look for config. | Change only if Slurm itself is configured elsewhere. |
| `ctld_port`, `slurmd_port`, `dbd_port` | `6817`, `6818`, `6819` | Daemon ports. | Open `ctld_port` and `slurmd_port` between hosts. `dbd_port` is local to the controller. |
| `srun_port_range` | `"60001-63000"` | Ports `srun` listens on for callbacks from nodes. | Must be open from the nodes to the controller, or `srun` hangs. |
| `state_save_location` | `"/var/spool/slurmctld"` | Controller job state. | Back it up. Losing it loses queued and running job records. |
| `slurmd_spool_dir` | `"/var/spool/slurmd"` | Node spool; also holds the configless config cache. | |
| `log_dir` | `"/var/log/slurm"` | `slurmctld.log`, `slurmd.log`, `slurmdbd.log`. | |
| `select_type_parameters` | `"CR_Core_Memory"` | What Slurm allocates as consumable: `CR_Core_Memory`, `CR_CPU_Memory`, `CR_Core`, ... | `CR_Core_Memory` hands out whole cores and enforces memory requests. `CR_CPU_Memory` hands out hyperthreads. |
| `scheduler_type` | `"sched/backfill"` | `sched/backfill` or `sched/builtin` (strict FIFO by priority). | Backfill lets small jobs run in gaps without delaying higher-priority ones. |
| `scheduler_parameters` | `["nohold_on_prolog_fail"]` | `SchedulerParameters` list. | Keep `nohold_on_prolog_fail`. Without it, a job requeued after a failed health check is held, and someone has to release it by hand. |
| `task_plugin` | `"task/cgroup,task/affinity"` | `TaskPlugin`. | `task/cgroup` enforces CPU, memory and device limits per job. |
| `preempt_type` | `"preempt/none"` | `preempt/none`, `preempt/qos` or `preempt/partition_prio`. | Anything other than none also needs QOS `Preempt` settings (via `[slurm.extra]`/sacctmgr). See queue_design ADR-07. |
| `accounting_enforce` | `["associations","limits","qos","safe"]` | `AccountingStorageEnforce` flags. | Removing `qos` lets users pick any QOS. Removing `limits` turns off every QOS and association limit. |
| `accounting_tres` | `["gres/gpu"]` | Extra TRES Slurm tracks (`AccountingStorageTRES`). | GPU limits (`gres/gpu` in QOS or `grp_tres`) are only enforced if `gres/gpu` is listed here. |
| `job_requeue` | `true` | `JobRequeue`: whether batch jobs may be requeued by default. | Jobs also ask with `--requeue` (the train template does). |
| `requeue_exit` | `75` | `RequeueExit`: exit code that makes Slurm requeue a job. The train template and `team7config.Config.requeue_exit` use it. | Must not be 1, 2, 126, 127 or above 128 (those come from crashes and signals). A crash must never requeue forever. |
| `return_to_service` | `1` | `ReturnToService`: `0` means a DOWN node stays DOWN; `1` means a node that was DOWN only for not responding returns when it registers; `2` means any DOWN node returns when it registers. | Drained nodes (health failures) always need a manual `resume`. |
| `slurmctld_timeout` | `120` | Seconds before a backup controller takes over (no backup here). | |
| `slurmd_timeout` | `300` | Seconds without contact before a node is marked DOWN and its `--requeue` jobs are requeued. | Lower means faster recovery from node failure, but more false alarms on a slow network. |
| `kill_wait` | `30` | Seconds between SIGTERM and SIGKILL at job end. | The checkpoint path uses SIGUSR1 well before this, so 30 s is enough. |
| `min_job_age` | `300` | Seconds a finished job stays visible in `scontrol show job`. | The tests read `Restarts` from there. Keep it at least 300. |
| `priority_decay_half_life` | `"7-00:00:00"` | Half-life of past usage in fairshare. | Shorter means past usage is forgotten faster. |
| `priority_max_age` | `"7-00:00:00"` | Queue time at which the age factor stops growing. | |

### [slurm.priority_weights]

| Key | Default | Meaning | Implications |
|---|---|---|---|
| `fairshare` | `10000` | Weight of fairshare. | Dominant factor, so heavy users of an account drop in priority within that account. |
| `age` | `1000` | Weight of time spent waiting. | |
| `qos` | `2000` | Weight of QOS `priority`. | Gives `debug` (priority 100) a boost over queued `train`/`eval` jobs. |
| `partition` | `0` | Weight of partition priority. | 0 because each partition has its own node. |
| `job_size` | `0` | Weight of job size. | |

### [slurm.cgroup]

| Key | Default | Meaning | Implications |
|---|---|---|---|
| `constrain_cores` | `true` | Jobs only run on their allocated cores. | |
| `constrain_ram_space` | `true` | Jobs are killed (`OUT_OF_MEMORY`) above their memory request. | Set `--mem` realistically. The train template uses `mem = "0"` (whole node). |
| `constrain_devices` | `true` | Jobs only see their allocated GPUs. | Needed for GPU isolation once several jobs share a multi-GPU node. |

### [slurm.extra]

A pass-through table of `slurm.conf` keys the installer doesn't model, written as `Key = "value"`. Keys the installer already writes (e.g. `RequeueExit`, `NodeName`, `PriorityWeight*`) are rejected, so a setting can't be defined twice.

## [health]

| Key | Default | Meaning and options | Implications |
|---|---|---|---|
| `enabled` | `false` | Adds a Prolog and/or `HealthCheckProgram` to `slurm.conf` and installs the wrapper scripts on the nodes. | When true, the node install runs the check once and stops if it fails. A broken check would otherwise drain the node on its first job. |
| `launch_script` | `"/opt/team7/src/health/launch.sh"` | The health workstream's entry point. Must exist and be executable on the nodes. | Deploy it with `[code]`. |
| `quick_args` | `"--quick"` | Arguments for the Prolog and periodic checks. | Must finish well within `check_timeout_s`. NCCL belongs in `--full`, not here. |
| `prolog` | `true` | Run the check before every job (`Prolog`, `PrologFlags=Alloc`). | A non-zero exit drains the node and requeues the job. Adds a few seconds to every job start. |
| `periodic` | `true` | Run the check every `interval_s` (`HealthCheckProgram`). | Slurm ignores this program's exit code, so the wrapper drains the node itself with reason `health: ...`. |
| `interval_s` | `300` | `HealthCheckInterval`. | |
| `node_state` | `"ANY"` | `HealthCheckNodeState`: `ANY`, `IDLE`, `ALLOC`, `MIXED`, `NONDRAINED_IDLE`. | `IDLE` avoids running checks next to training, but misses faults during long jobs. |
| `check_timeout_s` | `120` | Timeout the wrappers put around `launch_script`. | A timeout counts as a failure (exit 124), which drains the node. |
| `prolog_timeout_s` | `180` | `PrologEpilogTimeout`. | Keep it above `check_timeout_s`. |
| `log_dir` | `"/var/log/team7"` | Health output on the nodes (`prolog.jsonl`, `healthcheck.jsonl`, `*.log`). | |

## [storage.scratch] (GPU nodes)

| Key | Default | Meaning and options | Implications |
|---|---|---|---|
| `type` | `"nvme"` | `"nvme"`: the instance-store NVMe, formatted if blank and mounted on every boot by `team7-scratch.service`. `"dir"`: a plain directory on the root EBS volume. | NVMe is fast but **erased on stop/start or instance replacement** (a reboot keeps it). `dir` survives a stop but is slower and limited by the root volume size. |
| `mount` | `"/scratch"` | Mount point. | |
| `device` | `"auto"` | `auto` picks the first "Amazon EC2 NVMe Instance Storage" device. Otherwise a device path such as `/dev/nvme1n1`. | Device names can change between boots, so prefer `auto`. |
| `filesystem` | `"xfs"` | `xfs` or `ext4`, used only when formatting. | |

`slurmd` requires `team7-scratch.service`. If the disk can't be mounted, slurmd doesn't start and the node takes no jobs.

## [storage.shared] (all hosts)

| Key | Default | Meaning and options | Implications |
|---|---|---|---|
| `type` | `"none"` | `"none"`, `"efs"` or `"nfs"`. Mounted via a managed block in `/etc/fstab`. | With `none`, logs and local checkpoints stay on each node, and eval can only see training checkpoints through S3. |
| `mount` | `"/shared"` | Mount point on every host. | |
| `efs_id`, `region` | `""` | EFS file system ID and region (for `efs`). Mounted as `<id>.efs.<region>.amazonaws.com:/` over NFS 4.1. | The VPC needs DNS resolution and an EFS mount target in the subnet. The EFS security group must allow TCP 2049 from the hosts. |
| `nfs_server`, `nfs_export` | `""`, `"/"` | NFS server and export (for `nfs`). | |
| `options` | `""` | Mount options. Empty uses the AWS-recommended EFS/NFS options. | |

## [checkpoints] (training, plus directory setup by the installer)

| Key | Default | Meaning and options | Implications |
|---|---|---|---|
| `local_dir` | `"/scratch/ckpt"` | Fast first write. Created as `root:<group>` mode 2775. | On NVMe this is lost when the node is stopped or replaced. |
| `durable` | `"s3"` | Where checkpoints survive a node loss and where eval reads them: `"s3"`, `"shared"` (EFS/NFS) or `"none"`. | `none` passes validation with a warning: a node failure loses every checkpoint, and eval can't see training output. |
| `s3_uri` | `""` | `s3://bucket/prefix`, required for `s3`. | Needs instance-role access (read/write on train, read on eval). Upload only after the `COMPLETE` marker, with the manifest last. |
| `shared_dir` | `""` | Directory under `[storage.shared].mount`, required for `shared`. | The installer creates it with mode 2775. |
| `keep_local`, `keep_durable_last`, `keep_durable_every` | `2`, `3`, `0` | Retention: newest N locally, newest N durable, plus every Kth step (`0` = off). | Read by the checkpoint code. The installer doesn't prune anything. |

## [logs]

| Key | Default | Meaning | Implications |
|---|---|---|---|
| `dir` | `"/scratch/logs"` | Job `--output` directory in the templates, mode 1777. | On scratch, logs are only on the node that ran the job and are lost on stop/start. Use a path under `/shared` once shared storage exists. |

## [code] (all hosts)

| Key | Default | Meaning and options | Implications |
|---|---|---|---|
| `source` | `"none"` | `"none"`, `"local"` (rsync `local_path` to `dest`) or `"git"` (clone `git_url`, check out `git_ref`). | Jobs run from `dest` on the node, so the code must be there. The example config uses `local`. |
| `local_path` | `".."` | Source directory, relative to `cluster/`. Normally the repository root. | Copied with `rsync --delete`, so files removed locally disappear from `dest` too. |
| `git_url`, `git_ref` | `""`, `"main"` | Repository and branch, tag or commit. | The host needs read access (a deploy key or a public repo). Pin a tag or commit so every host runs the same code. |
| `dest` | `"/opt/team7/src"` | Install location. Also the jobs' `--chdir`. | Paths like `[training].script` normally point inside it. |
| `exclude` | `[".git", ".venv", "__pycache__", "*.pyc"]` | rsync excludes for `local`. | |

The `cluster/` directory itself is always copied to `/opt/team7/cluster`, whatever `source` is.

## [python] (GPU nodes)

| Key | Default | Meaning and options | Implications |
|---|---|---|---|
| `project` | `"pyenv"` | Directory (inside `cluster/`) with `pyproject.toml` and `uv.lock`. | Package versions and the torch index live there, not in the TOML (see below). |
| `version` | `"3.12"` | Python version uv installs (uv-managed, not the system Python). | Must satisfy the project's `requires-python`. |
| `uv_version` | `"0.13.6"` | Exact uv release, downloaded from GitHub. | Pin it so every node resolves identically. |
| `uv_sha256` | `""` | Optional sha256 of the uv tarball. | Set it to make the download verifiable. |
| `venv` | `"/opt/team7/venv"` | Virtual environment (root-owned, readable by all). | On local disk because importing torch from NFS is slow. |
| `python_install_dir` | `"/opt/team7/python"` | Where uv puts Python. | uv's default is under `/root`, which job users can't read. |
| `cache_dir` | `"/var/cache/team7-uv"` | uv download cache. | Makes reinstalls fast. Safe to delete. |
| `cuda_home` | `"/usr/local/cuda"` | CUDA toolkit that DeepSpeed uses to compile ops. Exported as `CUDA_HOME` in jobs. | Its major version must match torch's CUDA (12 for the cu126 wheels), or the install stops. The GPU AMI ships several toolkits; `ls /usr/local` shows them. |
| `cuda_arch_list` | `"7.5"` | `TORCH_CUDA_ARCH_LIST` for builds (7.5 = T4, 8.6 = A10G, 9.0 = H100). | Jobs use the same value. A different value in a job makes DeepSpeed rebuild ops at job start. |
| `torch_extensions_dir` | `"/opt/team7/torch_extensions"` | Shared cache of compiled DeepSpeed ops, group-writable by `<group>`. | Without a shared cache, each user compiles ops on their first job. |
| `deepspeed_prebuild_ops` | `["FusedAdamBuilder"]` | DeepSpeed op builders to compile at install time (class names from `deepspeed.ops.op_builder`). | `[]` skips the build and the nvcc check. Ops you use but don't list are compiled inside the first job, using its time limit. |

**Package versions (`pyenv/pyproject.toml`).** torch is pinned to `2.14.1` from the `cu126` index (supports the T4, needs NVIDIA driver ≥ 560), and DeepSpeed to `0.19.8`. DeepSpeed ships only a source package, and its `setup.py` imports torch. So it's built without build isolation, after torch is installed (two `uv sync` passes), and its dependencies are declared in `[[tool.uv.dependency-metadata]]` so `uv lock` doesn't have to build it. To change versions, edit the file, update the `dependency-metadata` block to the new DeepSpeed's requirements, run `uv lock` in `pyenv/`, commit `uv.lock`, and rerun `node <name>`.

## [training] (jobs, training)

| Key | Default | Meaning and options | Implications |
|---|---|---|---|
| `partition`, `account`, `qos` | `"train"` ×3 | `#SBATCH` values in `train.sbatch`. | Validation checks the account may use the partition and QOS. |
| `job_name` | `"train"` | `--job-name`, also the log file prefix. | |
| `script` | `"/opt/team7/src/train.py"` | Training entry point. Override per job with `TEAM7_TRAIN_SCRIPT`. | Path on the GPU node. |
| `args` | `[]` | Arguments always passed to the script. Arguments after `sbatch train.sbatch` are appended. | |
| `deepspeed_config` | `""` | Passed as `--deepspeed_config <path>` when set. Override with `TEAM7_DS_CONFIG` (an empty value passes none). | |
| `time` | `"24:00:00"` | `--time`. Must fit the QOS `max_wall` and the partition `max_time`. | Longer runs continue through self-requeue, one segment per `time`. |
| `signal_lead_s` | `300` | `--signal=B:USR1@<s>`: how long before the time limit the job is told to checkpoint. | Must cover a full checkpoint save plus margin: at least 2 × save time + 60 s. Too short and the job is killed mid-save, losing the segment's work. |
| `pid_dir` | `"/tmp/team7-{job_id}"` | Where each rank writes `rank-<RANK>.pid`. Must contain `{job_id}`. Exported to jobs as `TEAM7_PID_DIR`. | The batch script signals only these PIDs, never the DeepSpeed launcher processes. |
| `gpus_per_job` | `0` | `--gres=gpu:N`. `0` means all GPUs of the partition's first node. | |
| `cpus_per_task` | `0` | `--cpus-per-task`. `0` means all CPUs of that node. | |
| `mem` | `"0"` | `--mem`. `"0"` means all memory of the node. | With all resources taken, a debug job on the train node waits for training to end. |
| `master_port_base` | `20000` | DeepSpeed `--master_port` = base + job ID mod 20000. | Avoids port clashes when two jobs share a node. |
| `precision` | `"fp16"` | `fp16`, `bf16` or `fp32`, for your training code. | T4 GPUs have no hardware bf16. Validation warns about bf16 on t4/v100. |

## [eval] (jobs)

| Key | Default | Meaning | Implications |
|---|---|---|---|
| `partition`, `account`, `qos` | `"eval"` ×3 | `#SBATCH` values in `eval.sbatch`. | |
| `job_name` | `"eval"` | | |
| `script` | `"/opt/team7/src/eval.py"` | Run with the venv's Python. Override with `TEAM7_EVAL_SCRIPT`. | Should read only checkpoints with a `COMPLETE` marker. |
| `args` | `[]` | Fixed arguments. | |
| `time` | `"04:00:00"` | `--time`. | |
| `gpus_per_job`, `cpus_per_task`, `mem` | `1`, `2`, `"8G"` | Resources per eval job. | Smaller requests let more eval jobs share a multi-GPU node. |

## [debug] (jobs)

| Key | Default | Meaning | Implications |
|---|---|---|---|
| `qos` | `"debug"` | QOS of `debug.sbatch`. Users pass `-p train` or `-p eval`. | The QOS must be allowed on both partitions and in both accounts. |
| `time` | `"00:10:00"` | `--time`. | |
| `gpus_per_job`, `cpus_per_task`, `mem` | `1`, `1`, `"4G"` | Resources. | |

## [[partitions]]

| Key | Default | Meaning and options | Implications |
|---|---|---|---|
| `name` | required | Partition name. | |
| `nodes` | required | Node names. | One partition per node keeps teams apart (queue_design ADR-01). |
| `allow_accounts` | `[]` (all) | `AllowAccounts`. | Jobs from other accounts are rejected at submit. |
| `allow_qos` | `[]` (all) | `AllowQos`. | Stops users submitting e.g. the eval QOS to the train partition. |
| `max_time` | `"UNLIMITED"` | `MaxTime`. | A job asking for more pends with `PartitionTimeLimit`. |
| `default_time` | `""` | `DefaultTime` for jobs without `--time`. | |
| `state` | `"UP"` | `UP`, `DOWN`, `DRAIN` or `INACTIVE`. | `limits-test` is `INACTIVE`, and `test_limits.py` enables it only during T07/T08. |
| `default` | `false` | Default partition. At most one. | With no default, every job must name its partition, which avoids landing on the wrong node. |
| `extra` | `{}` | Other `PartitionName` options, e.g. `{ OverSubscribe = "EXCLUSIVE" }`. | |

## [[qos]]

Unset limits are cleared in Slurm (set to -1) on every `apply`, so the TOML stays the source of truth.

| Key | Default | Meaning | Implications |
|---|---|---|---|
| `name` | required | QOS name. | |
| `description` | `""` | Free text (defaults to the name). | |
| `priority` | `0` | QOS priority, scaled by `priority_weights.qos`. | |
| `max_wall` | `""` (none) | `MaxWall` per job. | A job asking for more pends with `QOSMaxWallDurationPerJobLimit`. |
| `max_tres_per_job` | `{}` | `MaxTRESPerJob`, e.g. `{ "gres/gpu" = 1 }`. | Over-limit jobs pend with e.g. `QOSMaxGRESPerJob`. |
| `max_tres_per_user` | `{}` | `MaxTRESPerUser`. | |
| `max_jobs_pu`, `max_jobs_pa` | none | Running jobs per user / per account. | `train` uses `max_jobs_pa = 1`: one training run per team at a time. |
| `max_submit_jobs_pu`, `max_submit_jobs_pa` | none | Queued plus running jobs per user / per account. | Submissions above it are rejected. |
| `flags` | `[]` | QOS flags, e.g. `["DenyOnLimit"]`. | `DenyOnLimit` rejects over-limit jobs at submit instead of leaving them pending. Removing a flag later needs `sacctmgr modify qos <name> set Flags-=<flag>`. |

## [[accounts]]

| Key | Default | Meaning | Implications |
|---|---|---|---|
| `name` | required | Account (team) name. `root` is reserved. | |
| `description`, `organization` | `""` | Free text. | |
| `fairshare` | `1` | Account share. | Accounts don't compete when each has its own node. |
| `grp_tres` | `{}` | `GrpTRES` for the whole account, e.g. `{ "gres/gpu" = 1 }`. | Over-limit jobs pend with `AssocGrpGRES`. With one node per account, the node already enforces this. It matters once nodes are shared. |
| `grp_jobs` | none | Running jobs for the account. | |
| `max_jobs` | none | Running jobs per user in the account. | |
| `qos` | `[]` | QOS the account's users may use. Copied onto each user association. | Empty leaves Slurm's default `normal`. |
| `default_qos` | `""` | QOS used when a job doesn't pass `--qos`. Must be in `qos`. | |

## [[users]]

| Key | Default | Meaning | Implications |
|---|---|---|---|
| `name` | required | Unix and Slurm user name. | Created on every host with a home directory. |
| `uid` | required | Fixed UID, at least 1000, unique, and the same on every host. | If the user already exists with another UID, the install stops. Fix it by hand. |
| `account` | required | The user's single account (queue_design ADR-02). Also the default account. | Moving a user leaves the old association in place. `apply` reports it, and you remove it with `sacctmgr remove user <name> account=<old>`. |
| `admin_level` | `"None"` | `None`, `Operator` or `Administrator` in Slurm. | Operators can manage other users' jobs and nodes. Keep `None` for normal users (test T11). |
| `fairshare` | `1` | Share inside the account. | |
| `ssh_keys` | `[]` | Public keys written to `~/.ssh/authorized_keys` on every host. | An empty list leaves an existing file alone. A non-empty list replaces the file. |
| `groups` | `[]` | Extra Unix groups (must exist). | |

## [tests]

| Key | Default | Meaning | Implications |
|---|---|---|---|
| `limits_partition` | `"limits-test"` | Partition spanning both nodes, used by T07/T08. | Without it, T07/T08 are skipped. |
| `limits_qos` | `"limits_test"` | QOS with no limits, added to the train account only during T07. | Without it, T07/T08 are skipped. |
| `requeue_cycles` | `3` | How many requeues T12/T12b expect. | Each cycle takes about `requeue_time`. |
| `requeue_time` | `"00:04:00"` | `--time` for the requeue test jobs. | |
| `requeue_signal_lead_s` | `60` | `--signal` lead for the requeue tests. | Must be shorter than `requeue_time`. |

---

## What a change needs

| Change | Command | Restarts |
|---|---|---|
| QOS, accounts, users' limits/QOS | `apply` (controller) | none |
| Partitions, priority, timeouts, `[slurm.extra]` | `apply` | `scontrol reconfigure` |
| `[[nodes]]`, ports, plugins, `ClusterName` | `apply` | slurmctld restart. It then prints that slurmd must be restarted on the nodes. |
| `[health]` | `apply`, then `node <name>` on each node | reconfigure |
| New Unix user, `ssh_keys`, `[unix]` | `apply`, then `node <name>` | none |
| `[training]`, `[eval]`, `[debug]` | `apply` (templates on the controller), then `node <name>` (TOML copy) | none |
| `[storage.*]`, `[python]`, `pyenv/` | `node <name>` | slurmd (scratch), or a venv rebuild |
| `[secrets]` source | controller, then each node with the key | munge |
