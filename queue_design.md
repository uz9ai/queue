# Queue Design (Slurm)

| Field | Value |
|---|---|
| Version | 0.2 (draft) |
| Status | Proposed, waiting for admin review |
| Authors | Team 7, Queue workstream |
| Related | `week1_plan.md`, `checkpoint_design.md`, `health/` |
| Replaces | Team 7 Infrastructure Queue Design v2.0 (PDF) for the Week 1 scope |

Values marked **(prov.)** are provisional. They come from the config v0 estimate or from Week 1 hardware and must be checked again before production. Values marked **TBD** are not decided yet; see [Open questions](#14-open-questions).

---

## 1. Goals and non-goals

### Goals

- Run one training job and its evaluation jobs on separate GPU nodes, each owned by one team (Slurm account).
- Enforce per-account and per-user limits, and prove they work with saved evidence.
- Lose no training work when a job hits its time limit, a node fails, or a health check removes a node.
- Requeue jobs automatically after those events and resume from the last good checkpoint.
- Keep bad nodes out of scheduling through the health checks in `health/`.

### Non-goals

- Multi-node training. Each training job runs on one node.
- Preemption between teams. The teams use different nodes.
- High availability for the controller. One controller with backups is enough for now.
- Running in Docker. We test on real EC2 nodes (see [ADR-08](#adr-08-test-on-real-ec2-nodes-not-docker)).

---

## 2. Cluster assumptions

| Item | Week 1 test bed | Target (Week 2 onward) |
|---|---|---|
| Training node | 1 × g4dn single-GPU instance (`train-01`), size TBD | 1 × multi-GPU instance (`train-01`), GPU count from config v0 (prov.) |
| Eval node | 1 × g4dn single-GPU instance (`eval-01`), size TBD | 1 × multi-GPU instance (`eval-01`), type TBD |
| Controller | 1 × small CPU instance (`ctl-01`) | Same |
| GPU | NVIDIA T4, 16 GB. No NVLink. No hardware bf16 (fp16 only). | TBD |
| GPUs per node (`G`) | 1 | TBD from config v0 (prov.) |
| Interconnect | Not needed (single-GPU, single-node jobs) | NVLink inside the node. No EFA needed. |
| Training launcher | DeepSpeed launcher (`deepspeed`), single node | Same |
| Shared storage | TBD (FSx for Lustre or EFS) for checkpoints, readable by both nodes | FSx for Lustre, with S3 for durable copies (prov.) |
| Slurm version | TBD. The same version on every host. | Same |
| OS users | 4 users with the same UID/GID on all hosts | Same |

The configuration below uses `G` for GPUs per node, so moving to multi-GPU nodes changes values, not structure.

Every single-GPU g4dn size (`g4dn.xlarge` to `g4dn.16xlarge`) has one T4. Larger sizes only add vCPUs, RAM and local NVMe. `g4dn.xlarge` has 4 vCPUs and 16 GiB RAM, which is tight for data loading next to training. The node definitions take CPU and memory from `slurmd -C`, so any size works with this design.

---

## 3. Slurm components

| Component | Runs on | Notes |
|---|---|---|
| `munge` | All hosts | One shared key, mode 0400, owned by `munge`. Never stored in git. |
| `slurmctld` | `ctl-01` | `StateSaveLocation=/var/spool/slurmctld` on an EBS volume. |
| `slurmdbd` | `ctl-01` | Talks to MariaDB on localhost. |
| MariaDB | `ctl-01` | Database `slurm_acct_db` on an EBS volume. |
| `slurmd` | `train-01`, `eval-01` | One per GPU node. |

### Backups

- **Configuration:** `slurm.conf`, `gres.conf`, `cgroup.conf`, `slurmdbd.conf` (secrets removed) and `setup_accounts.sh` live in this repo. Every change goes through git.
- **Controller state:** daily EBS snapshot of the `StateSaveLocation` volume.
- **Accounting database:** daily `mysqldump --single-transaction slurm_acct_db`, uploaded to S3.
- **Restore:** start a new `ctl-01` from the snapshot, restore the dump, start `slurmdbd` and then `slurmctld`. Running jobs on the GPU nodes keep running while the controller is down (see [§11](#11-failure-handling)).

---

## 4. Accounts and users

Two accounts, one per team. Each user belongs to exactly one account ([ADR-02](#adr-02-two-accounts-one-account-per-user)).

| Account | Users | Node | Default QOS | Allowed QOS |
|---|---|---|---|---|
| `train` | `tr_user1`, `tr_user2` (names TBD) | `train-01` | `train` | `train`, `debug` |
| `eval` | `ev_user1`, `ev_user2` (names TBD) | `eval-01` | `eval` | `eval`, `debug` |

The Slurm administrator account is separate and is not a member of either team.

---

## 5. Partitions

One partition per node ([ADR-01](#adr-01-one-partition-per-node)). Neither partition is the default, so every job must name its partition.

```
PartitionName=train Nodes=train-01 AllowAccounts=train AllowQos=train,debug MaxTime=<train MaxWall> DefaultTime=01:00:00 State=UP
PartitionName=eval  Nodes=eval-01  AllowAccounts=eval  AllowQos=eval,debug  MaxTime=<eval MaxWall>  DefaultTime=00:30:00 State=UP
```

A third partition, `limits-test`, spans both nodes. It exists only for the limit tests in [§12](#12-test-plan) and is kept `State=INACTIVE` at other times:

```
PartitionName=limits-test Nodes=train-01,eval-01 AllowAccounts=train AllowQos=limits_test,debug State=INACTIVE
```

Debug jobs use the `debug` QOS on the team's own partition, not a separate partition ([ADR-09](#adr-09-debug-jobs-use-a-qos-not-a-partition)).

---

## 6. Limits and QOS

### Enforcement settings

```
AccountingStorageType=accounting_storage/slurmdbd
AccountingStorageEnforce=associations,limits,qos,safe
AccountingStorageTRES=gres/gpu
```

- `associations` rejects jobs from users without a valid user/account association.
- `limits` enforces association and QOS limits.
- `qos` stops users from picking a QOS that is not on their association's list.
- `safe` only starts a job if it can finish within `GrpTRESMins`/`GrpWall` limits. We don't set those limits yet, but turning on `safe` now means adding them later won't need a config change.
- `AccountingStorageTRES=gres/gpu` makes Slurm track GPUs as a resource. Without it, `gres/gpu` limits are not enforced.

### QOS definitions

| QOS | MaxWall | MaxTRESPerJob | Job limits | Priority | Used by |
|---|---|---|---|---|---|
| `train` | 24:00:00 (prov.) | `gres/gpu=G` | `MaxJobsPA=1`, `MaxSubmitJobsPU=4` | 0 | `train` account |
| `eval` | 04:00:00 (prov.) | `gres/gpu=G` | `MaxJobsPU=G`, `MaxSubmitJobsPU=20` | 0 | `eval` account |
| `debug` | 00:30:00 | `gres/gpu=1` | `MaxJobsPU=1`, `MaxSubmitJobsPU=2` | 100 | both accounts |
| `limits_test` | 00:10:00 | none | none | 0 | `train`, only during tests |

`MaxJobsPA=1` on `train` means one training run per team at a time. That matches the hardware, which has one training node. The training MaxWall can be moderate because long runs continue through self-requeue ([§8](#8-time-limits-checkpoints-and-requeue)).

### Association limits

| Account | Fairshare | GrpTRES |
|---|---|---|
| `train` | 1 | `gres/gpu=G` |
| `eval` | 1 | `gres/gpu=G` |

Each user has `Fairshare=1` inside their account.

With one partition per node, the node itself already caps each account at `G` GPUs, so `GrpTRES` only acts as a backstop today. It becomes the real limit if nodes are ever shared, for example under the fallback option in [ADR-03](#adr-03-training-node-failure-the-job-waits). We still test it so that we know it works ([§12](#12-test-plan), T07).

### Account setup (`setup_accounts.sh`, outline)

```bash
sacctmgr -i add cluster team7

sacctmgr -i add qos train,eval,debug,limits_test
sacctmgr -i modify qos train       set MaxWall=24:00:00 MaxTRESPerJob=gres/gpu=$G MaxJobsPA=1 MaxSubmitJobsPU=4
sacctmgr -i modify qos eval        set MaxWall=04:00:00 MaxTRESPerJob=gres/gpu=$G MaxJobsPU=$G MaxSubmitJobsPU=20
sacctmgr -i modify qos debug       set MaxWall=00:30:00 MaxTRESPerJob=gres/gpu=1 MaxJobsPU=1 MaxSubmitJobsPU=2 Priority=100
sacctmgr -i modify qos limits_test set MaxWall=00:10:00

sacctmgr -i add account train Cluster=team7 Fairshare=1 GrpTRES=gres/gpu=$G QOS=train,debug DefaultQOS=train
sacctmgr -i add account eval  Cluster=team7 Fairshare=1 GrpTRES=gres/gpu=$G QOS=eval,debug  DefaultQOS=eval

for u in tr_user1 tr_user2; do sacctmgr -i add user "$u" Account=train DefaultAccount=train Fairshare=1; done
for u in ev_user1 ev_user2; do sacctmgr -i add user "$u" Account=eval  DefaultAccount=eval  Fairshare=1; done
```

The script must be safe to run twice. It should check for existing objects with `sacctmgr show` before adding them, and use `modify` to update limits.

---

## 7. Scheduling and priority

```
SchedulerType=sched/backfill
SelectType=select/cons_tres
SelectTypeParameters=CR_Core_Memory
GresTypes=gpu
PriorityType=priority/multifactor
PriorityDecayHalfLife=7-00:00:00
PriorityWeightFairshare=10000
PriorityWeightAge=1000
PriorityWeightQOS=2000
PriorityWeightPartition=0
PriorityWeightJobSize=0
PreemptType=preempt/none
```

- Fairshare matters between the two users inside each account. The two accounts don't compete because they use different nodes.
- The `debug` QOS priority places debug jobs ahead of queued training or eval jobs. It does not stop a running job ([ADR-07](#adr-07-no-preemption)).
- `PriorityWeightPartition=0` because each partition has its own node, so partition priority has no effect.

### Nodes, GRES and cgroups

```
# slurm.conf. CPUs and RealMemory come from `slurmd -C` on each node.
NodeName=train-01 CPUs=<from slurmd -C> RealMemory=<from slurmd -C, minus OS reserve> Gres=gpu:<G> State=UNKNOWN
NodeName=eval-01  CPUs=<from slurmd -C> RealMemory=<from slurmd -C, minus OS reserve> Gres=gpu:<G> State=UNKNOWN

# gres.conf. Week 1 (g4dn, one T4) has File=/dev/nvidia0. Multi-GPU nodes use File=/dev/nvidia[0-<G-1>].
NodeName=train-01 Name=gpu File=/dev/nvidia0
NodeName=eval-01  Name=gpu File=/dev/nvidia0

# cgroup.conf
ConstrainCores=yes
ConstrainRAMSpace=yes
ConstrainDevices=yes
```

```
ProctrackType=proctrack/cgroup
TaskPlugin=task/cgroup,task/affinity
JobAcctGatherType=jobacct_gather/cgroup
```

`ConstrainDevices=yes` means a job can only see the GPUs it was given. On multi-GPU nodes this is what keeps two eval jobs from using each other's GPUs.

---

## 8. Time limits, checkpoints and requeue

This section is where the queue design connects to the checkpoint design. Its decisions are recorded in [ADR-05](#adr-05-checkpoint-on-sigusr1-and-self-requeue-with-requeueexit).

### Settings

```
JobRequeue=1
RequeueExit=75
SchedulerParameters=nohold_on_prolog_fail
```

### How it works

Training runs under the DeepSpeed launcher inside the batch script, with no `srun`. The process tree on the node is:

```
batch script (bash)
└─ deepspeed           (runner, deepspeed/launcher/runner.py)
   └─ launcher         (deepspeed/launcher/launch.py)
      └─ train.py      (one process per GPU, i.e. per rank)
```

1. A training job is submitted with `--requeue` and `--signal=B:USR1@<lead>`. The `B:` prefix means Slurm signals only the batch shell, not the other processes.
2. `<lead>` seconds before the time limit, Slurm sends `SIGUSR1` to the batch script.
3. The batch script forwards `SIGUSR1` only to the `train.py` ranks. Each rank writes its PID to a file at startup, and the script signals those PIDs. The two DeepSpeed launcher processes are not signalled (see below).
4. Each rank saves its part of the checkpoint (atomic write, manifest, `COMPLETE` marker). All ranks then wait at a barrier until every save is done, and all ranks exit with code **75**.
5. The exit code reaches the batch script unchanged. `launch.py` sees a non-zero exit, stops any remaining ranks and exits with that code (75). `runner.py` exits with its child's code (75). The batch script exits 75.
6. `RequeueExit=75` makes Slurm requeue the job. It keeps the same job ID, returns to pending, and resumes from the newest valid checkpoint when it starts again.

Exit code 75 is reserved for "checkpoint saved, requeue me." Any other non-zero exit is a real failure. Those jobs end as `FAILED` and are not requeued, so a crashing job doesn't loop forever.

### Why the launcher processes must not get SIGUSR1

From the DeepSpeed source (`deepspeed/launcher/launch.py` and `runner.py` on `master`, checked 27 Oct 2026):

- Neither launcher process handles `SIGUSR1`. `launch.py` only handles `SIGINT` and `SIGTERM`. `runner.py` handles none for a single-node run. Python's default action for `SIGUSR1` ends the process, so a signalled launcher dies at once. The job then ends and Slurm kills the ranks, possibly in the middle of a checkpoint write.
- `launch.py` kills all other ranks as soon as any rank exits non-zero. This is why the barrier in step 4 comes **before** any rank exits. Without it, on a multi-GPU node, the first rank to exit 75 would kill ranks that are still writing. With one GPU per node this can't happen yet, but the rule applies from the start.

This is also why the batch script doesn't use `scancel --signal=USR1`. There is no `srun` step for it to signal, and `scancel --full --signal=USR1` would signal the batch script's child processes, which include the launchers.

Requirements for `train.py`, to be written into `checkpoint_design.md`:

- At startup, write `os.getpid()` to `$TEAM7_PID_DIR/rank-$RANK.pid`.
- Install a `SIGUSR1` handler that only sets a flag. Check the flag at the next step boundary.
- When the flag is set: save, run a barrier across all ranks, then exit 75 on every rank.

### Signal lead time

`<lead>` must cover a full checkpoint save plus a margin:

```
lead ≥ 2 × (estimated checkpoint save time) + 60 s
```

Week 1 uses `lead = 300 s` (prov.). We recompute it from the config v0 model-size estimate once `checkpoint_design.md` gives a save time, and again when the measured save time is known.

### Training batch script (template)

```bash
#!/bin/bash
#SBATCH --job-name=train
#SBATCH --account=train
#SBATCH --partition=train
#SBATCH --qos=train
#SBATCH --nodes=1
#SBATCH --ntasks=1                   # one batch task. The DeepSpeed launcher starts one rank per GPU.
#SBATCH --gres=gpu:1                 # = G
#SBATCH --time=24:00:00              # <= train QOS MaxWall (prov.)
#SBATCH --requeue
#SBATCH --signal=B:USR1@300          # lead time (prov.)
#SBATCH --open-mode=append           # keep logs across requeues
#SBATCH --output=logs/%x-%j.out

export TEAM7_PID_DIR="/tmp/team7-${SLURM_JOB_ID}"
mkdir -p "$TEAM7_PID_DIR"
trap 'rm -rf "$TEAM7_PID_DIR"' EXIT

# Signal only the train.py ranks, never the launcher processes.
forward_usr1() {
  for f in "$TEAM7_PID_DIR"/rank-*.pid; do
    [ -e "$f" ] && kill -USR1 "$(cat "$f")" 2>/dev/null
  done
}
trap forward_usr1 USR1

# Don't pass --num_gpus: the launcher then uses the GPUs Slurm put in CUDA_VISIBLE_DEVICES.
# A per-job port keeps two jobs on one node from colliding on the default 29500.
deepspeed --master_port=$((20000 + SLURM_JOB_ID % 20000)) \
  train.py --deepspeed_config ds_config.json --resume auto &
pid=$!

# `wait` returns early when the trap fires. Keep waiting until the launcher exits.
while true; do
  wait "$pid"; rc=$?
  kill -0 "$pid" 2>/dev/null || break
done
exit "$rc"
```

The batch script runs on the allocated node, so a single-node DeepSpeed job doesn't need `srun`. The launcher and the ranks run in the job's batch step, which still has the job's GPU and cgroup limits.

### Runs longer than MaxWall

Self-requeue covers this. Each requeue is one segment of the run. `scontrol show job` reports the count as `Restarts`. We don't use `--dependency` chains ([ADR-05](#adr-05-checkpoint-on-sigusr1-and-self-requeue-with-requeueexit)).

### Eval jobs

Eval jobs are short and don't use the signal. They read only checkpoints that have a `COMPLETE` marker and a valid manifest, so they never read a half-written checkpoint. A failed eval job is resubmitted, not requeued.

---

## 9. Preemption

None (`PreemptType=preempt/none`). The two teams use separate nodes, and debug jobs wait instead of interrupting work. See [ADR-07](#adr-07-no-preemption).

---

## 10. Health integration

Recorded in [ADR-06](#adr-06-health-checks-in-the-prolog-and-on-a-timer).

| Hook | Runs | Command | On failure |
|---|---|---|---|
| `Prolog` (with `PrologFlags=Alloc`) | Before each job starts, as root | `health/launch.sh --quick` | Prolog exits non-zero, Slurm drains the node and requeues the job. `nohold_on_prolog_fail` stops the requeued job from being held. |
| `HealthCheckProgram` | Every `HealthCheckInterval=300` s, `HealthCheckNodeState=ANY` | `health/health_quick.sh`, a wrapper that calls `launch.sh --quick` | The script drains the node itself with `scontrol update nodename=$(hostname -s) state=drain reason="health: <check>"` |
| Node bring-up | Once, before the node first joins | `health/launch.sh --full` (includes NCCL) | The node stays out of Slurm until it passes. |

Points for the health workstream:

- **Slurm does not drain a node based on `HealthCheckProgram`'s exit code.** The program must call `scontrol` itself. Only the Prolog's exit code drains the node. `week1_plan.md` says "Slurm uses these to drain nodes"; that is true only for the Prolog.
- `HealthCheckProgram` takes a path with no arguments, which is why the `health_quick.sh` wrapper is needed.
- `--quick` must finish well under `PrologEpilogTimeout`. NCCL runs only in `--full`.
- Drain reasons start with `health:` so an admin can tell health drains apart from manual ones.
- Drained nodes come back only when an admin runs `scontrol update nodename=<n> state=resume` after fixing the fault. `ReturnToService=1` returns nodes that were `DOWN` from lost contact, not nodes that were drained.

---

## 11. Failure handling

| Failure | What Slurm does | What happens to work | Recovery |
|---|---|---|---|
| Training job reaches its time limit | Sends `SIGUSR1` `<lead>` s early. The job exits 75 and is requeued. | None lost. Resumes from the checkpoint. | Automatic |
| Training job crashes (exit not 0 or 75) | Job ends `FAILED`, no requeue | Work since the last periodic checkpoint is lost | User fixes and resubmits with `--resume auto` |
| `train-01` fails during a job | Node marked `DOWN` after `SlurmdTimeout`. Job requeued (`--requeue`, `JobRequeue=1`). | Work since the last periodic checkpoint is lost (bounded by checkpoint frequency) | Job waits until the node returns or is replaced ([ADR-03](#adr-03-training-node-failure-the-job-waits)) |
| `eval-01` fails | Same as above, for eval jobs | Partial eval results lost | Jobs requeued and wait for the node |
| Health check fails at job start | Prolog fails. Node drained, job requeued, not held. | No work lost (the job had not started) | Admin fixes and resumes the node |
| Health check fails on a timer | Script drains the node. Running jobs continue. | None while the job runs | Admin decides whether to cancel the running job, then fixes and resumes the node |
| `slurmctld` stops | Running jobs keep running. New submissions fail. | None | Restart. State is read from `StateSaveLocation`. |
| `slurmdbd` or MariaDB stops | `slurmctld` keeps scheduling and buffers accounting records. Limits stay enforced from its cached associations. Account changes fail. | None | Restart. Buffered records are sent. |
| `ctl-01` is lost | No scheduling until restored. Running jobs continue as long as their nodes are up. | None while jobs run. Jobs finishing during the outage are recorded after restore. | Restore from snapshot and database dump ([§3](#3-slurm-components)) |
| Shared storage unavailable | Jobs fail at checkpoint save or load | Up to one checkpoint interval | Fix storage. Resubmit or wait for requeue. |

---

## 12. Test plan

All tests run on `ctl-01`, `train-01` and `eval-01`. `test_limits.sh` automates T01 to T11. T12 to T17 are run by hand and documented.

The expected reason strings below come from the Slurm documentation. Record the actual string in the evidence. If it differs, note it rather than treating the test as failed.

| ID | Test | Expected result |
|---|---|---|
| T01 | `sinfo -N -o "%N %P %G %t"` and `slurmd -C` on each node | Both nodes listed with `gpu:G`, state `idle` |
| T02 | `sacctmgr show assoc format=Account,User,QOS,DefaultQOS,GrpTRES,Fairshare` | Matches [§4](#4-accounts-and-users) and [§6](#6-limits-and-qos) |
| T03 | `tr_user1` submits a 1-GPU job to `train` | Runs on `train-01` |
| T04 | `ev_user1` submits a 1-GPU job to `eval` | Runs on `eval-01` |
| T05 | `tr_user1` submits with `--account=eval` | Rejected at submit (invalid account or account/partition combination) |
| T06 | `ev_user1` submits to partition `train` | Rejected at submit (account not allowed in partition) |
| T07 | **GPU limit.** Set the `limits-test` partition `UP`. Add `limits_test` to the `train` account's QOS list. Set `train` `GrpTRES=gres/gpu=1`. Submit two 1-GPU jobs with `--qos=limits_test`. | First runs. Second pends with `AssocGrpGRES` while the other node's GPU is idle. |
| T07b | Control: raise `train` `GrpTRES` to `gres/gpu=2` | Second job starts |
| T08 | **Per-job GPU limit.** With `GrpTRES=gres/gpu=2`, submit `--qos=debug --nodes=2 --gres=gpu:1` to `limits-test` | Pends with `QOSMaxGRESPerJob` |
| T09 | Submit to `train` with `--time` above the `train` QOS MaxWall | Pends with `QOSMaxWallDurationPerJobLimit`, or is rejected. Record which. |
| T10 | `tr_user1` and `tr_user2` each submit a `train` QOS job | Second pends with a QOS max-jobs-per-account reason |
| T11 | `tr_user2` runs `scancel` on `tr_user1`'s job | Refused. The owner can cancel. |
| T12 | **Self-requeue.** Job with `--time=00:05:00 --signal=B:USR1@60`. The script traps USR1, writes a marker file and exits 75. | Job requeued with the same ID. `Restarts` goes up by one. The second run sees the marker. Repeat for 3 cycles and check that no requeue cap stops it. |
| T12b | **Self-requeue through the DeepSpeed launcher.** Same as T12, using the template in [§8](#8-time-limits-checkpoints-and-requeue) with a dummy `train.py` that writes its PID file, sleeps, and on `SIGUSR1` writes a marker and exits 75 | `sacct` shows exit code 75 and the job requeued. The launcher processes were not killed by the signal (job log shows a normal exit with code 75). |
| T13 | **Node failure.** Stop `slurmd` on `train-01` (or stop the instance) during a `--requeue` job | Node `DOWN`. Job requeued and pending until the node returns, then runs. |
| T14 | **Prolog health failure.** Force `launch.sh --quick` to fail with a test flag, then submit a job | Node drained with a `health:` reason. Job requeued and pending, not held. |
| T15 | **Timed health failure.** Force `health_quick.sh` to fail | Node drained within `HealthCheckInterval` |
| T16 | Restart `slurmctld` while a job runs | Job keeps running and completes. `sacct` shows it. |
| T17 | Stop `slurmdbd`, run a job, start `slurmdbd` | Job runs. `sacct` shows it after `slurmdbd` returns. |
| T18 | (Optional) Fairshare. `tr_user1` uses GPU time, then both users queue jobs. | `sshare -l` and `sprio -l` show `tr_user2` with higher priority |

After T07 and T08, set `limits-test` back to `INACTIVE`, remove `limits_test` from the `train` account, and restore `GrpTRES=gres/gpu=G`.

### Evidence

For each test, save the following to `evidence/<test-id>/`:

- The exact commands, the user who ran them, the job IDs and timestamps.
- Output of `squeue`, `scontrol show job <id>`, `sacct -j <id> --format=JobID,User,Account,QOS,Partition,State,ExitCode,Restarts,Elapsed`, and, where relevant, `sshare -l`, `sprio -l` and `scontrol show assoc_mgr`.
- The observed result next to the expected result.

### What Week 1 does not prove

| Area | Proven in Week 1 | Needs multi-GPU nodes |
|---|---|---|
| Accounts, QOS, partition access | Yes | No |
| GPU limits | Yes, across two 1-GPU nodes on `limits-test` | Repeat on one node (e.g. limit 4, 6-GPU job) |
| GPU device isolation between jobs | No (one GPU per node) | Two eval jobs on `eval-01`, each sees only its GPUs |
| Self-requeue and signal | Yes, with a dummy job and through the DeepSpeed launcher | With real multi-rank checkpoint saves and the barrier before exit |
| Signal lead time | Formula only | Measured save time |
| NCCL and NVLink health | NCCL runs only. The T4 has no NVLink, so the NVLink check reports `skipped`. | Bandwidth thresholds and NVLink status |
| `CUDA_VISIBLE_DEVICES` passed through the DeepSpeed launcher | Trivial with one GPU (`0`) | Check that a job given GPUs 2 and 3 runs ranks on exactly those GPUs |

---

## 13. Decisions

Each decision is short and has status *Proposed* until the admin approves this document.

### ADR-01: One partition per node

- **Context:** Training runs on one node and eval on another. The PDF design split partitions by job length (`debug`/`train`) over shared nodes.
- **Decision:** Partition `train` on `train-01` and partition `eval` on `eval-01`, each limited by `AllowAccounts`.
- **Alternatives:** (a) One partition over both nodes, with separation by account limits only. Easy to misconfigure, and an eval job could land on the training node. (b) The PDF's length-based partitions on shared nodes. Training and eval would compete for the same GPUs.
- **Consequences:** The node boundary does most of the isolation. GPU limits on accounts act only as a backstop. A temporary test partition is needed to prove those limits (T07).

### ADR-02: Two accounts, one account per user

- **Context:** There are two teams, with two users each for now.
- **Decision:** Accounts `train` and `eval`. Each user belongs to exactly one.
- **Alternatives:** Let users belong to both accounts. That's useful for someone who trains and evaluates, but it makes the cross-account tests (T05, T06) depend on which default account a user has.
- **Consequences:** Clean isolation and simple tests. A user who needs both roles gets a second association later, and this ADR is revisited.

### ADR-03: Training node failure, the job waits

- **Context:** There is one training node, so a requeued training job has no other training node to run on.
- **Decision:** The requeued job pends until `train-01` is repaired or replaced, then resumes from the checkpoint.
- **Alternatives:** (a) Add `eval-01` to the `train` partition with a higher `Weight` and let training preempt eval there. That needs identical GPU types and a preemption policy. (b) Replace the instance automatically with a launch template or ASG. Planned for Week 2 or 3.
- **Consequences:** No work is lost beyond the last checkpoint, but wall-clock time is lost while the node is down. Option (b) is the likely next step.

### ADR-04: Controller and database on a separate CPU instance

- **Context:** If `slurmctld` runs on the training node, a node failure also stops the scheduler that should requeue the job.
- **Decision:** `slurmctld`, `slurmdbd` and MariaDB run on `ctl-01`, a small CPU instance.
- **Alternatives:** Run them on `train-01` or `eval-01`. Cheaper, but a node failure would also take out scheduling, and the T13 test would no longer mean anything.
- **Consequences:** A small extra cost. Backups are needed for `ctl-01` ([§3](#3-slurm-components)).

### ADR-05: Checkpoint on SIGUSR1 and self-requeue with RequeueExit

- **Context:** Training runs are longer than any sensible MaxWall, and hitting the time limit must not lose work.
- **Decision:** `--signal=B:USR1@<lead>`. The batch script forwards the signal to the ranks, the ranks checkpoint, and the job exits 75. `RequeueExit=75` requeues the job with the same ID.
- **Alternatives:** (a) `--dependency=afterany` chains. A new job ID per segment, more scripting, and a chain breaks if one link is cancelled. (b) Rely only on periodic checkpoints and accept losing work at the time limit.
- **Consequences:** One job ID for the whole run, with `Restarts` counting segments. Exit code 75 is reserved. The training code must handle `SIGUSR1` on every rank (see `checkpoint_design.md`). How the signal reaches the ranks under the DeepSpeed launcher is covered by ADR-10.

### ADR-06: Health checks in the Prolog and on a timer

- **Context:** Bad nodes must stay out of scheduling, and a node can fail between jobs.
- **Decision:** A Prolog runs `launch.sh --quick` before each job. `HealthCheckProgram` runs the same quick check every 300 s. `--full`, which includes NCCL, runs only at node bring-up.
- **Alternatives:** Only `HealthCheckProgram`, which misses a fault that appears just before a job starts. Only the Prolog, which misses faults while the node is idle.
- **Consequences:** A short delay at every job start. The quick check must stay fast.

### ADR-07: No preemption

- **Context:** The teams use separate nodes, and training jobs are expensive to interrupt.
- **Decision:** `PreemptType=preempt/none`. Debug jobs get a higher QOS priority but wait for running jobs.
- **Alternatives:** QOS-based preemption, with debug preempting train and requeueing it. Possible later, because training already handles requeue.
- **Consequences:** A debug job on a busy node waits for the running job to finish or requeue.

### ADR-08: Test on real EC2 nodes, not Docker

- **Context:** Docker was suggested because multiple machines might not be available. We now have two GPU instances.
- **Decision:** All tests run on `ctl-01`, `train-01` and `eval-01`.
- **Alternatives:** A Docker cluster with count-only fake GPUs. Cheaper, but it doesn't test real GRES, cgroup device limits or the health scripts.
- **Consequences:** GPU limits are tested across two 1-GPU nodes on a test partition (T07). Instance costs apply while testing.

### ADR-09: Debug jobs use a QOS, not a partition

- **Context:** Users need short test jobs, but each node already belongs to a team's partition.
- **Decision:** A `debug` QOS (30 min, 1 GPU, 1 running job per user, higher priority), allowed on both partitions.
- **Alternatives:** A separate `debug` partition over both nodes. That would let a team's debug jobs run on the other team's node.
- **Consequences:** Debug jobs stay on the team's own node and wait if a long job holds all of its GPUs.

### ADR-10: DeepSpeed launcher in the batch step, signal forwarded through PID files

- **Context:** Training uses DeepSpeed and its launcher on one node. The launcher's two processes don't handle `SIGUSR1` and would die if they received it ([§8](#8-time-limits-checkpoints-and-requeue)).
- **Decision:** Run `deepspeed` directly in the batch script, without `srun`. Use `--signal=B:USR1@<lead>` so only the batch shell is signalled. The batch shell forwards the signal to the rank PIDs that `train.py` writes at startup.
- **Alternatives:** (a) `srun deepspeed ...` with the signal sent to the step. The signal would reach the launchers and kill them. (b) Skip the launcher and start one rank per GPU with `srun python train.py`, setting `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR` and `MASTER_PORT` from Slurm's variables before `deepspeed.initialize`. Every rank then gets the signal directly with no PID files, but the team gives up the DeepSpeed launcher's environment setup.
- **Consequences:** `train.py` must write PID files and use a barrier before exiting. The launcher's behavior is tied to the DeepSpeed version, so T12b is rerun after any DeepSpeed upgrade. Alternative (b) is the fallback if the launcher causes problems on multi-GPU nodes.

---

## 14. Open questions

| # | Question | Needed for | Owner |
|---|---|---|---|
| 1 | g4dn size for Week 1 (vCPUs and RAM; `g4dn.xlarge` may be too small for data loading), production instance type, and the GPU count `G` | Node definitions, QOS limits | TBD |
| 2 | Checkpoint save time from the config v0 estimate | Signal lead time ([§8](#8-time-limits-checkpoints-and-requeue)) | Checkpoint workstream |
| 3 | DeepSpeed version to pin. The launcher's signal and exit-code behavior was checked on `master` and must be rechecked on the pinned version (T12b). | ADR-10 | Checkpoint workstream |
| 4 | Shared storage for Week 1 (FSx for Lustre or EFS) | Checkpoints readable by eval | TBD |
| 5 | Final MaxWall values for `train` and `eval` | QOS ([§6](#6-limits-and-qos)) | TBD |
| 6 | Slurm version | All hosts | TBD |
| 7 | User names and UIDs | [§4](#4-accounts-and-users) | Admin |
| 8 | Whether any requeue cap applies to `RequeueExit` cycles | ADR-05, checked in T12 | Queue workstream |
| 9 | Whether to adopt automatic node replacement (ADR-03 option b) | Week 2 or 3 | TBD |
| 10 | On-demand or Spot instances. Spot interruptions would need the same requeue path with a two-minute warning. | Cost, ADR-03 | TBD |

---

## 15. Admin approval

| Field | Value |
|---|---|
| Approver | |
| Date | |
| Version approved | |
| Decision | Approved / Approved with changes / Rejected |
| Comments | |
