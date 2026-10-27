#!/usr/bin/env bash
# Snapshot of cluster state for the admin. Run on the controller:
#   sudo ./tests/collect_evidence.sh [DIR]
# Default DIR: ./evidence/snapshot-<UTC timestamp>
set -uo pipefail

DIR="${1:-evidence/snapshot-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$DIR"

capture() {
  local name="$1"; shift
  {
    echo "# $(date -u +%FT%TZ) \$ $*"
    "$@" 2>&1
    echo "# exit=$?"
  } >"$DIR/$name.txt"
  echo "  $DIR/$name.txt"
}

echo "Writing evidence to $DIR"
capture versions        bash -c 'slurmctld -V; sacctmgr -V; munge --version 2>&1 | head -1'
capture ping            scontrol ping
capture sinfo           sinfo -N -o "%N %P %t %C %m %G %f %E"
capture nodes           scontrol show nodes
capture partitions      scontrol show partition
capture qos             sacctmgr show qos format=name,priority,maxwall,maxtrespj%30,maxtrespu%30,maxjobspu,maxjobspa,maxsubmitpu
capture associations    sacctmgr show assoc format=cluster,account,user,share,qos%30,defaultqos,grptres%30,grpjobs,maxjobs
capture users           sacctmgr show user format=user,defaultaccount,adminlevel
capture assoc_mgr       scontrol show assoc_mgr flags=assoc,qos
capture sshare          sshare -l
capture sprio           sprio -l
capture squeue          squeue -o "%.10i %.12u %.10a %.10P %.10q %.10T %.24R"
capture sacct_week      sacct -a -S now-7days -X -D --format=JobID,User,Account,Partition,QOS,State,ExitCode,Elapsed,NodeList
capture slurm_conf      cat /etc/slurm/slurm.conf
capture gres_conf       cat /etc/slurm/gres.conf
capture cgroup_conf     cat /etc/slurm/cgroup.conf
capture cluster_toml    bash -c 'sha256sum /etc/team7/cluster.toml; cat /etc/team7/cluster.toml'
echo "done"
