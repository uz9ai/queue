"""Reconcile Slurm accounting (cluster, QOS, accounts, users) with cluster.toml.

Adds what is missing and sets every managed attribute to the TOML value
(unset limits become -1, which clears them). Objects that exist in Slurm but
not in the TOML are reported, never deleted.
"""

from __future__ import annotations

from ..util import Ctx, InstallError, header, info, ok, out, run, warn

BUILTIN_QOS = {"normal"}
BUILTIN_ACCOUNTS = {"root"}
BUILTIN_USERS = {"root"}


def _sacctmgr(ctx: Ctx, *args: str) -> None:
    cp = run(ctx, ["sacctmgr", "-i", *args], check=False)
    text = (cp.stdout or "") + (cp.stderr or "")
    if cp.returncode != 0 and "Nothing modified" not in text and "Nothing new added" not in text:
        raise InstallError(f"sacctmgr {' '.join(args)} failed ({cp.returncode}): {text.strip()}")


def _list(ctx: Ctx, *args: str) -> list[list[str]]:
    text = out(ctx, ["sacctmgr", "-n", "-P", *args])
    return [line.split("|") for line in text.splitlines() if line.strip()]


def _int_or_clear(v) -> str:
    return "-1" if v is None else str(v)


def _tres_set(ctx: Ctx, tres: dict, extra_keys: set[str]) -> str:
    """Full TRES assignment: TOML values, every other known TRES cleared."""
    known = {"cpu", "mem", "node"} | set(ctx.cfg["slurm"]["accounting_tres"]) | extra_keys
    parts = {k: "-1" for k in known}
    parts.update({k: str(v) for k, v in tres.items()})
    return ",".join(f"{k}={v}" for k, v in sorted(parts.items()))


def accounts(ctx: Ctx) -> None:
    header("Slurm accounting: cluster, QOS, accounts, users")
    cfg = ctx.cfg
    cluster = cfg["cluster"]["name"]

    # -- cluster -------------------------------------------------------------------
    clusters = {row[0] for row in _list(ctx, "show", "cluster", "format=cluster")}
    if cluster not in clusters:
        _sacctmgr(ctx, "add", "cluster", cluster)
    else:
        ok(f"cluster {cluster} registered")

    tres_keys: set[str] = set()
    for q in cfg["qos"]:
        tres_keys |= set(q["max_tres_per_job"]) | set(q["max_tres_per_user"])
    for a in cfg["accounts"]:
        tres_keys |= set(a["grp_tres"])

    # -- QOS -------------------------------------------------------------------------
    existing_qos = {row[0] for row in _list(ctx, "show", "qos", "format=name")}
    for q in cfg["qos"]:
        if q["name"] not in existing_qos:
            _sacctmgr(ctx, "add", "qos", q["name"])
        settings = [
            f"Description={q['description'] or q['name']}",
            f"Priority={q['priority']}",
            f"MaxWall={q['max_wall'] or '-1'}",
            f"MaxTRESPerJob={_tres_set(ctx, q['max_tres_per_job'], tres_keys)}",
            f"MaxTRESPerUser={_tres_set(ctx, q['max_tres_per_user'], tres_keys)}",
            f"MaxJobsPU={_int_or_clear(q['max_jobs_pu'])}",
            f"MaxJobsPA={_int_or_clear(q['max_jobs_pa'])}",
            f"MaxSubmitJobsPU={_int_or_clear(q['max_submit_jobs_pu'])}",
            f"MaxSubmitJobsPA={_int_or_clear(q['max_submit_jobs_pa'])}",
        ]
        if q["flags"]:
            settings.append("Flags=" + ",".join(q["flags"]))
        _sacctmgr(ctx, "modify", "qos", "where", f"name={q['name']}", "set", *settings)
        ok(f"QOS {q['name']}")
    for name in sorted(existing_qos - {q['name'] for q in cfg['qos']} - BUILTIN_QOS):
        info(f"QOS {name!r} exists in Slurm but not in cluster.toml (left alone)")

    # -- accounts ----------------------------------------------------------------------
    existing_accts = {row[0] for row in _list(ctx, "show", "account", "format=account")}
    for a in cfg["accounts"]:
        if a["name"] not in existing_accts:
            _sacctmgr(ctx, "add", "account", a["name"], f"Cluster={cluster}",
                      f"Description={a['description'] or a['name']}",
                      f"Organization={a['organization'] or cluster}")
        else:
            _sacctmgr(ctx, "modify", "account", "where", f"name={a['name']}", "set",
                      f"Description={a['description'] or a['name']}",
                      f"Organization={a['organization'] or cluster}")
        settings = [
            f"Fairshare={a['fairshare']}",
            f"GrpTRES={_tres_set(ctx, a['grp_tres'], tres_keys)}",
            f"GrpJobs={_int_or_clear(a['grp_jobs'])}",
            f"MaxJobs={_int_or_clear(a['max_jobs'])}",
        ]
        if a["qos"]:
            settings.append("QOS=" + ",".join(a["qos"]))
        if a["default_qos"]:
            settings.append(f"DefaultQOS={a['default_qos']}")
        _sacctmgr(ctx, "modify", "account", "where", f"name={a['name']}", f"cluster={cluster}",
                  "set", *settings)
        ok(f"account {a['name']}")
    for name in sorted(existing_accts - {a['name'] for a in cfg['accounts']} - BUILTIN_ACCOUNTS):
        warn(f"account {name!r} exists in Slurm but not in cluster.toml (left alone)")

    # -- users ---------------------------------------------------------------------------
    assocs = {
        (row[0], row[1])
        for row in _list(ctx, "show", "assoc", "where", f"cluster={cluster}", "format=user,account")
        if len(row) >= 2 and row[0]
    }
    acct = {a["name"]: a for a in cfg["accounts"]}
    wanted = {(u["name"], u["account"]) for u in cfg["users"]}
    for u in cfg["users"]:
        a = acct[u["account"]]
        if (u["name"], u["account"]) not in assocs:
            _sacctmgr(ctx, "add", "user", u["name"], f"Account={u['account']}",
                      f"DefaultAccount={u['account']}", f"Cluster={cluster}")
        settings = [f"Fairshare={u['fairshare']}"]
        if a["qos"]:
            settings.append("QOS=" + ",".join(a["qos"]))
        if a["default_qos"]:
            settings.append(f"DefaultQOS={a['default_qos']}")
        _sacctmgr(ctx, "modify", "user", "where", f"name={u['name']}", f"account={u['account']}",
                  f"cluster={cluster}", "set", *settings)
        _sacctmgr(ctx, "modify", "user", "where", f"name={u['name']}", "set",
                  f"DefaultAccount={u['account']}", f"AdminLevel={u['admin_level']}")
        ok(f"user {u['name']} -> account {u['account']}")
    for user, account in sorted(assocs - wanted):
        if user in BUILTIN_USERS:
            continue
        warn(f"Slurm has user {user!r} in account {account!r}, which cluster.toml does not list (left alone)")
