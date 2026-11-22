# Training on Google Colab's free T4

`scripts/colab_run.py` is a second free GPU provider next to `scripts/kaggle_run.py`. It works the same way: the
code is embedded in the job script (the repo stays private), the dataset is regenerated from its seeds on the VM
(sha256 lines in the log let you check it matches), and the outputs (`runs/<name>/{log.csv,summary.json,stdout.txt,
model.pt}` and the results CSVs) are downloaded and merged into `runs/` and `results/`.

It drives Colab through Google's official CLI ([google-colab-cli](https://github.com/googlecolab/google-colab-cli),
version 0.8.0, the latest on PyPI on 22 Nov 2026). It only uses `colab new`, `exec`, `download`, `status` and `stop`;
it never uses `colab ssh`, because free runtimes forbid remote shells.

## One-time setup

### 1. Install the CLI (separate venv, the project's `.venv` is untouched)

From the repo root:

```bash
uv venv .colab-venv
uv pip install --python .colab-venv/bin/python google-colab-cli==0.8.0 keyrings.alt
.colab-venv/bin/colab version
```

`keyrings.alt` is there because this laptop (WSL) has no OS keyring. Without it the CLI can't keep its sign-in and
asks you to sign in again on every command. The runner points the CLI at `keyrings.alt`'s file store
(`~/.local/share/python_keyring/keyring_pass.cfg`, mode 600). That file holds your Google refresh token in **plain
text**, so treat it like a password. To use a different keyring, set `PYTHON_KEYRING_BACKEND` yourself.

### 2. Sign in with your own Google account

The CLI's built-in sign-in waits for you to paste a code into the terminal, which a `! command` in Claude Code can't
do. So the runner splits the same sign-in (the CLI's own Google client and permissions) into two commands. Type these
in the Claude Code prompt, from the repo root:

```
! python3 scripts/colab_run.py login
```

1. Open the URL it prints (your phone is fine) and sign in with **your** Google account.
2. Approve all the permissions it asks for. They are the CLI's standard set: your name and email, Google Cloud,
   Colab, and Drive files the app creates.
3. Google shows an authorization code. Copy it and run:

```
! python3 scripts/colab_run.py login --code PASTE_THE_CODE_HERE
! python3 scripts/colab_run.py check
```

`check` runs `colab sessions` and `colab usage`. If both work, you're done. The code only works once and expires
after a few minutes; if it's rejected, run `login` again.

If you're at a normal terminal, you can use the CLI's own sign-in instead (it asks for the code interactively):

```bash
PYTHON_KEYRING_BACKEND=keyrings.alt.file.PlaintextKeyring .colab-venv/bin/colab sessions
```

To sign out, revoke the CLI's access at https://myaccount.google.com/permissions and delete the token:

```bash
PYTHON_KEYRING_BACKEND=keyrings.alt.file.PlaintextKeyring .colab-venv/bin/python -c \
  "import keyring; keyring.delete_password('google-colab-cli', 'oauth_token')"
```

## Running jobs

Same job format as Kaggle: `"name=train args"`. Any `python3` works for the runner; it only needs the standard
library and calls the CLI in `.colab-venv`.

```bash
python3 scripts/colab_run.py push  --slug lp-lam --jobs "lam0.01_s0=--lam 0.01 --seed 0" "lam0.03_s0=--lam 0.03 --seed 0"
python3 scripts/colab_run.py wait  --slug lp-lam      # polls every 60 s, downloads, stops the VM
python3 scripts/colab_run.py merge --slug lp-lam      # copies into runs/ and results/
```

- `push` allocates a T4 (`colab new -s <slug> --gpu T4`) and starts the job script in the background
  (`colab exec ... --timeout 0`). It returns after about 20 s. Add `--scene push` for the push-the-block scene, and
  `--dry-run` to only write `colab_jobs/<slug>/script.py` without touching Colab.
- On the VM the jobs run one after another in a single notebook cell. After every job the outputs are packed into
  `/content/lp_out.tar.gz`, so a VM that gets cut off still leaves the finished jobs to download. The current step
  (and the last line of the training log, refreshed every minute) is in `/content/lp_status.txt`.
- `wait` prints that status line each poll. When it reads `ALL_JOBS_DONE` or `FAILED ...`, it downloads the tarball
  into `colab_jobs/<slug>/out/` and stops the VM. `--keep` leaves the VM running. If `wait` runs out of time
  (`--timeout`, default 12 h), it downloads what's there so far and leaves the VM up; run `wait` again later.
- `merge` never overwrites an existing `runs/<name>` and only appends CSV rows that aren't there yet. It also copies
  the VM's log to `results/colab_<slug>.log`.
- `stop --slug X` releases the VM by hand. Always stop a VM you're done with.

Local files per job, in `colab_jobs/<slug>/` (git-ignored): `script.py`, `jobs.json`, `exec.log` (the cell's live
output), `exec.pid`, `status.txt`, `lp_out.tar.gz`, `out/`.

The laptop doesn't need to stay connected while the job runs: the cell keeps running on the VM if the local
`colab exec` process stops. But the outputs live only on the VM until `wait` downloads them, and Colab deletes a VM
some time after its code finishes. So run `wait` (or at least check in) before the VM is reclaimed.

Handy CLI commands (prefix with `PYTHON_KEYRING_BACKEND=keyrings.alt.file.PlaintextKeyring .colab-venv/bin/`):
`colab sessions` (what's running), `colab status -s <slug>`, `colab url -s <slug>` (opens the VM in the Colab web
page), `colab stop -s <slug>`.

## Quotas and terms

From the [Colab FAQ](https://research.google.com/colaboratory/faq.html):

- The free tier gives a T4 **when one is available**. Limits are dynamic and not published, and a request can be
  refused. `push` then says it couldn't get a T4; try again later.
- A VM runs **at most 12 hours**, often less, depending on availability and your usage. Idle timeouts only start
  once code stops running. "Colab prioritizes interactive compute", so long unattended runs may be cut short. Keep
  a push to a few hours of jobs.
- Free runtimes "may be terminated at any time without warning" and must not be used for: remote control such as
  SSH shells or remote desktops, bypassing the notebook UI to interact mainly through a web UI, chess training, or
  running distributed computing workers.
- **One account per person.** All runtimes ban "using multiple accounts to work around access or resource usage
  restrictions". Use only your own Google account. Don't create or rotate extra accounts to stretch the free GPU
  time. Kaggle is a separate service with its own account, so using both providers is fine.
- `colab usage` shows your compute-unit balance. A free account has none, which is expected.

## Not verified yet

Nothing has been run against a real Colab VM, since that needs your sign-in. Checked so far: the CLI installs and
runs (`colab version` 0.8.0), every flag the runner passes exists in `--help`, the runner compiles, and `push
--dry-run` builds a job script that compiles. That script was also run end to end on the laptop's GPU with its
`/content` paths pointed at a temp folder (data generation, a short training job, eval, packing). The sign-in URL
step works, and a bad code gets a clean error. Still open:

1. Whether a free account gets a T4 through the CLI. The CLI's docs neither confirm nor rule it out, and its own
   notes say most accounts may only get CPU. Third-party reviews say free T4s work when available.
2. Whether Colab keeps a free VM running for hours when no browser tab is open. The CLI says the backend keeps a
   session alive while its kernel is busy, and the job keeps the kernel busy the whole time.
3. That the cell keeps running after the local `colab exec` process exits (normal Jupyter behaviour, not tested
   on Colab).
4. Whether Google treats CLI-driven runs on a free account as interactive use. The FAQ doesn't mention the CLI. It
   is Google's own tool and the runner avoids SSH, but long unattended runs are the kind of use Colab says it
   deprioritises.
5. Downloading a big tarball (about 5 MB per run, so about 50 MB for 10 jobs). The CLI sends it as one base64
   response.
6. The two-step sign-in reuses internals of CLI 0.8.0 (`colab_cli.auth`'s client, scopes and keyring names). A CLI
   update could move them; the terminal sign-in above is the fallback. The GitHub docs already describe behaviour
   newer than 0.8.0 (for example `--timeout`); the runner passes `--timeout 0` (no limit) either way.
7. Colab's preinstalled torch and numpy versions differ from `requirements.txt`. Compare the `SHA256` lines in
   `results/colab_<slug>.log` with the local data, and expect small numeric differences from local or Kaggle runs.
