# Connecting the bot to Interactive Brokers

One-time setup. Do all of it against a **paper** account first; live
credentials are a separate registration at the end.

The bot authenticates with OAuth 1.0a, which is the only IBKR scheme that
works with no TWS and no IB Gateway running. That is what makes a scheduled
job on a GitHub runner possible at all.

Budget about 30 minutes, plus a wait: IBKR sometimes takes up to a weekend to
activate a new consumer key.

---

## 1. Generate the key material

Run this locally — not in a cloud session, and not in this repository.

```bash
mkdir -p ~/ibkr-oauth && cd ~/ibkr-oauth

# Signing key: signs every API request.
openssl genrsa -out private_signature.pem 2048
openssl rsa -in private_signature.pem -outform PEM -pubout -out public_signature.pem

# Encryption key: decrypts the access token secret IBKR issues you.
openssl genrsa -out private_encryption.pem 2048
openssl rsa -in private_encryption.pem -outform PEM -pubout -out public_encryption.pem

# Diffie-Hellman parameters for the live session token exchange.
openssl dhparam -out dhparam.pem 2048        # takes a minute or two

chmod 600 *.pem
```

Five files. The two `private_*.pem` files and `dhparam.pem` are secrets — they
never leave your machine except as GitHub Secrets.

## 2. Register with IBKR

1. Sign in at <https://www.interactivebrokers.com> → **Settings** → **API** →
   **Self-Service OAuth**.
2. Create a **consumer key**: exactly 9 characters, letters and digits.
   Letters are upper-cased. Write it down.
3. Upload `public_signature.pem`, `public_encryption.pem`, and `dhparam.pem`.
   - If the `dhparam.pem` upload returns a 403, the file has Windows line
     endings. Fix with `dos2unix dhparam.pem` and retry.
4. Generate the **access token** and **access token secret**. The secret is
   shown once and never again — copy it now.
5. Toggle **Enable OAuth Access** on.

Register the paper account separately from the live account, with its own key
pair. Reusing one key across both is the most common cause of confusing
authentication failures.

> Activation is not always immediate. If step 4 of the next section fails with
> an authentication error, wait a day and retry before assuming a mistake.

## 3. Extract the DH prime

The bot needs the prime from `dhparam.pem` as a hex string:

```bash
python3 -c "
from cryptography.hazmat.primitives.serialization import load_pem_parameters
with open('dhparam.pem','rb') as fh:
    params = load_pem_parameters(fh.read())
print(format(params.parameter_numbers().p, 'x'))
"
```

If `cryptography` is not installed, `pip install cryptography` first.

## 4. Store the credentials as GitHub Secrets

In the repository: **Settings** → **Secrets and variables** → **Actions**.

First create an environment named `trading` (**Settings** → **Environments**),
then add these secrets **to that environment**. The trading workflow declares
`environment: trading`, so nothing else in the repository can read them, and
you can require a manual approval on it later if you want a human in the loop.

| Secret | Value |
| --- | --- |
| `IBKR_ACCOUNT_ID` | Your account number. Paper accounts start with `DU`. |
| `IBKR_CONSUMER_KEY` | The 9-character key from step 2. |
| `IBKR_ACCESS_TOKEN` | From step 4 of the IBKR portal. |
| `IBKR_ACCESS_TOKEN_SECRET` | The secret shown once. |
| `IBKR_DH_PRIME` | The hex string from step 3. |
| `IBKR_SIGNATURE_KEY` | Full contents of `private_signature.pem`, including the BEGIN/END lines. |
| `IBKR_ENCRYPTION_KEY` | Full contents of `private_encryption.pem`. |

Paste the PEM files whole, newlines and all. GitHub preserves them correctly.

## 5. Verify the connection

Actions → **Trade** → **Run workflow** → mode `dry_run`.

A dry run authenticates, reads the account, and prints the orders it *would*
place without sending any. The log should show your account number, equity,
and account type.

To check the same thing locally:

```bash
export IBKR_ACCOUNT_ID=DU1234567
export IBKR_CONSUMER_KEY=...
export IBKR_ACCESS_TOKEN=...
export IBKR_ACCESS_TOKEN_SECRET=...
export IBKR_DH_PRIME=...
export IBKR_SIGNATURE_KEY="$(cat ~/ibkr-oauth/private_signature.pem)"
export IBKR_ENCRYPTION_KEY="$(cat ~/ibkr-oauth/private_encryption.pem)"

python -m bot.cli check-account
```

`check-account` reports what IBKR actually says: the entity carrying the
account, whether it is cash or margin, settled cash, open positions, and —
importantly — any position with no resting stop order.

## 6. Going live

Only after the paper account has run for several weeks and its results line up
with the backtest.

1. Register a second consumer key against the live account (step 2, new keys).
2. Replace the secrets in the `trading` environment with the live values.
3. Set the repository variable `SCHEDULED_EXECUTION_MODE` to `live` under
   **Settings** → **Secrets and variables** → **Actions** → **Variables**.

Live trading is enabled by that variable and nothing else. Editing
`config.yaml` cannot start it, which means no code change — by you, by me, or
by a merged pull request — can move the bot from paper to real money.

---

## Troubleshooting

**`Missing IBKR credentials: ...`** — the named environment variables are not
set. In Actions, check they are on the `trading` *environment*, not only at
repository level.

**Authentication fails right after registration** — activation can lag by up
to a weekend. Retry the next day before changing anything.

**403 uploading `dhparam.pem`** — line endings. `dos2unix dhparam.pem`.

**Signature errors** — the PEM lost its newlines. The bot restores literal
`\n` sequences automatically, but the cleanest fix is to re-paste the file
whole.

**Orders rejected for an ETF** — expected on an IBIE (Ireland) account. PRIIPs
blocks US-domiciled ETFs for EEA retail clients. The universe is single stocks
for exactly this reason; the bot reads SPY as a signal but refuses to submit an
order for it.

## If you need to stop the bot immediately

Fastest first:

1. **Actions** → **Trade** → **⋯** → **Disable workflow**. No further runs.
2. Set `SCHEDULED_EXECUTION_MODE` to `dry_run`. Runs continue but place no orders.
3. Close positions yourself in TWS or the IBKR mobile app.

Disabling the workflow stops the bot from *acting*. It does not cancel the
stop orders already resting at IBKR — those stay live and keep protecting open
positions, which is the intended behaviour.
