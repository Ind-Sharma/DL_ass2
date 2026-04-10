"""Minimal W&B login/auth-check helper for report scripts."""

from __future__ import annotations

import argparse
import os
from getpass import getpass

import wandb

# Optional: paste your key here (kept empty by default).
# Example: USER_API_KEY = "wandb_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
USER_API_KEY = ""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=str, default="da6401-ass2")
    p.add_argument("--entity", type=str, default="")
    p.add_argument("--name", type=str, default="wandb-login-check")
    p.add_argument("--api_key_env", type=str, default="WANDB_API_KEY")
    p.add_argument(
        "--api_key",
        type=str,
        default="",
        help="Optional direct API key. If set, it overrides file/env/prompt.",
    )
    args = p.parse_args()

    api_key = args.api_key.strip() or USER_API_KEY.strip() or os.environ.get(args.api_key_env, "").strip()
    if api_key:
        wandb.login(key=api_key)
    else:
        # Prompt in terminal (hidden input) to avoid hardcoding keys in code.
        entered = getpass("Enter W&B API key (input hidden): ").strip().strip('"').strip("'")
        if entered:
            if len(entered) < 40:
                print(
                    f"Provided key looks too short ({len(entered)} chars). "
                    "Falling back to interactive W&B login instead."
                )
                wandb.login()
            else:
                wandb.login(key=entered)
        else:
            # Falls back to normal interactive login prompt if user presses enter.
            wandb.login()

    run = wandb.init(
        project=args.project,
        entity=(args.entity.strip() or None),
        name=args.name,
        config={"purpose": "auth-check"},
    )
    run.log({"auth_check": 1})
    run.finish()
    print("W&B login + test run successful.")


if __name__ == "__main__":
    main()

