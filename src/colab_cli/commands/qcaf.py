from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import nbformat
import typer
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
from typing_extensions import Annotated

ALLOWED_FRAMEWORKS = {"qcaf", "pennylane", "qiskit", "hybrid"}


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip()).strip("_")
    return slug or "qcaf_experiment"


def _read_request(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise typer.BadParameter(f"Request file not found: {path}")
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"Request file is not valid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise typer.BadParameter("QCAF request must be a JSON object")
    observations = payload.get("observations")
    if not isinstance(observations, list) or not observations:
        raise typer.BadParameter(
            "QCAF request must contain a non-empty observations list"
        )
    return payload


def build_qcaf_notebook(
    payload: dict,
    *,
    title: str,
    framework: str,
    repository: str,
    repository_ref: str,
):
    framework = framework.lower().strip()
    if framework not in ALLOWED_FRAMEWORKS:
        raise ValueError(
            f"framework must be one of: {', '.join(sorted(ALLOWED_FRAMEWORKS))}"
        )

    extras = []
    if framework in ("pennylane", "hybrid"):
        extras.append("pennylane")
    if framework in ("qiskit", "hybrid"):
        extras.extend(["qiskit", "qiskit-aer"])

    git_url = f"git+https://github.com/{repository}.git@{repository_ref}"
    packages = " ".join([git_url, *extras])
    payload_json = json.dumps(payload, indent=2, ensure_ascii=False)
    payload_literal = repr(payload_json)

    cells = [
        new_markdown_cell(
            f"""# {title}

This Google Colab notebook runs QCAF — Quantum Constraint Admissibility
Framework — over a finite, explicitly supplied set of quantum observations.

An INADMISSIBLE result applies only to this bounded experimental slice and is
not a proof of global physical or mathematical impossibility.

Framework profile: {framework}.
"""
        ),
        new_code_cell(
            f"""# @title Install QCAF and scientific dependencies
!apt-get -qq update >/dev/null
!apt-get -qq install -y minizinc >/dev/null
%pip -q install {packages}
print("QCAF environment ready.")
"""
        ),
        new_code_cell(
            """# @title Load QCAF
import json
from pprint import pprint

from main import quantum_admissibility_core
from qcaf import QuantumAdmissibilityRequest
"""
        ),
        new_code_cell(
            f"""# @title Reproducible QCAF request
request_payload = json.loads({payload_literal})
qcaf_request = QuantumAdmissibilityRequest(**request_payload)
pprint(request_payload)
"""
        ),
    ]

    if framework in ("pennylane", "hybrid"):
        cells.append(
            new_markdown_cell(
                """## PennyLane adapter

Generate the declared parameter/noise grid with PennyLane and append each
reproducible observation to request_payload["observations"]. Preserve backend,
seed, circuit hash, package versions and variational parameters where possible.
"""
            )
        )

    if framework in ("qiskit", "hybrid"):
        cells.append(
            new_markdown_cell(
                """## Qiskit adapter

Generate counts or state metrics with Qiskit/Aer, derive target_probability,
record transpiled circuit depth, shots, noise settings and provenance, then
append each run to request_payload["observations"].
"""
            )
        )

    cells.extend(
        [
            new_code_cell(
                """# @title Solve bounded quantum admissibility
result = await quantum_admissibility_core(qcaf_request)
result_data = result.model_dump() if hasattr(result, "model_dump") else result.dict()
pprint(result_data)
"""
            ),
            new_code_cell(
                """# @title Publication-oriented interpretation
print("Admissibility:", result.admissibility)
print("Solver status:", result.solver_status)
print("Objective:", result.objective)
print("Interpretation boundary:")
print(result.interpretation_boundary)

if result.selected_observation is not None:
    print("\\nSelected witness:")
    selected = (
        result.selected_observation.model_dump()
        if hasattr(result.selected_observation, "model_dump")
        else result.selected_observation.dict()
    )
    pprint(selected)
"""
            ),
        ]
    )

    notebook = new_notebook(cells=cells)
    notebook.metadata["colab"] = {
        "name": f"{_slugify(title)}.ipynb",
        "provenance": [],
    }
    notebook.metadata["qcaf"] = {
        "framework": framework,
        "repository": repository,
        "repository_ref": repository_ref,
    }
    return notebook


def qcaf_notebook_command(
    request: Annotated[
        Path,
        typer.Argument(help="Path to a QCAF request JSON file."),
    ],
    output: Annotated[
        Optional[Path],
        typer.Option("-o", "--output", help="Output .ipynb path."),
    ] = None,
    title: Annotated[
        str,
        typer.Option("--title", help="Notebook title."),
    ] = "QCAF Quantum Admissibility Experiment",
    framework: Annotated[
        str,
        typer.Option(
            "--framework",
            help="Notebook integration profile: qcaf, pennylane, qiskit, or hybrid.",
        ),
    ] = "hybrid",
    repository: Annotated[
        str,
        typer.Option("--repository", help="QCAF GitHub repository owner/name."),
    ] = "MarceloClaro/minizinc-mcp",
    repository_ref: Annotated[
        str,
        typer.Option("--ref", help="Git branch, tag, or commit to install."),
    ] = "main",
    execute: Annotated[
        bool,
        typer.Option("--execute", help="Execute the generated notebook in Colab."),
    ] = False,
    session: Annotated[
        Optional[str],
        typer.Option("-s", "--session", help="Colab session name for --execute."),
    ] = None,
    timeout: Annotated[
        float,
        typer.Option("--timeout", help="Per-cell execution timeout in seconds."),
    ] = 120.0,
):
    """Generate a reproducible QCAF Google Colab notebook."""

    payload = _read_request(request)
    framework = framework.lower().strip()
    if framework not in ALLOWED_FRAMEWORKS:
        raise typer.BadParameter(
            f"--framework must be one of: {', '.join(sorted(ALLOWED_FRAMEWORKS))}"
        )

    notebook = build_qcaf_notebook(
        payload,
        title=title,
        framework=framework,
        repository=repository,
        repository_ref=repository_ref,
    )

    output_path = output or Path(f"{_slugify(title)}.ipynb")
    if output_path.suffix.lower() != ".ipynb":
        raise typer.BadParameter("--output must end with .ipynb")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        nbformat.write(notebook, handle)

    typer.echo(f"[colab] QCAF notebook written to '{output_path}'.")

    if execute:
        from colab_cli.commands.execution import exec_command

        exec_command(
            session=session,
            file=str(output_path),
            output_image=None,
            timeout=timeout,
            env=None,
        )


def register(app: typer.Typer) -> None:
    app.command(name="qcaf-notebook")(qcaf_notebook_command)
