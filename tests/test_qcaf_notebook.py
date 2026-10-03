import json

import nbformat

from colab_cli.commands.qcaf import build_qcaf_notebook


def request_payload():
    return {
        "observations": [
            {
                "label": "run-001",
                "framework": "pennylane",
                "backend": "default.mixed",
                "target_probability": 0.94,
                "ber": 0.03,
                "noise_probability": 0.04,
                "circuit_depth": 12,
                "shots": 8192,
                "fidelity": 0.97,
                "seed": 42,
            }
        ],
        "constraints": {
            "min_target_probability": 0.90,
            "max_ber": 0.05,
            "max_noise_probability": 0.10,
            "max_circuit_depth": 20,
            "min_shots": 8192,
        },
        "objective": "satisfy",
    }


def test_build_qcaf_notebook_is_nbformat_v4():
    notebook = build_qcaf_notebook(
        request_payload(),
        title="QCAF Test",
        framework="hybrid",
        repository="MarceloClaro/minizinc-mcp",
        repository_ref="feature/qcaf-quantum-admissibility",
    )

    assert notebook.nbformat == 4
    assert notebook.metadata["colab"]["name"] == "QCAF_Test.ipynb"
    assert notebook.metadata["qcaf"]["framework"] == "hybrid"
    assert len(notebook.cells) >= 6

    serialized = nbformat.writes(notebook)
    loaded = nbformat.reads(serialized, as_version=4)
    assert loaded.nbformat == 4


def test_notebook_embeds_qcaf_request_and_install_ref():
    notebook = build_qcaf_notebook(
        request_payload(),
        title="QCAF",
        framework="qiskit",
        repository="MarceloClaro/minizinc-mcp",
        repository_ref="feature/qcaf-quantum-admissibility",
    )

    sources = "\n".join(cell.source for cell in notebook.cells)
    assert '"target_probability": 0.94' in sources
    assert "qiskit-aer" in sources
    assert "feature/qcaf-quantum-admissibility" in sources
    assert "not a proof of global" in sources
