#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
run_dir="${1:-$PWD/all_series_reproduction}"
python_bin="${PYTHON_EXECUTABLE:-python}"
export PYTHONPATH="$script_dir/vendor${PYTHONPATH:+:$PYTHONPATH}"
"$python_bin" "$script_dir/test_confidence_transport.py"
"$python_bin" "$script_dir/prepare_inputs.py" --work "$run_dir"
cp "$script_dir/registry_non_Q1.json" "$run_dir/registry_non_Q1.json"
"$python_bin" "$script_dir/validate_inputs.py" --work "$run_dir"
"$python_bin" "$script_dir/infer_roma.py" --work "$run_dir" --positions Q1-1-B2 Q1-1-Z3 Q1-2-B3 Q1-2-Z2 Q1-2-Z3 Q1-3-Z3
"$python_bin" "$script_dir/localize_q1.py" --work "$run_dir"
"$python_bin" "$script_dir/infer_roma.py" --work "$run_dir" --positions Q1-3-Z3 --extra-pair T1_to_T3
"$python_bin" "$script_dir/assemble_q1.py" --work "$run_dir"
"$python_bin" "$script_dir/render_results.py" --work "$run_dir" --registry "$run_dir/registry_all.json"
"$python_bin" "$script_dir/infer_roma.py" --work "$run_dir" --view posed
"$python_bin" "$script_dir/render_results.py" --work "$run_dir" --registry "$run_dir/registry_all.json" --confidence
"$python_bin" "$script_dir/export_partial_q1_3.py" --work "$run_dir"
"$python_bin" "$script_dir/summary_figure.py" --work "$run_dir"
"$python_bin" "$script_dir/finish_report.py" --work "$run_dir"
"$python_bin" "$script_dir/verify_results.py" --work "$run_dir"
"$python_bin" "$script_dir/package_results.py" --work "$run_dir" --scripts "$script_dir"
