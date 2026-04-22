# Accelerator4

This repository contains the SystemVerilog RTL and testbenches for a small HDC-FPGA inference accelerator.

## Vivado Minimal File List

Use the following files when creating a Vivado project.

### Design Sources

Add these files to `Design Sources`:

- `uart_rx.sv`
- `uart_tx.sv`
- `frame_ctrl.sv`
- `projection_engine.sv`
- `binarizer.sv`
- `centroid_bram.sv`
- `hamming_sim.sv`
- `argmax.sv`
- `hdc_top.sv`

### Simulation Sources

Add these files to `Simulation Sources`:

- `tb_uart_rx.sv`
- `tb_uart_tx.sv`
- `tb_frame_ctrl.sv`
- `tb_binarizer.sv`
- `tb_projection_engine.sv`
- `tb_hamming_argmax.sv`
- `tb_hdc_top.sv`
- `tb_ucihar_core.sv`

### Memory Init File

Add these files to `Simulation Sources` as well so XSim can find them at run time:

- `centroids_tb.mem`
- `weights_tb.mem`

If `centroids_tb.mem` is not added to the simulation set, `tb_hdc_top.sv` will fail when `centroid_bram.sv` calls `$readmemh`.

## Notes

- `tb_hdc_top.sv` is the end-to-end UART-to-classification testbench (UART frame in, 1-byte class ID out).
- `tb_ucihar_core.sv` is a no-UART core-level alignment testbench for UCIHAR-style vectors. It instantiates:
  `projection_engine -> binarizer -> centroid_bram -> hamming_sim -> argmax` and compares predicted `class_id` against a software-generated golden file.
- `hamming_sim.sv` exports a `busy` signal; `hdc_top.sv` includes a minimal buffer/handshake to avoid dropping an `hv_valid` pulse when the Hamming engine is still busy.

## Recommended Top Modules

- For synthesis, set top module to `hdc_top`
- For end-to-end simulation (UART), set top module to `tb_hdc_top`
- For software/hardware alignment (no UART), set top module to `tb_ucihar_core`
- For unit simulation, you can switch the simulation top to one of the module-level testbenches such as `tb_uart_rx`, `tb_uart_tx`, `tb_frame_ctrl`, `tb_binarizer`, `tb_projection_engine`, or `tb_hamming_argmax`

## Vivado GUI Setup

Use this flow when creating a new Vivado project:

1. Create a new RTL project and skip adding default sources during project creation if you want a clean manual setup.
2. Add the files listed under `Design Sources`.
3. Add the files listed under `Simulation Sources`.
4. Add the memory init files (`centroids_tb.mem`, `weights_tb.mem`) to `Simulation Sources` when running `tb_hdc_top`.
5. In the `Sources` window, right-click `hdc_top` and choose `Set as Top` for synthesis.
6. Open the `Simulation Sources` set, right-click `tb_hdc_top`, and choose `Set as Top` for simulation.
7. Run `Compile Order` update if Vivado asks for it.
8. Start simulation with `Run Simulation -> Run Behavioral Simulation`.

## Important Simulation Reminder

- XSim resolves `centroids_tb.mem` relative to the simulation run directory, not relative to `tb_hdc_top.sv` or `centroid_bram.sv`
- If `centroids_tb.mem` is missing from the simulation set, `centroid_bram.sv` will fail when `$readmemh` runs
- `tb_hdc_top.sv` uses a realistic UART rate of `115200`, so one 10-byte input frame takes about `868000 ns`
- The first top-level classification result appears close to `955000 ns`, so `run 1000ns` is far too short and `centroid_bram.rd_data` will usually still look idle
- For full-top simulation in Vivado, prefer `Run All` or run for at least `1000000 ns`

## UCIHAR Alignment Flow (No UART)

This flow is for verifying that the RTL core matches a fixed software reference model under a fixed quantization rule.

1. Generate three files from software (all placed beside the simulation run directory or added as Simulation Sources):
   - `weights_*.memb` for `$readmemb` in `projection_engine`
   - `centroids_*.memh` for `$readmemh` in `centroid_bram`
   - `testcases_*.memh` for `$readmemh` in `tb_ucihar_core` (layout: expected class, then 561 int16 features)
2. Set the simulation top to `tb_ucihar_core`.
3. Update `tb_ucihar_core` parameters `WEIGHT_INIT_FILE`, `CENTROID_INIT_FILE`, `TESTCASE_FILE`, and `NUM_CASES` as needed.
4. Run simulation and check `PASS/FAIL` summary at the end.

For a ready-to-run example (D=1000, seed=123), this folder already includes:
- `weights_D1000_seed123.memb`
- `centroids_D1000_seed123.memh`
- `testcases_D1000_seed123_n5_off0.memh`
