| Program | Formats | GPU codegen |
| --- | --- | --- |
| SpMV | CSR matrix, dense vector and output | Enabled |
| SpGEMM | CSR inputs and output | Skipped: new sparse output construction disables parallel loops |
| SDDMM | CSR sampling matrix and output, dense factors | Skipped: new sparse output construction disables parallel loops |
| MTTKRP | Fully compressed CSF tensor, dense factors and output | Enabled for the dense rank loop; sparse traversal stays on the host |
