---
name: ISSUE TEMPLATE
about: Bug and Feature Report

---

Before you open an issue, please make sure you have tried the following steps:

1. Make sure your ENV is the same with (https://mace.readthedocs.io/en/latest/installation/env_requirement.html).
2. Have you ever read the document for your usage?
3. The form below must be filled.

------------------------

### System information
- **OS Platform and Distribution (e.g., Linux Ubuntu 16.04)**:
- **NDK version(e.g. 15c)**:
- **MACE version (Use the command: git describe --long --tags)**:
- **Python version(2.7)**: 
- **Bazel version (e.g. 0.13.0)**:

### Model deploy file (*.yml)
```yaml
......
```

### Describe the problem
A clear and concise description of what the bug is.

### To Reproduce
Steps to reproduce the behavior:
```bash
1. cd /path/to/mace
2. python tools/converter.py convert --config_file=/path/to/your/model_deployment_file
```

### Error information / logs
```bash
LOGs
```

### Additional context
Add any other context about the problem here.
